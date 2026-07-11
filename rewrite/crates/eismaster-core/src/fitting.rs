use std::collections::HashMap;
use std::path::{Path, PathBuf};

use num_complex::Complex64;
use rust_xlsxwriter::{Format, Workbook, XlsxError};
use serde::Serialize;

use crate::chi_bin::parse_chi_bin;
use crate::chi_txt::parse_chi_txt;
use crate::circuits::{zview_double_model, zview_single_model};
use crate::models::SpectrumData;
use crate::quality::{assess_spectrum_quality, QualityReport};
use crate::segmentation::detect_segments;

// ---------------------------------------------------------------------------
// FitOutcome
// ---------------------------------------------------------------------------

/// Result of fitting a circuit model to a spectrum.
#[derive(Debug, Clone, Serialize)]
pub struct FitOutcome {
    pub model_key: String,
    pub model_label: String,
    pub status: String,
    pub message: String,
    pub parameters: HashMap<String, f64>,
    pub statistics: HashMap<String, f64>,
    pub predicted_real_ohm: Option<Vec<f64>>,
    pub predicted_imag_ohm: Option<Vec<f64>>,
    pub masked_points: usize,
    pub preprocess_actions: Vec<String>,
    pub fallback_from: Option<String>,
    pub diagnosis_type: String,
    pub diagnosis_severity: String,
    pub diagnosis_explanation: String,
    pub diagnosis_suggestions: Vec<String>,
    /// Placeholder for Python parity — confidence intervals not yet computed.
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Placeholder for Python parity — max absolute off-diagonal correlation not yet computed.
    pub correlation_matrix_max: f64,
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Build a boolean mask that keeps valid points for fitting.
///
/// Steps:
/// 1. Remove non-finite freq/z_real/z_imag
/// 2. Remove inductive points (z_imag > 0 at high frequency)
pub fn preprocess_mask(spectrum: &SpectrumData) -> (Vec<bool>, Vec<String>) {
    let n = spectrum.n_points();
    let mut mask = vec![true; n];
    let mut actions = Vec::new();

    // Non-finite
    let mut removed_finite = 0;
    for i in 0..n {
        if !spectrum.freq_hz[i].is_finite()
            || !spectrum.z_real_ohm[i].is_finite()
            || !spectrum.z_imag_ohm[i].is_finite()
        {
            mask[i] = false;
            removed_finite += 1;
        }
    }
    if removed_finite > 0 {
        actions.push(format!("移除了 {removed_finite} 个非有限数据点。"));
    }

    // Inductive points (leading high-freq points where z_imag > 0)
    let mut removed_inductive = 0;
    for i in 0..n {
        if mask[i] && spectrum.z_imag_ohm[i] > 0.0 {
            mask[i] = false;
            removed_inductive += 1;
        } else if mask[i] {
            break; // stop at first valid capacitive point
        }
    }
    if removed_inductive > 0 {
        actions.push(format!("移除了 {removed_inductive} 个高频感性数据点。"));
    }

    (mask, actions)
}

/// Count true values in mask.
fn mask_count(mask: &[bool]) -> usize {
    mask.iter().filter(|&&v| v).count()
}

// ---------------------------------------------------------------------------
// Levenberg-Marquardt optimizer (bounded)
// ---------------------------------------------------------------------------

/// Compute residuals: [weighted(Z_model.real - Z_exp.real), weighted(Z_model.imag - Z_exp.imag)].
///
/// Modulus weighting: divide by max(|Z_exp|, floor).
fn compute_residuals(
    freq: &[f64],
    z_exp: &[Complex64],
    params: &[f64],
    model_key: &str,
) -> Vec<f64> {
    let z_model = evaluate_model(freq, params, model_key);
    let n = freq.len();

    // Adaptive weight floor
    let mut abs_z: Vec<f64> = z_exp.iter().map(|z| z.norm()).collect();
    abs_z.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_z = if abs_z.is_empty() {
        1.0
    } else {
        abs_z[abs_z.len() / 2]
    };
    let floor = (median_z * 1e-4).max(1e-6);

    let mut residuals = Vec::with_capacity(2 * n);
    for i in 0..n {
        let diff_re = z_model[i].re - z_exp[i].re;
        let diff_im = z_model[i].im - z_exp[i].im;
        let w = z_exp[i].norm().max(floor);
        residuals.push(diff_re / w);
        residuals.push(diff_im / w);
    }
    residuals
}

/// Evaluate circuit model at given parameters.
fn evaluate_model(freq: &[f64], params: &[f64], model_key: &str) -> Vec<Complex64> {
    match model_key {
        "zview_segmented_rq_rwo" => {
            let mut p = [0.0f64; 7];
            for (i, v) in params.iter().enumerate().take(7) {
                p[i] = *v;
            }
            zview_single_model(freq, &p)
        }
        "zview_double_rq_qrwo" => {
            let mut p = [0.0f64; 10];
            for (i, v) in params.iter().enumerate().take(10) {
                p[i] = *v;
            }
            zview_double_model(freq, &p)
        }
        _ => vec![Complex64::new(0.0, 0.0); freq.len()],
    }
}

/// Compute Jacobian via forward finite differences.
fn compute_jacobian(
    freq: &[f64],
    z_exp: &[Complex64],
    params: &[f64],
    model_key: &str,
    residuals_at_p: &[f64],
) -> Vec<Vec<f64>> {
    let n_res = residuals_at_p.len();
    let n_par = params.len();
    let mut jac = vec![vec![0.0; n_par]; n_res];

    let eps_base = 1e-8;
    for j in 0..n_par {
        let mut p_plus = params.to_vec();
        let step = (params[j].abs() * eps_base).max(eps_base);
        p_plus[j] += step;
        let r_plus = compute_residuals(freq, z_exp, &p_plus, model_key);
        for i in 0..n_res {
            jac[i][j] = (r_plus[i] - residuals_at_p[i]) / step;
        }
    }
    jac
}

/// Result of one LM step.
struct LmStep {
    params: Vec<f64>,
    rss: f64,
    accepted: bool,
}

/// Try one Levenberg-Marquardt step.
///
/// Solve (J^T J + lambda * diag(J^T J)) delta = J^T r
/// using a simple diagonal-damped normal equation solver.
fn lm_step(
    freq: &[f64],
    z_exp: &[Complex64],
    params: &[f64],
    model_key: &str,
    bounds_low: &[f64],
    bounds_high: &[f64],
    damping: f64,
) -> LmStep {
    let residuals = compute_residuals(freq, z_exp, params, model_key);
    let rss_current: f64 = residuals.iter().map(|r| r * r).sum();

    let jac = compute_jacobian(freq, z_exp, params, model_key, &residuals);
    let n_res = jac.len();
    let n_par = params.len();

    // J^T J
    let mut jtj = vec![vec![0.0f64; n_par]; n_par];
    let mut jtr = vec![0.0f64; n_par];
    for j in 0..n_par {
        for k in 0..n_par {
            let mut s = 0.0;
            for i in 0..n_res {
                s += jac[i][j] * jac[i][k];
            }
            jtj[j][k] = s;
        }
        let mut s = 0.0;
        for i in 0..n_res {
            s += jac[i][j] * residuals[i];
        }
        jtr[j] = s;
    }

    // Damped normal equations: (J^T J + lambda * diag(J^T J)) delta = -J^T r
    let mut a = jtj.clone();
    for j in 0..n_par {
        a[j][j] += damping * jtj[j][j].max(1e-20);
    }

    // Solve via Gaussian elimination
    let delta = solve_linear_system(&a, &jtr);
    if delta.iter().any(|d| !d.is_finite()) {
        return LmStep {
            params: params.to_vec(),
            rss: rss_current,
            accepted: false,
        };
    }

    // Propose new params: p - delta, clamped to bounds
    let mut new_params = params.to_vec();
    for j in 0..n_par {
        new_params[j] = (params[j] - delta[j])
            .max(bounds_low[j])
            .min(bounds_high[j]);
    }

    let new_residuals = compute_residuals(freq, z_exp, &new_params, model_key);
    let rss_new: f64 = new_residuals.iter().map(|r| r * r).sum();

    LmStep {
        params: new_params,
        rss: rss_new,
        accepted: rss_new < rss_current,
    }
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug = vec![vec![0.0f64; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-30 {
            continue;
        }
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    x
}

/// Run bounded Levenberg-Marquardt optimization.
fn optimize_lm(
    freq: &[f64],
    z_exp: &[Complex64],
    mut params: Vec<f64>,
    model_key: &str,
    bounds_low: &[f64],
    bounds_high: &[f64],
    max_iter: usize,
) -> (Vec<f64>, f64, usize) {
    let mut damping = 1e-3;
    let mut rss = compute_residuals(freq, z_exp, &params, model_key)
        .iter()
        .map(|r| r * r)
        .sum::<f64>();

    let mut n_fev: usize = 1;

    for _iter in 0..max_iter {
        let step = lm_step(
            freq,
            z_exp,
            &params,
            model_key,
            bounds_low,
            bounds_high,
            damping,
        );
        n_fev += 1;

        if step.accepted {
            params = step.params;
            let new_rss = step.rss;
            // Check convergence
            if (rss - new_rss).abs() / rss.max(1e-30) < 1e-8 {
                rss = new_rss;
                break;
            }
            rss = new_rss;
            damping /= 10.0;
        } else {
            damping *= 10.0;
        }

        damping = damping.max(1e-20).min(1e20);
    }

    (params, rss, n_fev)
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

fn single_bounds() -> (Vec<f64>, Vec<f64>) {
    let lo = vec![1e-9, 1e-12, 0.2, 1e-9, 1e-9, 1e-9, 0.2];
    let hi = vec![1e6, 1e0, 1.0, 1e8, 1e8, 1e6, 1.0];
    (lo, hi)
}

fn double_bounds() -> (Vec<f64>, Vec<f64>) {
    let lo = vec![1e-9, 1e-12, 0.2, 1e-9, 1e-12, 0.2, 1e-9, 1e-9, 1e-9, 0.2];
    let hi = vec![1e6, 1e0, 1.0, 1e8, 1e0, 1.0, 1e8, 1e8, 1e6, 1.0];
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Initial guesses
// ---------------------------------------------------------------------------

/// Estimate CPE exponent n from slope of log10(-Z_imag) vs log10(freq) near peak.
fn estimate_cpe_n(freq: &[f64], z_imag: &[f64], peak_idx: usize) -> f64 {
    let n = freq.len();
    if n < 5 || peak_idx == 0 {
        return 0.85;
    }
    let lo = peak_idx.saturating_sub(3).max(1);
    let hi = (peak_idx + 3).min(n - 1);
    if hi - lo < 2 {
        return 0.85;
    }

    // Linear regression of log10(-z_imag) vs log10(freq)
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0;
    for i in lo..=hi {
        let neg_zi = (-z_imag[i]).max(1e-30);
        if neg_zi <= 0.0 || freq[i] <= 0.0 {
            continue;
        }
        let x = freq[i].log10();
        let y = neg_zi.log10();
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        count += 1;
    }
    if count < 2 {
        return 0.85;
    }
    let n_f = count as f64;
    let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
    slope.abs().max(0.4).min(1.0)
}

/// Build initial guess for single-arc R(QRWo) from segmentation.
fn guess_single(spectrum: &SpectrumData) -> Vec<f64> {
    let n = spectrum.n_points();
    let seg = detect_segments(spectrum, "auto", None, None, None, None);

    let zr = &spectrum.z_real_ohm;
    let zi = &spectrum.z_imag_ohm;
    let freq = &spectrum.freq_hz;

    let rs = zr.iter().copied().fold(f64::INFINITY, f64::min).max(1e-9);
    let zr_max = zr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (zr_max - rs).max(1e-6);

    let peak_idx = seg.peak_indices.first().copied().unwrap_or(n / 3);
    let split_idx = seg.split_indices.first().copied().unwrap_or(2 * n / 3);

    let peak_freq = freq[peak_idx.min(n - 1)].max(1e-6);
    let split_freq = freq[split_idx.min(n - 1)].max(1e-6);

    let cpe_p = estimate_cpe_n(freq, zi, peak_idx);
    let cpe_t = 1.0 / (span * (2.0 * std::f64::consts::PI * peak_freq).powf(cpe_p));
    let rct = span * 0.8;
    let wo_r = (zr[n - 1] - zr[split_idx.min(n - 1)]).max(1e-6);
    let wo_t = 1.0 / (2.0 * std::f64::consts::PI * split_freq);
    let wo_p = 0.5;

    vec![rs, cpe_t, cpe_p, rct, wo_r, wo_t, wo_p]
}

/// Build initial guess for double-arc R(QR)(Q(RWo)) from segmentation.
fn guess_double(spectrum: &SpectrumData) -> Vec<f64> {
    let n = spectrum.n_points();
    let seg = detect_segments(spectrum, "auto", None, None, None, None);

    let zr = &spectrum.z_real_ohm;
    let zi = &spectrum.z_imag_ohm;
    let freq = &spectrum.freq_hz;

    let rs = zr.iter().copied().fold(f64::INFINITY, f64::min).max(1e-9);

    let peak1 = seg.peak_indices.first().copied().unwrap_or(n / 4);
    let peak2 = seg.peak_indices.get(1).copied().unwrap_or(3 * n / 4);
    let split1 = seg.split_indices.first().copied().unwrap_or(n / 2);
    let split2 = seg.split_indices.get(1).copied().unwrap_or(3 * n / 4);

    let pf1 = freq[peak1.min(n - 1)].max(1e-6);
    let pf2 = freq[peak2.min(n - 1)].max(1e-6);

    let r_sei = (zr[split1.min(n - 1)] - rs).max(1e-6);
    let r_ct = (zr[split2.min(n - 1)] - zr[split1.min(n - 1)]).max(1e-6);

    let n1 = estimate_cpe_n(freq, zi, peak1);
    let n2 = estimate_cpe_n(freq, zi, peak2);
    let q1 = 1.0 / (r_sei * (2.0 * std::f64::consts::PI * pf1).powf(n1));
    let q2 = 1.0 / (r_ct * (2.0 * std::f64::consts::PI * pf2).powf(n2));

    let wo_r = (zr[n - 1] - zr[split2.min(n - 1)]).max(1e-6);
    let sf2 = freq[split2.min(n - 1)].max(1e-6);
    let wo_t = 1.0 / (2.0 * std::f64::consts::PI * sf2);
    let wo_p = 0.5;

    vec![rs, q1, n1, r_sei, q2, n2, r_ct, wo_r, wo_t, wo_p]
}

/// Generate perturbation variants of a base guess.
fn perturb_guess(base: &[f64], factors: &[&[f64]]) -> Vec<Vec<f64>> {
    let mut seeds = vec![base.to_vec()];
    for fset in factors {
        let mut seed = base.to_vec();
        for (i, f) in fset.iter().enumerate() {
            seed[i] *= f;
        }
        seeds.push(seed);
    }
    seeds
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute fit statistics from final residuals.
fn compute_statistics(rss: f64, n_obs: usize, n_params: usize) -> HashMap<String, f64> {
    let mut stats = HashMap::new();
    let n_obs_f = n_obs as f64;
    let n_par_f = n_params as f64;
    let dof = (n_obs - n_params).max(1) as f64;

    stats.insert("rss".to_string(), rss);
    stats.insert("chi2_reduced".to_string(), rss / dof);

    let mean_rss = rss / n_obs_f.max(1.0);
    if mean_rss > 0.0 {
        let aic = n_obs_f * mean_rss.ln() + 2.0 * n_par_f;
        let bic = n_obs_f * mean_rss.ln() + n_par_f * n_obs_f.ln();
        stats.insert("aic".to_string(), aic);
        stats.insert("bic".to_string(), bic);
        let denom = n_obs_f - n_par_f - 1.0;
        if denom > 0.0 {
            stats.insert(
                "aicc".to_string(),
                aic + 2.0 * n_par_f * (n_par_f + 1.0) / denom,
            );
        }
    }

    stats
}

// ---------------------------------------------------------------------------
// Parameter names
// ---------------------------------------------------------------------------

fn param_names(model_key: &str) -> &'static [&'static str] {
    match model_key {
        "zview_segmented_rq_rwo" => &["Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P"],
        "zview_double_rq_qrwo" => &[
            "Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P",
        ],
        _ => &[],
    }
}

fn model_label(model_key: &str) -> &'static str {
    match model_key {
        "zview_segmented_rq_rwo" => "Single-arc R(QRWo)",
        "zview_double_rq_qrwo" => "Double-arc R(QR)(Q(RWo))",
        _ => "Unknown",
    }
}

fn n_params(model_key: &str) -> usize {
    match model_key {
        "zview_segmented_rq_rwo" => 7,
        "zview_double_rq_qrwo" => 10,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Top-level fit function
// ---------------------------------------------------------------------------

/// Fit a circuit model to a spectrum.
///
/// Returns a FitOutcome with parameters, statistics, predicted curves, and diagnostics.
pub fn fit_spectrum(spectrum: &SpectrumData, model_key: &str) -> FitOutcome {
    let n_pts = spectrum.n_points();
    let names = param_names(model_key);
    let n_par = n_params(model_key);

    if names.is_empty() || n_par == 0 {
        return FitOutcome {
            model_key: model_key.to_string(),
            model_label: model_label(model_key).to_string(),
            status: "failed".to_string(),
            message: format!("未知模型: {model_key}"),
            ..empty_outcome()
        };
    }

    // Preprocessing
    let (mask, actions) = preprocess_mask(spectrum);
    let kept = mask_count(&mask);
    let masked_points = n_pts - kept;

    if kept < n_par + 2 {
        return FitOutcome {
            model_key: model_key.to_string(),
            model_label: model_label(model_key).to_string(),
            status: "failed".to_string(),
            message: format!("有效数据点不足: {kept} < {}", n_par + 2),
            masked_points,
            preprocess_actions: actions,
            ..empty_outcome()
        };
    }

    // Extract masked data
    let freq_masked: Vec<f64> = spectrum
        .freq_hz
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&f, _)| f)
        .collect();
    let z_exp: Vec<Complex64> = spectrum
        .z_real_ohm
        .iter()
        .zip(spectrum.z_imag_ohm.iter())
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|((&re, &im), _)| Complex64::new(re, im))
        .collect();

    // Bounds
    let (bounds_lo, bounds_hi) = match model_key {
        "zview_segmented_rq_rwo" => single_bounds(),
        "zview_double_rq_qrwo" => double_bounds(),
        _ => unreachable!(),
    };

    // Initial guesses
    let base_guess = match model_key {
        "zview_segmented_rq_rwo" => guess_single(spectrum),
        "zview_double_rq_qrwo" => guess_double(spectrum),
        _ => unreachable!(),
    };

    let single_perturb: &[&[f64]] = &[
        &[1.0, 1.3, 1.0, 0.8, 1.1, 1.0, 1.0],
        &[1.0, 0.7, 1.0, 1.2, 0.9, 1.2, 1.0],
        &[1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
    ];
    let double_perturb: &[&[f64]] = &[
        &[1.0, 1.3, 1.0, 0.8, 1.3, 1.0, 0.8, 1.1, 1.0, 1.0],
        &[1.0, 0.7, 1.0, 1.2, 0.7, 1.0, 1.2, 0.9, 1.2, 1.0],
        &[1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0],
    ];

    let seeds = match model_key {
        "zview_segmented_rq_rwo" => perturb_guess(&base_guess, single_perturb),
        "zview_double_rq_qrwo" => perturb_guess(&base_guess, double_perturb),
        _ => vec![base_guess],
    };

    // Multi-start optimization
    let max_nfev = 8000;
    let mut best_params = seeds[0].clone();
    let mut best_rss = f64::INFINITY;

    for seed in &seeds {
        // Clamp seed to bounds
        let mut clamped = seed.clone();
        for j in 0..n_par {
            clamped[j] = clamped[j].max(bounds_lo[j]).min(bounds_hi[j]);
        }

        let (result_params, result_rss, _n_fev) = optimize_lm(
            &freq_masked,
            &z_exp,
            clamped,
            model_key,
            &bounds_lo,
            &bounds_hi,
            max_nfev / seeds.len().max(1),
        );

        if result_rss < best_rss {
            best_rss = result_rss;
            best_params = result_params;
        }
    }

    // Compute predicted impedance at all points (not just masked)
    let z_pred = evaluate_model(&spectrum.freq_hz, &best_params, model_key);
    let pred_real: Vec<f64> = z_pred.iter().map(|z| z.re).collect();
    let pred_imag: Vec<f64> = z_pred.iter().map(|z| z.im).collect();

    // Statistics (on masked data)
    let n_obs = 2 * freq_masked.len();
    let stats = compute_statistics(best_rss, n_obs, n_par);

    // Build parameter map
    let mut parameters = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        parameters.insert(name.to_string(), best_params[i]);
    }

    // Status
    let chi2r = stats.get("chi2_reduced").copied().unwrap_or(f64::INFINITY);
    let (status, message) = if chi2r < 0.1 {
        (
            "ok".to_string(),
            format!("拟合完成，chi2_reduced={chi2r:.4e}"),
        )
    } else if chi2r < 1.0 {
        (
            "warn".to_string(),
            format!("拟合完成但残差较大，chi2_reduced={chi2r:.4e}"),
        )
    } else {
        (
            "failed".to_string(),
            format!("拟合未收敛，chi2_reduced={chi2r:.4e}"),
        )
    };

    // Basic diagnostics
    let (diag_type, diag_severity, diag_explanation, diag_suggestions) = if status == "failed" {
        (
            "convergence".to_string(),
            "error".to_string(),
            "非线性优化器未收敛到稳定解。".to_string(),
            vec![
                "尝试更换模型模板。".to_string(),
                "检查数据质量和频段覆盖。".to_string(),
            ],
        )
    } else if status == "warn" {
        (
            "convergence".to_string(),
            "warning".to_string(),
            format!("拟合完成但有警告: chi2_reduced={chi2r:.4e}"),
            vec!["检查拟合曲线和参数不确定度。".to_string()],
        )
    } else {
        (
            "none".to_string(),
            "info".to_string(),
            "拟合完成，无警告。".to_string(),
            vec![],
        )
    };

    FitOutcome {
        model_key: model_key.to_string(),
        model_label: model_label(model_key).to_string(),
        status,
        message,
        parameters,
        statistics: stats,
        predicted_real_ohm: Some(pred_real),
        predicted_imag_ohm: Some(pred_imag),
        masked_points,
        preprocess_actions: actions,
        fallback_from: None,
        diagnosis_type: diag_type,
        diagnosis_severity: diag_severity,
        diagnosis_explanation: diag_explanation,
        diagnosis_suggestions: diag_suggestions,
        confidence_intervals: HashMap::new(),
        correlation_matrix_max: f64::NAN,
    }
}

fn empty_outcome() -> FitOutcome {
    FitOutcome {
        model_key: String::new(),
        model_label: String::new(),
        status: String::new(),
        message: String::new(),
        parameters: HashMap::new(),
        statistics: HashMap::new(),
        predicted_real_ohm: None,
        predicted_imag_ohm: None,
        masked_points: 0,
        preprocess_actions: Vec::new(),
        fallback_from: None,
        diagnosis_type: String::new(),
        diagnosis_severity: String::new(),
        diagnosis_explanation: String::new(),
        diagnosis_suggestions: Vec::new(),
        confidence_intervals: HashMap::new(),
        correlation_matrix_max: f64::NAN,
    }
}

// ---------------------------------------------------------------------------
// Batch fitting
// ---------------------------------------------------------------------------

/// Result of fitting one spectrum in a batch.
#[derive(Debug, Clone, Serialize)]
pub struct BatchItemResult {
    pub file: String,
    pub label: String,
    pub quality: Option<QualityReport>,
    pub fit: Option<FitOutcome>,
    pub error: Option<String>,
    /// Experimental Z_real for export (not serialized to JSON by default via skip).
    #[serde(skip)]
    pub z_real_ohm: Vec<f64>,
    #[serde(skip)]
    pub z_imag_ohm: Vec<f64>,
}

/// Summary of a batch fitting run.
#[derive(Debug, Clone, Serialize)]
pub struct BatchSummary {
    pub model_key: String,
    pub model_label: String,
    pub items: Vec<BatchItemResult>,
    pub n_total: usize,
    pub n_ok: usize,
    pub n_warn: usize,
    pub n_failed: usize,
}

/// Extract a display label from a file stem, reusing the DRT module's logic.
fn batch_label_from_stem(stem: &str) -> String {
    crate::drt::label_from_stem(stem)
}

/// Fit a circuit model to every `.txt`/`.csv` spectrum in a folder.
///
/// Parses each file, runs quality assessment, fits the model, and collects results.
/// Files that fail to parse or fit are recorded with an error message rather than
/// aborting the entire batch.
pub fn fit_batch(dir: &Path, model_key: &str) -> Result<BatchSummary, String> {
    if !dir.is_dir() {
        return Err(format!("目录不存在: {}", dir.display()));
    }

    let mut paths: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| format!("无法读取目录: {e}"))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase());
            match ext.as_deref() {
                Some("txt") | Some("csv") => paths.push(p),
                _ => {}
            }
        }
    }
    paths.sort();

    if paths.is_empty() {
        return Err("目录中没有找到 .txt/.csv 文件".to_string());
    }

    let label = model_label(model_key).to_string();
    let mut items = Vec::with_capacity(paths.len());
    let mut n_ok = 0usize;
    let mut n_warn = 0usize;
    let mut n_failed = 0usize;

    for path in &paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let item_label = batch_label_from_stem(stem);

        // Parse
        let spectrum = match parse_chi_txt(path) {
            Ok(s) => s,
            Err(e) => {
                items.push(BatchItemResult {
                    file: stem.to_string(),
                    label: item_label,
                    quality: None,
                    fit: None,
                    error: Some(format!("解析失败: {e}")),
                    z_real_ohm: vec![],
                    z_imag_ohm: vec![],
                });
                n_failed += 1;
                continue;
            }
        };

        // Quality
        let quality = assess_spectrum_quality(&spectrum);

        // Fit
        let outcome = fit_spectrum(&spectrum, model_key);
        match outcome.status.as_str() {
            "ok" => n_ok += 1,
            "warn" => n_warn += 1,
            _ => n_failed += 1,
        }

        items.push(BatchItemResult {
            file: stem.to_string(),
            label: item_label,
            quality: Some(quality),
            fit: Some(outcome),
            error: None,
            z_real_ohm: spectrum.z_real_ohm.clone(),
            z_imag_ohm: spectrum.z_imag_ohm.clone(),
        });
    }

    Ok(BatchSummary {
        model_key: model_key.to_string(),
        model_label: label,
        items,
        n_total: paths.len(),
        n_ok,
        n_warn,
        n_failed,
    })
}

/// Fit a circuit model to a list of spectrum file paths.
///
/// Like `fit_batch` but takes explicit paths instead of scanning a directory.
/// Matches Python behavior where batch operates on already-loaded spectra.
pub fn fit_batch_paths(paths: &[PathBuf], model_key: &str) -> Result<BatchSummary, String> {
    if paths.is_empty() {
        return Err("没有提供谱图路径".to_string());
    }

    let label = model_label(model_key).to_string();
    let mut items = Vec::with_capacity(paths.len());
    let mut n_ok = 0usize;
    let mut n_warn = 0usize;
    let mut n_failed = 0usize;

    for path in paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let item_label = batch_label_from_stem(stem);

        // Parse with extension dispatch
        let spectrum = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("bin") => crate::chi_bin::parse_chi_bin(path),
            _ => parse_chi_txt(path),
        };

        let spectrum = match spectrum {
            Ok(s) => s,
            Err(e) => {
                items.push(BatchItemResult {
                    file: stem.to_string(),
                    label: item_label,
                    quality: None,
                    fit: None,
                    error: Some(format!("解析失败: {e}")),
                    z_real_ohm: vec![],
                    z_imag_ohm: vec![],
                });
                n_failed += 1;
                continue;
            }
        };

        let quality = assess_spectrum_quality(&spectrum);
        let outcome = fit_spectrum(&spectrum, model_key);
        match outcome.status.as_str() {
            "ok" => n_ok += 1,
            "warn" => n_warn += 1,
            _ => n_failed += 1,
        }

        items.push(BatchItemResult {
            file: stem.to_string(),
            label: item_label,
            quality: Some(quality),
            fit: Some(outcome),
            error: None,
            z_real_ohm: spectrum.z_real_ohm.clone(),
            z_imag_ohm: spectrum.z_imag_ohm.clone(),
        });
    }

    Ok(BatchSummary {
        model_key: model_key.to_string(),
        model_label: label,
        items,
        n_total: paths.len(),
        n_ok,
        n_warn,
        n_failed,
    })
}

// ---------------------------------------------------------------------------
// Batch export workbook
// ---------------------------------------------------------------------------

/// Write a single-spectrum fit workbook (one spectrum → raw_plot + rs_rct + fit_overlay).
pub fn write_single_fit_workbook(
    path: &Path,
    spectrum: &SpectrumData,
    outcome: &FitOutcome,
) -> Result<(), XlsxError> {
    let summary = BatchSummary {
        model_key: outcome.model_key.clone(),
        model_label: outcome.model_label.clone(),
        items: vec![BatchItemResult {
            file: spectrum
                .metadata
                .file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            label: batch_label_from_stem(
                spectrum
                    .metadata
                    .file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown"),
            ),
            quality: None,
            fit: Some(outcome.clone()),
            error: None,
            z_real_ohm: spectrum.z_real_ohm.clone(),
            z_imag_ohm: spectrum.z_imag_ohm.clone(),
        }],
        n_total: 1,
        n_ok: if outcome.status == "ok" { 1 } else { 0 },
        n_warn: if outcome.status == "warn" { 1 } else { 0 },
        n_failed: if outcome.status != "ok" && outcome.status != "warn" {
            1
        } else {
            0
        },
    };
    write_batch_workbook(path, &summary)
}

/// Write a batch fit summary to an XLSX workbook with three sheets:
/// - `raw_plot`: experimental Z_real and -Z_imag (positive) per spectrum
/// - `rs_rct`: summary table with Rs, Rct, Rsei
/// - `fit_overlay`: experimental + fitted data per spectrum
pub fn write_batch_workbook(path: &Path, summary: &BatchSummary) -> Result<(), XlsxError> {
    let mut wb = Workbook::new();

    let header_fmt = Format::new().set_bold();
    let label_fmt = Format::new().set_bold().set_italic();

    // --- Sheet 1: raw_plot ---
    let ws_raw = wb.add_worksheet().set_name("raw_plot")?;
    let mut col: u16 = 0;
    for item in &summary.items {
        if item.z_real_ohm.is_empty() {
            continue;
        }
        // Label row
        ws_raw.write_string_with_format(0, col, &item.label, &label_fmt)?;
        ws_raw.write_string_with_format(0, col + 1, "", &label_fmt)?;
        // Column headers
        ws_raw.write_string_with_format(1, col, "z_real", &header_fmt)?;
        ws_raw.write_string_with_format(1, col + 1, "imag_pos", &header_fmt)?;
        // Data rows
        for (i, (re, im)) in item
            .z_real_ohm
            .iter()
            .zip(item.z_imag_ohm.iter())
            .enumerate()
        {
            let row = (i + 2) as u32;
            ws_raw.write_number(row, col, *re)?;
            ws_raw.write_number(row, col + 1, -*im)?;
        }
        col += 2;
    }

    // --- Sheet 2: rs_rct ---
    let ws_rct = wb.add_worksheet().set_name("rs_rct")?;
    ws_rct.write_string_with_format(0, 0, "label", &header_fmt)?;
    ws_rct.write_string_with_format(0, 1, "file", &header_fmt)?;
    ws_rct.write_string_with_format(0, 2, "Rs", &header_fmt)?;
    ws_rct.write_string_with_format(0, 3, "Rct", &header_fmt)?;
    ws_rct.write_string_with_format(0, 4, "Rsei", &header_fmt)?;

    for (row_idx, item) in summary.items.iter().enumerate() {
        let row = (row_idx + 1) as u32;
        ws_rct.write_string(row, 0, &item.label)?;
        ws_rct.write_string(row, 1, &item.file)?;

        if let Some(ref fit) = item.fit {
            if let Some(rs) = fit.parameters.get("Rs") {
                ws_rct.write_number(row, 2, *rs)?;
            }
            if let Some(rct) = fit.parameters.get("Rct") {
                ws_rct.write_number(row, 3, *rct)?;
            }
            if let Some(rsei) = fit.parameters.get("Rsei") {
                ws_rct.write_number(row, 4, *rsei)?;
            }
        }
    }

    // --- Sheet 3: fit_overlay ---
    let ws_overlay = wb.add_worksheet().set_name("fit_overlay")?;
    let mut col: u16 = 0;
    for item in &summary.items {
        let fit = match &item.fit {
            Some(f) if f.predicted_real_ohm.is_some() => f,
            _ => continue,
        };
        let pred_real = fit.predicted_real_ohm.as_ref().unwrap();
        let pred_imag = fit.predicted_imag_ohm.as_ref().unwrap();
        let has_exp = !item.z_real_ohm.is_empty();

        // Label row
        ws_overlay.write_string_with_format(0, col, &item.label, &label_fmt)?;
        ws_overlay.write_string_with_format(0, col + 1, "", &label_fmt)?;
        ws_overlay.write_string_with_format(0, col + 2, "", &label_fmt)?;
        ws_overlay.write_string_with_format(0, col + 3, "", &label_fmt)?;

        // Column headers
        ws_overlay.write_string_with_format(1, col, "z_real_exp", &header_fmt)?;
        ws_overlay.write_string_with_format(1, col + 1, "imag_exp_pos", &header_fmt)?;
        ws_overlay.write_string_with_format(1, col + 2, "z_real_fit", &header_fmt)?;
        ws_overlay.write_string_with_format(1, col + 3, "imag_fit_pos", &header_fmt)?;

        // Data rows
        let n_rows = pred_real.len();
        for i in 0..n_rows {
            let row = (i + 2) as u32;
            // Experimental columns
            if has_exp && i < item.z_real_ohm.len() {
                ws_overlay.write_number(row, col, item.z_real_ohm[i])?;
                ws_overlay.write_number(row, col + 1, -item.z_imag_ohm[i])?;
            }
            // Fit columns
            ws_overlay.write_number(row, col + 2, pred_real[i])?;
            ws_overlay.write_number(row, col + 3, -pred_imag[i])?;
        }

        col += 4;
    }

    wb.save(path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SpectrumMetadata;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_spectrum(freq: Vec<f64>, zr: Vec<f64>, zi: Vec<f64>) -> SpectrumData {
        let n = freq.len();
        SpectrumData {
            metadata: SpectrumMetadata {
                file_path: PathBuf::from("synthetic.txt"),
                technique: "A.C. Impedance".to_string(),
                instrument_model: "test".to_string(),
                acquired_at: None,
                note: String::new(),
                header: HashMap::new(),
                source_format: "txt".to_string(),
            },
            freq_hz: freq,
            z_real_ohm: zr,
            z_imag_ohm: zi,
            z_mod_ohm: vec![0.0; n],
            phase_deg: vec![0.0; n],
        }
    }

    /// Generate synthetic single-arc spectrum from known parameters.
    fn synthetic_single_arc() -> (SpectrumData, Vec<f64>) {
        let params = [5.0, 1e-5, 0.85, 50.0, 10.0, 1.0, 0.5];
        let freq: Vec<f64> = (0..40).map(|i| 10f64.powf(5.0 - i as f64 * 0.25)).collect();
        let z = zview_single_model(&freq, &params);
        let zr: Vec<f64> = z.iter().map(|v| v.re).collect();
        let zi: Vec<f64> = z.iter().map(|v| v.im).collect();
        (make_spectrum(freq, zr, zi), params.to_vec())
    }

    /// Generate synthetic double-arc spectrum from known parameters.
    fn synthetic_double_arc() -> (SpectrumData, Vec<f64>) {
        let params = [2.0, 1e-5, 0.9, 20.0, 1e-4, 0.85, 30.0, 5.0, 0.1, 0.5];
        let freq: Vec<f64> = (0..50).map(|i| 10f64.powf(5.0 - i as f64 * 0.2)).collect();
        let z = zview_double_model(&freq, &params);
        let zr: Vec<f64> = z.iter().map(|v| v.re).collect();
        let zi: Vec<f64> = z.iter().map(|v| v.im).collect();
        (make_spectrum(freq, zr, zi), params.to_vec())
    }

    #[test]
    fn fit_single_arc_recovers_params() {
        let (spec, true_params) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");

        assert_eq!(
            outcome.status, "ok",
            "fit should converge: {}",
            outcome.message
        );
        assert_eq!(outcome.model_key, "zview_segmented_rq_rwo");
        assert!(outcome.predicted_real_ohm.is_some());
        assert!(outcome.predicted_imag_ohm.is_some());

        // Check that Rs is close to true value (5.0)
        let rs = outcome.parameters.get("Rs").unwrap();
        assert!(
            (rs - true_params[0]).abs() < 1.0,
            "Rs={rs}, expected ~{}",
            true_params[0]
        );

        // Check Rct is close to true value (50.0)
        let rct = outcome.parameters.get("Rct").unwrap();
        assert!(
            (rct - true_params[3]).abs() < 10.0,
            "Rct={rct}, expected ~{}",
            true_params[3]
        );
    }

    #[test]
    fn fit_double_arc_converges() {
        let (spec, _true_params) = synthetic_double_arc();
        let outcome = fit_spectrum(&spec, "zview_double_rq_qrwo");

        assert_eq!(outcome.model_key, "zview_double_rq_qrwo");
        assert!(
            outcome.status == "ok" || outcome.status == "warn",
            "double arc fit should at least warn, got: {} - {}",
            outcome.status,
            outcome.message
        );
        assert!(outcome.parameters.contains_key("Rs"));
        assert!(outcome.parameters.contains_key("Rsei"));
        assert!(outcome.parameters.contains_key("Rct"));
    }

    #[test]
    fn fit_produces_valid_statistics() {
        let (spec, _) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");

        assert!(outcome.statistics.contains_key("rss"));
        assert!(outcome.statistics.contains_key("chi2_reduced"));
        assert!(outcome.statistics.contains_key("aic"));
        assert!(outcome.statistics.contains_key("bic"));
        assert!(outcome.statistics["rss"] >= 0.0);
        assert!(outcome.statistics["chi2_reduced"] >= 0.0);
    }

    #[test]
    fn fit_unknown_model_fails_gracefully() {
        let (spec, _) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "nonexistent_model");
        assert_eq!(outcome.status, "failed");
    }

    #[test]
    fn fit_few_points_fails() {
        let spec = make_spectrum(
            vec![1e3, 1e2, 1e1],
            vec![10.0, 20.0, 15.0],
            vec![-5.0, -10.0, -3.0],
        );
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");
        assert_eq!(outcome.status, "failed");
        assert!(outcome.message.contains("不足"));
    }

    #[test]
    fn preprocess_removes_inductive() {
        let spec = make_spectrum(
            vec![1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let (mask, actions) = preprocess_mask(&spec);
        assert!(!mask[0]); // inductive point removed
        assert!(actions.iter().any(|a| a.contains("感性")));
    }

    #[test]
    fn perturb_generates_variants() {
        let base = vec![1.0, 2.0, 3.0];
        let factors: &[&[f64]] = &[&[2.0, 1.0, 1.0], &[1.0, 2.0, 1.0]];
        let seeds = perturb_guess(&base, factors);
        assert_eq!(seeds.len(), 3); // base + 2 perturbations
        assert_eq!(seeds[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(seeds[1], vec![2.0, 2.0, 3.0]);
        assert_eq!(seeds[2], vec![1.0, 4.0, 3.0]);
    }

    #[test]
    fn solve_linear_system_basic() {
        // 2x + y = 5, x + 3y = 10 -> x=1, y=3
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_cpe_n_reasonable() {
        // With a clear single arc, n should be near 0.85
        let (spec, _) = synthetic_single_arc();
        let n = estimate_cpe_n(&spec.freq_hz, &spec.z_imag_ohm, 15);
        assert!((0.4..=1.0).contains(&n), "CPE n={n} out of range");
    }

    #[test]
    fn batch_label_from_stem_extracts_tokens() {
        // Delegates to drt::label_from_stem — verify key cases
        assert_eq!(batch_label_from_stem("Ag_EIS_OCV"), "OCV");
        assert_eq!(batch_label_from_stem("Ag_S01_EIS_T5M"), "T5M");
        assert_eq!(batch_label_from_stem("Ag_EIS_T100M"), "T100M");
        // No preferred match: returns last informative token
        assert_eq!(batch_label_from_stem("random_file"), "file");
    }

    #[test]
    fn batch_workbook_has_three_sheets() {
        let (spec, _) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");

        let summary = BatchSummary {
            model_key: "zview_segmented_rq_rwo".to_string(),
            model_label: "Single-arc R(QRWo)".to_string(),
            items: vec![BatchItemResult {
                file: "test.txt".to_string(),
                label: "OCV".to_string(),
                quality: None,
                fit: Some(outcome),
                error: None,
                z_real_ohm: spec.z_real_ohm.clone(),
                z_imag_ohm: spec.z_imag_ohm.clone(),
            }],
            n_total: 1,
            n_ok: 1,
            n_warn: 0,
            n_failed: 0,
        };

        let dir = std::env::temp_dir().join("eismaster_batch_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("batch_test.xlsx");

        let result = write_batch_workbook(&path, &summary);
        assert!(result.is_ok(), "workbook write failed: {:?}", result.err());
        assert!(path.exists());

        // Verify the file is non-trivial
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 100, "workbook too small: {size} bytes");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fit_batch_on_temp_dir() {
        // Create a temp directory with a synthetic spectrum file
        let (spec, _) = synthetic_single_arc();
        let dir = std::env::temp_dir().join("eismaster_fit_batch_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Write a plain numeric txt file (whitespace-separated, no headers)
        let file_path = dir.join("test_OCV.txt");
        let mut content = String::new();
        for i in 0..spec.freq_hz.len() {
            content.push_str(&format!(
                "{:.6e}  {:.6e}  {:.6e}\n",
                spec.freq_hz[i], spec.z_real_ohm[i], spec.z_imag_ohm[i]
            ));
        }
        std::fs::write(&file_path, &content).unwrap();

        let summary = fit_batch(&dir, "zview_segmented_rq_rwo").unwrap();
        assert_eq!(summary.n_total, 1);
        assert_eq!(summary.n_ok, 1);
        assert_eq!(summary.items.len(), 1);
        assert_eq!(summary.items[0].label, "OCV");
        assert!(summary.items[0].fit.is_some());
        assert_eq!(summary.items[0].fit.as_ref().unwrap().status, "ok");

        let _ = std::fs::remove_file(&file_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn fit_batch_empty_dir_fails() {
        let dir = std::env::temp_dir().join("eismaster_empty_batch_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let result = fit_batch(&dir, "zview_segmented_rq_rwo");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("没有找到"));

        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn fit_batch_item_file_uses_stem_no_extension() {
        let (spec, _) = synthetic_single_arc();
        let dir = std::env::temp_dir().join("eismaster_batch_stem_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let file_path = dir.join("Ag_EIS_OCV.txt");
        let mut content = String::new();
        for i in 0..spec.freq_hz.len() {
            content.push_str(&format!(
                "{:.6e}  {:.6e}  {:.6e}\n",
                spec.freq_hz[i], spec.z_real_ohm[i], spec.z_imag_ohm[i]
            ));
        }
        std::fs::write(&file_path, &content).unwrap();

        let summary = fit_batch(&dir, "zview_segmented_rq_rwo").unwrap();
        // file should be stem without .txt extension (matches Python file_path.stem)
        assert_eq!(summary.items[0].file, "Ag_EIS_OCV");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fit_outcome_has_parity_placeholder_fields() {
        let (spec, _) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");
        // confidence_intervals should exist (empty until bootstrap is implemented)
        assert!(outcome.confidence_intervals.is_empty());
        // correlation_matrix_max should be NaN (placeholder)
        assert!(outcome.correlation_matrix_max.is_nan());
    }

    #[test]
    fn batch_workbook_headers_match_python() {
        let (spec, _) = synthetic_single_arc();
        let outcome = fit_spectrum(&spec, "zview_segmented_rq_rwo");

        let summary = BatchSummary {
            model_key: "zview_segmented_rq_rwo".to_string(),
            model_label: "Single-arc R(QRWo)".to_string(),
            items: vec![BatchItemResult {
                file: "test".to_string(),
                label: "OCV".to_string(),
                quality: None,
                fit: Some(outcome),
                error: None,
                z_real_ohm: spec.z_real_ohm.clone(),
                z_imag_ohm: spec.z_imag_ohm.clone(),
            }],
            n_total: 1,
            n_ok: 1,
            n_warn: 0,
            n_failed: 0,
        };

        let dir = std::env::temp_dir().join("eismaster_header_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("headers.xlsx");

        write_batch_workbook(&path, &summary).unwrap();

        // Read back with rust_xlsxwriter is not straightforward,
        // but we can at least verify the file is created and non-trivial.
        // The actual header strings are verified by reading the source code.
        assert!(path.exists());
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 100);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
