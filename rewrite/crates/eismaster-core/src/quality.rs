use serde::Serialize;

use crate::models::SpectrumData;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single quality issue found during assessment.
#[derive(Debug, Clone, Serialize)]
pub struct QualityIssue {
    pub severity: String,
    pub message: String,
}

/// Quality assessment result for a spectrum.
#[derive(Debug, Clone, Serialize)]
pub struct QualityReport {
    pub status: String,
    pub issues: Vec<QualityIssue>,
    pub kk_status: String,
    pub kk_message: String,
}

// ---------------------------------------------------------------------------
// Assessment
// ---------------------------------------------------------------------------

/// Assess the quality of an EIS spectrum.
///
/// Checks: point count, finite values, positive frequency, descending frequency,
/// duplicate frequencies, negative z_real, inductive start, outliers.
/// KK/Z-HIT is always "not_run" (Python-only feature).
pub fn assess_spectrum_quality(spectrum: &SpectrumData) -> QualityReport {
    let mut issues: Vec<QualityIssue> = Vec::new();
    let n = spectrum.n_points();

    if n < 8 {
        issues.push(QualityIssue {
            severity: "error".to_string(),
            message: "数据点过少，无法稳定拟合。".to_string(),
        });
    }

    let has_non_finite = spectrum.freq_hz.iter().any(|v| !v.is_finite())
        || spectrum.z_real_ohm.iter().any(|v| !v.is_finite())
        || spectrum.z_imag_ohm.iter().any(|v| !v.is_finite());
    if has_non_finite {
        issues.push(QualityIssue {
            severity: "error".to_string(),
            message: "谱图中存在非有限值。".to_string(),
        });
    }

    if spectrum.freq_hz.iter().any(|&v| v <= 0.0) {
        issues.push(QualityIssue {
            severity: "error".to_string(),
            message: "频率必须为正值。".to_string(),
        });
    }

    if n >= 2 && spectrum.freq_hz.windows(2).any(|w| w[0] <= w[1]) {
        issues.push(QualityIssue {
            severity: "warning".to_string(),
            message: "频率序列不是严格降序。".to_string(),
        });
    }

    if has_duplicate_freqs(&spectrum.freq_hz) {
        issues.push(QualityIssue {
            severity: "warning".to_string(),
            message: "检测到重复频点。".to_string(),
        });
    }

    if spectrum.z_real_ohm.iter().any(|&v| v < 0.0) {
        issues.push(QualityIssue {
            severity: "warning".to_string(),
            message: "检测到负实部阻抗。".to_string(),
        });
    }

    if let Some(&first_zi) = spectrum.z_imag_ohm.first() {
        if first_zi > 0.0 {
            issues.push(QualityIssue {
                severity: "info".to_string(),
                message: "高频端虚部起点高于零。".to_string(),
            });
        }
    }

    let outlier_count = detect_outlier_count(spectrum);
    if outlier_count > 0 {
        issues.push(QualityIssue {
            severity: "warning".to_string(),
            message: format!("检测到 {outlier_count} 个可能异常点。"),
        });
    }

    let status = if issues.iter().any(|i| i.severity == "error") {
        "fail"
    } else if issues.iter().any(|i| i.severity == "warning") {
        "warn"
    } else {
        "pass"
    };

    QualityReport {
        status: status.to_string(),
        issues,
        kk_status: "not_run".to_string(),
        kk_message: "KK/Z-HIT 未执行。".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

/// Count outliers using curvature/slope/log-frequency MAD-based voting.
fn detect_outlier_count(spectrum: &SpectrumData) -> usize {
    let n = spectrum.n_points();
    if n < 7 {
        return 0;
    }

    let curvature_scale = 6.0;
    let slope_scale = 6.0;
    let gradient_scale = 7.0;
    let vote_threshold = 2;

    // y = -z_imag
    let y: Vec<f64> = spectrum.z_imag_ohm.iter().map(|&v| -v).collect();

    // Curvature: |y[i-1] - 2*y[i] + y[i+1]|
    let mut curv = vec![0.0; n];
    for i in 1..n - 1 {
        curv[i] = (y[i - 1] - 2.0 * y[i] + y[i + 1]).abs();
    }
    let med_c = median(&curv);
    let mad_c = scaled_mad(&curv);
    let thresh_c = med_c + curvature_scale * mad_c.max(1e-12);

    // Slope difference: |slopes[k] - slopes[k-1]|
    // Python: slopes = arctan2(dy, dx), slope_diff[1:-1] = |diff(slopes)|
    let x = &spectrum.z_real_ohm;
    let mut slopes: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        let dy = y[i + 1] - y[i];
        slopes.push(dy.atan2(dx.abs().max(1e-30)));
    }
    let mut slope_diff = vec![0.0; n];
    for k in 1..n - 1 {
        if k < slopes.len() && k - 1 < slopes.len() {
            slope_diff[k] = (slopes[k] - slopes[k - 1]).abs();
        }
    }
    let med_s = median(&slope_diff);
    let mad_s = scaled_mad(&slope_diff);
    let thresh_s = med_s + slope_scale * mad_s.max(1e-12);

    // Log-frequency gradient smoothness
    // Python: log_grad = np.gradient(y, log_f), smooth_jump[1:-1] = |diff(log_grad, 2)|
    let log_f: Vec<f64> = spectrum
        .freq_hz
        .iter()
        .map(|&v| v.max(1e-30).log10())
        .collect();
    let mut log_grad = vec![0.0; n];
    for i in 0..n {
        if i == 0 && n >= 2 {
            log_grad[i] = (y[1] - y[0]) / (log_f[1] - log_f[0]).max(1e-30);
        } else if i == n - 1 && n >= 2 {
            log_grad[i] = (y[n - 1] - y[n - 2]) / (log_f[n - 1] - log_f[n - 2]).max(1e-30);
        } else if n >= 3 {
            log_grad[i] = (y[i + 1] - y[i - 1]) / (log_f[i + 1] - log_f[i - 1]).max(1e-30);
        }
    }
    let mut smooth_jump = vec![0.0; n];
    if n >= 3 {
        for i in 1..n - 1 {
            smooth_jump[i] = (log_grad[i + 1] - 2.0 * log_grad[i] + log_grad[i - 1]).abs();
        }
    }
    let med_g = median(&smooth_jump);
    let mad_g = scaled_mad(&smooth_jump);
    let thresh_g = med_g + gradient_scale * mad_g.max(1e-12);

    // Vote
    let mut count = 0;
    for k in 1..n - 1 {
        let mut score = 0;
        if curv[k] > thresh_c {
            score += 1;
        }
        if slope_diff[k] > thresh_s {
            score += 1;
        }
        if smooth_jump[k] > thresh_g {
            score += 1;
        }
        if score >= vote_threshold {
            count += 1;
        }
    }
    count
}

/// Median of a slice.
fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Scaled MAD: 1.4826 * median(|x - median(x)|).
fn scaled_mad(values: &[f64]) -> f64 {
    let med = median(values);
    let abs_dev: Vec<f64> = values.iter().map(|&v| (v - med).abs()).collect();
    1.4826 * median(&abs_dev)
}

/// Check for duplicate frequencies (rounded to 12 decimal places).
fn has_duplicate_freqs(freq: &[f64]) -> bool {
    if freq.len() < 2 {
        return false;
    }
    let mut rounded: Vec<i64> = freq.iter().map(|&v| (v * 1e12).round() as i64).collect();
    rounded.sort();
    rounded.windows(2).any(|w| w[0] == w[1])
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
                file_path: PathBuf::from("test.txt"),
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

    fn good_spectrum() -> SpectrumData {
        make_spectrum(
            vec![1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        )
    }

    #[test]
    fn good_spectrum_passes() {
        let report = assess_spectrum_quality(&good_spectrum());
        assert_eq!(report.status, "pass");
        assert!(report.issues.is_empty());
    }

    #[test]
    fn too_few_points_is_error() {
        let spec = make_spectrum(
            vec![1e5, 1e4, 1e3],
            vec![5.0, 6.0, 8.0],
            vec![-0.5, -2.0, -5.0],
        );
        let report = assess_spectrum_quality(&spec);
        assert_eq!(report.status, "fail");
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "error" && i.message.contains("数据点过少")));
    }

    #[test]
    fn non_finite_values_is_error() {
        let spec = make_spectrum(
            vec![1e5, 1e4, f64::NAN, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert_eq!(report.status, "fail");
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "error" && i.message.contains("非有限值")));
    }

    #[test]
    fn non_positive_freq_is_error() {
        let spec = make_spectrum(
            vec![1e5, 0.0, 1e3, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert_eq!(report.status, "fail");
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "error" && i.message.contains("频率必须为正值")));
    }

    #[test]
    fn non_descending_freq_is_warning() {
        let spec = make_spectrum(
            vec![1e3, 1e4, 1e2, 1e1, 1.0, 0.1, 0.01, 0.001],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert_eq!(report.status, "warn");
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "warning" && i.message.contains("降序")));
    }

    #[test]
    fn duplicate_freq_is_warning() {
        let spec = make_spectrum(
            vec![1e5, 1e4, 1e4, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "warning" && i.message.contains("重复频点")));
    }

    #[test]
    fn negative_z_real_is_warning() {
        let spec = make_spectrum(
            vec![1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, -1.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![-0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "warning" && i.message.contains("负实部")));
    }

    #[test]
    fn inductive_start_is_info() {
        let spec = make_spectrum(
            vec![1e5, 1e4, 1e3, 1e2, 1e1, 1.0, 0.1, 0.01],
            vec![5.0, 6.0, 8.0, 12.0, 18.0, 22.0, 24.0, 25.0],
            vec![0.5, -2.0, -5.0, -8.0, -6.0, -3.0, -1.0, -0.3],
        );
        let report = assess_spectrum_quality(&spec);
        assert!(report
            .issues
            .iter()
            .any(|i| i.severity == "info" && i.message.contains("虚部起点")));
    }

    #[test]
    fn kk_is_always_not_run() {
        let report = assess_spectrum_quality(&good_spectrum());
        assert_eq!(report.kk_status, "not_run");
    }

    #[test]
    fn median_odd_length() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn median_even_length() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn scaled_mad_zero_for_constant() {
        let v = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert_eq!(scaled_mad(&v), 0.0);
    }
}
