use num_complex::Complex64;

// ---------------------------------------------------------------------------
// Impedance primitives
// ---------------------------------------------------------------------------

/// CPE (Constant Phase Element) impedance.
///
/// `Z_CPE = 1 / (T * (j*omega)^P)`
pub fn z_cpe(omega: f64, t: f64, p: f64) -> Complex64 {
    let jw = Complex64::new(0.0, omega);
    let jw_p = jw.powf(p);
    Complex64::new(1.0, 0.0) / (t * jw_p)
}

/// Warburg open (finite-length Warburg) impedance.
///
/// `Z_Wo = Wo_R / (x * tanh(x))` where `x = (j*omega*Wo_T)^Wo_P`.
///
/// Handles numerical edge cases:
/// - |x| > 50: tanh(x) ≈ 1
/// - |x| < 1e-8: use Taylor expansion `x*tanh(x) ≈ x²(1 - x²/3)`
pub fn z_warburg_open(omega: f64, wo_r: f64, wo_t: f64, wo_p: f64) -> Complex64 {
    let jw = Complex64::new(0.0, omega);
    let x = (jw * wo_t).powf(wo_p);
    let abs_x = x.norm();

    let tanh_x = if abs_x > 50.0 {
        Complex64::new(1.0, 0.0)
    } else {
        x.tanh()
    };

    let denom = if abs_x < 1e-8 {
        // Taylor: x*tanh(x) ≈ x²(1 - x²/3)
        let x2 = x * x;
        x2 * (Complex64::new(1.0, 0.0) - x2 / 3.0)
    } else {
        x * tanh_x
    };

    let denom = if denom.norm() < 1e-30 {
        Complex64::new(1e-30, 0.0)
    } else {
        denom
    };

    Complex64::new(wo_r, 0.0) / denom
}

// ---------------------------------------------------------------------------
// Circuit models
// ---------------------------------------------------------------------------

/// Single-arc R(QRWo) model.
///
/// `Z = R1 + 1/(1/Z_CPE + 1/(R2 + Z_Wo))`
///
/// Params: [R1, CPE_T, CPE_P, R2, Wo_R, Wo_T, Wo_P]
pub fn zview_single_model(freq_hz: &[f64], params: &[f64; 7]) -> Vec<Complex64> {
    let [r1, cpe_t, cpe_p, r2, wo_r, wo_t, wo_p] = *params;
    freq_hz
        .iter()
        .map(|&f| {
            let omega = 2.0 * std::f64::consts::PI * f;
            let zcpe = z_cpe(omega, cpe_t, cpe_p);
            let zwo = z_warburg_open(omega, wo_r, wo_t, wo_p);
            let z_branch = Complex64::new(r2, 0.0) + zwo;
            Complex64::new(r1, 0.0)
                + Complex64::new(1.0, 0.0)
                    / (Complex64::new(1.0, 0.0) / zcpe + Complex64::new(1.0, 0.0) / z_branch)
        })
        .collect()
}

/// Double-arc R(QR)(Q(RWo)) model.
///
/// `Z = Rs + (Q1||R2) + (Q2||(R3+Z_Wo))`
///
/// Params: [Rs, Q1, n1, R2, Q2, n2, R3, Wo_R, Wo_T, Wo_P]
pub fn zview_double_model(freq_hz: &[f64], params: &[f64; 10]) -> Vec<Complex64> {
    let [rs, q1, n1, r2, q2, n2, r3, wo_r, wo_t, wo_p] = *params;
    freq_hz
        .iter()
        .map(|&f| {
            let omega = 2.0 * std::f64::consts::PI * f;
            let zcpe1 = z_cpe(omega, q1, n1);
            let zcpe2 = z_cpe(omega, q2, n2);
            let zwo = z_warburg_open(omega, wo_r, wo_t, wo_p);
            let one = Complex64::new(1.0, 0.0);
            let z_arc1 = one / (one / Complex64::new(r2, 0.0) + one / zcpe1);
            let z_arc2 = one / (one / zcpe2 + one / (Complex64::new(r3, 0.0) + zwo));
            Complex64::new(rs, 0.0) + z_arc1 + z_arc2
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Template metadata
// ---------------------------------------------------------------------------

/// Circuit template descriptor (matches Python `CircuitTemplate`).
#[derive(Debug, Clone)]
pub struct CircuitTemplate {
    pub key: &'static str,
    pub label: &'static str,
    pub parameter_names: &'static [&'static str],
    pub primary_exports: &'static [&'static str],
}

/// All built-in circuit templates.
pub fn templates() -> &'static [CircuitTemplate] {
    &[
        CircuitTemplate {
            key: "zview_segmented_rq_rwo",
            label: "Single-arc R(QRWo)",
            parameter_names: &[
                "Rs",
                "CPE_T",
                "CPE_P",
                "Rct",
                "Wo_R",
                "Wo_T",
                "Wo_P",
                "split_freq_hz",
            ],
            primary_exports: &["Rs", "Rct"],
        },
        CircuitTemplate {
            key: "zview_double_rq_qrwo",
            label: "Double-arc R(QR)(Q(RWo))",
            parameter_names: &[
                "Rs",
                "Q1",
                "n1",
                "Rsei",
                "Q2",
                "n2",
                "Rct",
                "Wo_R",
                "Wo_T",
                "Wo_P",
                "split1_freq_hz",
                "split2_freq_hz",
            ],
            primary_exports: &["Rs", "Rsei", "Rct"],
        },
    ]
}

/// Look up a template by key.
pub fn get_template(key: &str) -> Option<&'static CircuitTemplate> {
    templates().iter().find(|t| t.key == key)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpe_at_1khz() {
        // T=1e-5, P=0.9, f=1000Hz
        // omega = 2*pi*1000 ≈ 6283.185
        // Z_CPE = 1/(1e-5 * (j*6283.185)^0.9)
        let z = z_cpe(2.0 * std::f64::consts::PI * 1000.0, 1e-5, 0.9);
        // CPE has negative imaginary part for 0 < P < 1
        assert!(z.im < 0.0);
        assert!(z.re > 0.0);
    }

    #[test]
    fn warburg_open_dc_limit() {
        // At very low frequency, Warburg open should be finite
        let z = z_warburg_open(0.001, 10.0, 1.0, 0.5);
        assert!(z.re.is_finite());
        assert!(z.im.is_finite());
    }

    #[test]
    fn warburg_open_high_freq() {
        // At high frequency, tanh → 1, Z_Wo → Wo_R/x
        let z = z_warburg_open(1e6, 10.0, 1e-3, 0.5);
        assert!(z.re.is_finite());
        assert!(z.im.is_finite());
    }

    #[test]
    fn single_model_produces_correct_count() {
        let freqs = vec![1e5, 1e4, 1e3, 1e2, 1e1];
        let params = [5.0, 1e-5, 0.9, 20.0, 10.0, 1.0, 0.5];
        let z = zview_single_model(&freqs, &params);
        assert_eq!(z.len(), 5);
        for val in &z {
            assert!(val.re.is_finite());
            assert!(val.im.is_finite());
        }
    }

    #[test]
    fn single_model_dc_resistance() {
        // At DC (very low freq), CPE → open, Warburg → finite
        // Z → R1 + R2 (series)
        let freqs = vec![1e-6];
        let params = [10.0, 1e-5, 0.9, 50.0, 10.0, 1.0, 0.5];
        let z = zview_single_model(&freqs, &params);
        // At very low freq, CPE impedance is huge → Z ≈ R1 + R2
        // But Warburg complicates things; just check finiteness
        assert!(z[0].re.is_finite());
        assert!(z[0].im.is_finite());
    }

    #[test]
    fn double_model_produces_correct_count() {
        let freqs = vec![1e5, 1e3, 1e1, 0.1];
        let params = [5.0, 1e-5, 0.9, 20.0, 1e-4, 0.85, 30.0, 10.0, 1.0, 0.5];
        let z = zview_double_model(&freqs, &params);
        assert_eq!(z.len(), 4);
        for val in &z {
            assert!(val.re.is_finite());
            assert!(val.im.is_finite());
        }
    }

    #[test]
    fn template_keys_exist() {
        assert!(get_template("zview_segmented_rq_rwo").is_some());
        assert!(get_template("zview_double_rq_qrwo").is_some());
        assert!(get_template("nonexistent").is_none());
    }

    #[test]
    fn template_param_names_match_python() {
        let single = get_template("zview_segmented_rq_rwo").unwrap();
        assert_eq!(
            single.parameter_names,
            &[
                "Rs",
                "CPE_T",
                "CPE_P",
                "Rct",
                "Wo_R",
                "Wo_T",
                "Wo_P",
                "split_freq_hz"
            ]
        );
        let double = get_template("zview_double_rq_qrwo").unwrap();
        assert_eq!(
            double.parameter_names,
            &[
                "Rs",
                "Q1",
                "n1",
                "Rsei",
                "Q2",
                "n2",
                "Rct",
                "Wo_R",
                "Wo_T",
                "Wo_P",
                "split1_freq_hz",
                "split2_freq_hz"
            ]
        );
    }

    #[test]
    fn single_model_real_increases_toward_dc() {
        // For a typical battery spectrum, Z_real should increase at lower freq
        let freqs = vec![1e5, 1e4, 1e3, 1e2, 1e1, 1.0];
        let params = [1.0, 1e-5, 0.9, 10.0, 5.0, 1.0, 0.5];
        let z = zview_single_model(&freqs, &params);
        // Real part at lowest freq should be > real part at highest freq
        assert!(
            z.last().unwrap().re > z.first().unwrap().re,
            "Z_real should increase toward DC"
        );
    }

    #[test]
    fn single_model_imag_negative_for_capacitive() {
        // For typical EIS, -Z_imag > 0 (capacitive) at mid frequencies
        let freqs = vec![1e3];
        let params = [1.0, 1e-5, 0.9, 10.0, 5.0, 1.0, 0.5];
        let z = zview_single_model(&freqs, &params);
        assert!(z[0].im < 0.0, "Z_imag should be negative (capacitive)");
    }
}
