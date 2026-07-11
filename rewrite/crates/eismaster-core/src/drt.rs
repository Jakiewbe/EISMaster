use std::f64::consts::LN_10;
use std::path::Path;

/// One parsed DRT curve with sorted logtau, gamma, and computed area.
#[derive(Debug, Clone)]
pub struct DrtCurve {
    pub label: String,
    pub logtau: Vec<f64>,
    pub tau_s: Vec<f64>,
    pub gamma_tau: Vec<f64>,
    pub area_dln_tau: Vec<f64>,
}

/// Raw parsed tau/gamma values before sorting and area computation.
struct RawDrt {
    tau: Vec<f64>,
    gamma: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a single `_DRT.txt` file and return sorted (logtau, tau_s, gamma, area).
pub fn parse_drt_file(path: &Path, label: String) -> Result<DrtCurve, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let lines: Vec<&str> = text.lines().collect();
    let raw = parse_drt_lines(&lines)?;
    build_curve(label, raw)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn parse_drt_lines(lines: &[&str]) -> Result<RawDrt, String> {
    // Look for continuous headers: "tau" or "freq"
    for (i, line) in lines.iter().enumerate() {
        let lower = line.trim().to_lowercase();
        if lower.starts_with("freq") {
            let raw = parse_continuous_data(&lines[i + 1..]);
            return Ok(convert_freq_to_tau(raw));
        }
        if lower.starts_with("tau") {
            return Ok(parse_continuous_data(&lines[i + 1..]));
        }
    }
    // Fallback: peak-fit format
    parse_peak_fit(lines)
}

fn parse_continuous_data(lines: &[&str]) -> RawDrt {
    let mut tau = Vec::new();
    let mut gamma = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) else {
            continue;
        };
        tau.push(x);
        gamma.push(y);
    }
    RawDrt { tau, gamma }
}

fn convert_freq_to_tau(raw: RawDrt) -> RawDrt {
    let tau = raw
        .tau
        .iter()
        .map(|&f| {
            if f > 0.0 {
                1.0 / (2.0 * std::f64::consts::PI * f)
            } else {
                f64::NAN
            }
        })
        .collect();
    RawDrt {
        tau,
        gamma: raw.gamma,
    }
}

fn parse_peak_fit(lines: &[&str]) -> Result<RawDrt, String> {
    let mut start_idx = None;
    for (i, line) in lines.iter().enumerate() {
        if line.trim().to_lowercase().starts_with("peak number") {
            start_idx = Some(i + 1);
            break;
        }
    }
    let start_idx =
        start_idx.ok_or_else(|| "no tau/freq/peak header found in DRT file".to_string())?;

    let mut peaks: Vec<(f64, f64, f64)> = Vec::new(); // (height, mu, sigma)
    for line in &lines[start_idx..] {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let (Ok(h), Ok(m), Ok(s)) = (
            parts[1].parse::<f64>(),
            parts[2].parse::<f64>(),
            parts[3].parse::<f64>(),
        ) else {
            continue;
        };
        let sigma = s.abs();
        if sigma > 0.0 {
            peaks.push((h, m, sigma));
        }
    }
    if peaks.is_empty() {
        return Err("no peaks found in peak-fit DRT file".to_string());
    }

    let start = peaks
        .iter()
        .map(|&(_, mu, sigma)| mu - 4.0 * sigma)
        .fold(f64::INFINITY, f64::min);
    let stop = peaks
        .iter()
        .map(|&(_, mu, sigma)| mu + 4.0 * sigma)
        .fold(f64::NEG_INFINITY, f64::max);

    let n = 160;
    let step = if n > 1 {
        (stop - start) / (n - 1) as f64
    } else {
        0.0
    };

    let mut logtau_ln = Vec::with_capacity(n);
    let mut gamma = vec![0.0; n];
    for j in 0..n {
        let x = start + j as f64 * step;
        logtau_ln.push(x);
        for &(height, mu, sigma) in &peaks {
            let z = (x - mu) / sigma;
            gamma[j] += height * (-0.5 * z * z).exp();
        }
    }
    let tau = logtau_ln.iter().map(|&lt| lt.exp()).collect();
    Ok(RawDrt { tau, gamma })
}

// ---------------------------------------------------------------------------
// Curve building (sort, area computation)
// ---------------------------------------------------------------------------

fn build_curve(label: String, raw: RawDrt) -> Result<DrtCurve, String> {
    // Filter to finite positive tau
    let mut pairs: Vec<(f64, f64)> = raw
        .tau
        .into_iter()
        .zip(raw.gamma.into_iter())
        .filter(|(t, g)| t.is_finite() && *t > 0.0 && g.is_finite())
        .collect();

    if pairs.is_empty() {
        return Err("no finite tau/gamma pairs in DRT data".to_string());
    }

    // Sort by logtau ascending
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n = pairs.len();
    let logtau: Vec<f64> = pairs.iter().map(|(t, _)| t.log10()).collect();
    let tau_s: Vec<f64> = pairs.iter().map(|(t, _)| *t).collect();
    let gamma_tau: Vec<f64> = pairs.iter().map(|(_, g)| *g).collect();

    let area_dln_tau = if n == 1 {
        vec![gamma_tau[0]]
    } else {
        compute_area_dln_tau(&logtau, &gamma_tau)
    };

    Ok(DrtCurve {
        label,
        logtau,
        tau_s,
        gamma_tau,
        area_dln_tau,
    })
}

/// Compute area = gamma * d(ln tau) using the same edge-midpoint scheme as Python.
///
/// - edges are midpoints between adjacent logtau values.
/// - first/last edges extend by half the adjacent spacing.
/// - dln_tau = diff(edges) * ln(10)
fn compute_area_dln_tau(logtau: &[f64], gamma: &[f64]) -> Vec<f64> {
    let n = logtau.len();
    let mut edges = vec![0.0; n + 1];
    // Interior edges: midpoints
    for i in 1..n {
        edges[i] = (logtau[i - 1] + logtau[i]) / 2.0;
    }
    // First edge extends by half the first spacing
    edges[0] = logtau[0] - (logtau[1] - logtau[0]) / 2.0;
    // Last edge extends by half the last spacing
    edges[n] = logtau[n - 1] + (logtau[n - 1] - logtau[n - 2]) / 2.0;

    let mut area = Vec::with_capacity(n);
    for i in 0..n {
        let dln_tau = (edges[i + 1] - edges[i]) * LN_10;
        area.push(gamma[i] * dln_tau);
    }
    area
}

// ---------------------------------------------------------------------------
// Label parsing (match Python _export_label_from_stem / _time_minutes_from_label)
// ---------------------------------------------------------------------------

/// Extract a display label from a file stem.
/// Prefers tokens like OCV, T10M, T100M, PRE01, POST01.
pub fn label_from_stem(stem: &str) -> String {
    let parts: Vec<&str> = stem
        .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return stem.to_string();
    }

    // Preferred tokens matching Python regex: (?i)(ocv|t\d+[smhd]?|e\d+|c\d+|soc\d+|rest|charge|discharge|before|after|init|mid|end)
    let preferred: Vec<&&str> = parts.iter().filter(|p| is_preferred_label(p)).collect();
    if let Some(last) = preferred.last() {
        return last.to_string();
    }

    // Informative tokens: contain letters, not pure numeric, length <= 16
    let informative: Vec<&&str> = parts
        .iter()
        .filter(|p| {
            p.len() <= 16
                && p.chars().any(|c| c.is_ascii_alphabetic())
                && !p.chars().all(|c| c.is_ascii_digit() || c == '.')
        })
        .collect();
    if let Some(last) = informative.last() {
        return last.to_string();
    }

    parts.last().unwrap().to_string()
}

fn is_preferred_label(s: &str) -> bool {
    let lo = s.to_ascii_lowercase();
    if lo == "ocv"
        || lo == "rest"
        || lo == "charge"
        || lo == "discharge"
        || lo == "before"
        || lo == "after"
        || lo == "init"
        || lo == "mid"
        || lo == "end"
    {
        return true;
    }
    // T<digits>[smhd]?
    if lo.starts_with('t') {
        let rest = &lo[1..];
        if !rest.is_empty() {
            let mut chars = rest.chars().peekable();
            let mut has_digit = false;
            while let Some(c) = chars.next() {
                if c.is_ascii_digit() {
                    has_digit = true;
                } else if has_digit
                    && (c == 's' || c == 'm' || c == 'h' || c == 'd')
                    && chars.peek().is_none()
                {
                    return true;
                } else {
                    break;
                }
            }
            if has_digit && chars.next().is_none() {
                return rest.chars().all(|c| c.is_ascii_digit());
            }
        }
    }
    // E<number>, C<number>, SOC<number>
    if lo.starts_with('e') && lo[1..].chars().all(|c| c.is_ascii_digit()) && !lo[1..].is_empty() {
        return true;
    }
    if lo.starts_with('c') && lo[1..].chars().all(|c| c.is_ascii_digit()) && !lo[1..].is_empty() {
        return true;
    }
    if lo.starts_with("soc") && lo[3..].chars().all(|c| c.is_ascii_digit()) && !lo[3..].is_empty() {
        return true;
    }
    false
}

/// Extract time in minutes from a label like T10M, T30S, T2H, T1D.
pub fn time_minutes_from_label(label: &str) -> Option<f64> {
    let lo = label.to_ascii_lowercase();
    if !lo.contains('t') {
        return None;
    }
    // Find T followed by digits then optional s/m/h/d
    let bytes = lo.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b't' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
                end += 1;
            }
            let Ok(val) = lo[start..end].parse::<f64>() else {
                i = end;
                continue;
            };
            let unit = if end < bytes.len() {
                bytes[end] as char
            } else {
                '\0'
            };
            return Some(match unit {
                's' => val / 60.0,
                'h' => val * 60.0,
                'd' => val * 1440.0,
                _ => val, // 'm' or no unit
            });
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    #[test]
    fn parse_tau_gamma_file() {
        let path = fixture_dir().join("sample_drt_tau.txt");
        let curve = parse_drt_file(&path, "OCV".into()).expect("failed to parse tau DRT");
        assert_eq!(curve.logtau.len(), 3);
        assert!((curve.logtau[0] - 0.0).abs() < 1e-9); // log10(1) = 0
        assert!((curve.logtau[1] - 1.0).abs() < 1e-9); // log10(10) = 1
        assert!((curve.logtau[2] - 2.0).abs() < 1e-9); // log10(100) = 2
        assert!((curve.gamma_tau[0] - 1.0).abs() < 1e-9);
        assert!((curve.gamma_tau[1] - 2.0).abs() < 1e-9);
        assert!((curve.gamma_tau[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn parse_freq_gamma_file() {
        let path = fixture_dir().join("sample_drt_freq.txt");
        let curve = parse_drt_file(&path, "T10M".into()).expect("failed to parse freq DRT");
        // Data sorted ascending by tau. freq=100 -> tau=0.00159 is first.
        assert_eq!(curve.tau_s.len(), 3);
        // freq=100 -> tau = 1/(2*pi*100)
        let tau_from_100 = 1.0 / (2.0 * std::f64::consts::PI * 100.0);
        assert!((curve.tau_s[0] - tau_from_100).abs() < 1e-12);
        // freq=1 -> tau = 1/(2*pi*1) is last after sort
        let tau_from_1 = 1.0 / (2.0 * std::f64::consts::PI * 1.0);
        assert!((curve.tau_s[2] - tau_from_1).abs() < 1e-12);
        assert_eq!(curve.label, "T10M");
    }

    #[test]
    fn parse_peak_fit_file() {
        let path = fixture_dir().join("sample_drt_peak.txt");
        let curve = parse_drt_file(&path, "peak_test".into()).expect("failed to parse peak DRT");
        assert_eq!(curve.logtau.len(), 160);
        // Single peak at mu=0, sigma=0.5, height=2
        // Peak gamma should be approximately 2.0 at logtau≈0
        let peak_idx = curve.logtau.len() / 2;
        assert!((curve.gamma_tau[peak_idx] - 2.0).abs() < 0.1);
    }

    #[test]
    fn area_computation_matches_python() {
        // Same fixture as Python test: tau=1,10,100; gamma=1,2,3
        let path = fixture_dir().join("sample_drt_tau.txt");
        let curve = parse_drt_file(&path, "test".into()).unwrap();
        // Python expects total_area ≈ 13.8155
        let total: f64 = curve.area_dln_tau.iter().sum();
        assert!((total - 13.815510557964274).abs() < 1e-6);
    }

    #[test]
    fn label_from_stem_ocv() {
        assert_eq!(label_from_stem("Ag_EIS_OCV"), "OCV");
    }

    #[test]
    fn label_from_stem_time() {
        assert_eq!(label_from_stem("Ag_S01_EIS_T5M"), "T5M");
        assert_eq!(label_from_stem("Ag_S01_EIS_T100M"), "T100M");
    }

    #[test]
    fn time_minutes_parsing() {
        assert!((time_minutes_from_label("T10M").unwrap() - 10.0).abs() < 1e-9);
        assert!((time_minutes_from_label("T30S").unwrap() - 0.5).abs() < 1e-9);
        assert!((time_minutes_from_label("T2H").unwrap() - 120.0).abs() < 1e-9);
        assert!((time_minutes_from_label("T1D").unwrap() - 1440.0).abs() < 1e-9);
        assert!(time_minutes_from_label("OCV").is_none());
    }
}
