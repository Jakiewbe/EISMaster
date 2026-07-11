use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDateTime;

use crate::models::{SpectrumData, SpectrumMetadata};

/// Date formats tried on the first line of a CHI TXT file.
const DATE_FORMATS: &[&str] = &[
    "%b. %d, %Y   %H:%M:%S",
    "%b %d, %Y   %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a CHI TXT file (or plain-numeric EIS text) and return SpectrumData.
pub fn parse_chi_txt(path: &Path) -> Result<SpectrumData, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Err(format!("empty file: {}", path.display()));
    }

    // If every line in the first 12 is purely numeric, treat as plain numeric.
    if looks_like_plain_numeric_eis(&lines) {
        let rows = parse_numeric_text_rows(&lines);
        return build_spectrum(
            path,
            "txt",
            "A.C. Impedance",
            "CHI660F",
            None,
            "",
            HashMap::new(),
            &rows,
        );
    }

    // Otherwise parse the CHI header format.
    if lines.len() < 5 {
        return Err(format!("text file too short: {}", path.display()));
    }

    let acquired_at = parse_datetime_line(lines[0].trim());
    let technique = lines
        .get(1)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let mut header: HashMap<String, String> = HashMap::new();
    let mut instrument_model = String::new();
    let mut note = String::new();
    let mut data_start: Option<usize> = None;

    for (idx, line) in lines.iter().enumerate() {
        let stripped = line.trim();
        if stripped.starts_with("Freq/Hz") {
            // Data starts two lines after the column header (skip the units row).
            data_start = Some(idx + 2);
            break;
        }
        if let Some((key, value)) = stripped.split_once(':') {
            let k = key.trim().to_string();
            let v = value.trim().to_string();
            if k == "Instrument Model" {
                instrument_model = v.clone();
            } else if k == "Note" {
                note = v.clone();
            }
            header.insert(k, v);
        }
    }

    let data_start = data_start
        .ok_or_else(|| format!("unable to find 'Freq/Hz' data header in {}", path.display()))?;

    let mut rows: Vec<(f64, f64, f64, Option<f64>, Option<f64>)> = Vec::new();
    for line in &lines[data_start..] {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        let parts: Vec<&str> = stripped.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 {
            continue;
        }
        let freq = match parts[0].parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let z_real = match parts[1].parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let z_imag = match parts[2].parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let z_mod = if parts.len() > 3 && !parts[3].is_empty() {
            parts[3].parse::<f64>().ok()
        } else {
            None
        };
        let phase = if parts.len() > 4 && !parts[4].is_empty() {
            parts[4].parse::<f64>().ok()
        } else {
            None
        };
        rows.push((freq, z_real, z_imag, z_mod, phase));
    }

    build_spectrum(
        path,
        "txt",
        &technique,
        &instrument_model,
        acquired_at,
        &note,
        header,
        &rows,
    )
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn build_spectrum(
    path: &Path,
    source_format: &str,
    technique: &str,
    instrument_model: &str,
    acquired_at: Option<NaiveDateTime>,
    note: &str,
    header: HashMap<String, String>,
    rows: &[(f64, f64, f64, Option<f64>, Option<f64>)],
) -> Result<SpectrumData, String> {
    if rows.is_empty() {
        return Err(format!("no EIS rows parsed from {}", path.display()));
    }

    let mut freq_hz = Vec::with_capacity(rows.len());
    let mut z_real_ohm = Vec::with_capacity(rows.len());
    let mut z_imag_ohm = Vec::with_capacity(rows.len());
    let mut z_mod_ohm = Vec::with_capacity(rows.len());
    let mut phase_deg = Vec::with_capacity(rows.len());

    for &(freq, zr, zi, zm, ph) in rows {
        freq_hz.push(freq);
        z_real_ohm.push(zr);
        z_imag_ohm.push(zi);
        z_mod_ohm.push(zm.unwrap_or_else(|| zr.hypot(zi)));
        // Phase: atan2(imag, real) in degrees, matching Python's math.degrees(atan2(z_imag, z_real)).
        phase_deg.push(ph.unwrap_or_else(|| zi.atan2(zr).to_degrees()));
    }

    let metadata = SpectrumMetadata {
        file_path: path.to_path_buf(),
        technique: technique.to_string(),
        instrument_model: instrument_model.to_string(),
        acquired_at,
        note: note.to_string(),
        header,
        source_format: source_format.to_string(),
    };

    Ok(SpectrumData {
        metadata,
        freq_hz,
        z_real_ohm,
        z_imag_ohm,
        z_mod_ohm,
        phase_deg,
    })
}

fn parse_datetime_line(text: &str) -> Option<NaiveDateTime> {
    for fmt in DATE_FORMATS {
        if let Ok(dt) = NaiveDateTime::parse_from_str(text, fmt) {
            return Some(dt);
        }
    }
    None
}

fn looks_like_plain_numeric_eis(lines: &[&str]) -> bool {
    let mut checked = 0usize;
    let mut numeric = 0usize;
    for line in lines.iter().take(12) {
        let parts = split_numeric_line(line);
        if parts.len() >= 3 {
            checked += 1;
            if parts[0].parse::<f64>().is_ok()
                && parts[1].parse::<f64>().is_ok()
                && parts[2].parse::<f64>().is_ok()
            {
                numeric += 1;
            }
        }
    }
    checked > 0 && numeric == checked
}

fn parse_numeric_text_rows(lines: &[&str]) -> Vec<(f64, f64, f64, Option<f64>, Option<f64>)> {
    let mut rows = Vec::new();
    for line in lines {
        let parts = split_numeric_line(line);
        if parts.len() < 3 {
            continue;
        }
        let (Ok(freq), Ok(zr), Ok(zi)) = (
            parts[0].parse::<f64>(),
            parts[1].parse::<f64>(),
            parts[2].parse::<f64>(),
        ) else {
            continue;
        };
        let z_mod = if parts.len() > 3 {
            parts[3].parse::<f64>().ok()
        } else {
            None
        };
        let phase = if parts.len() > 4 {
            parts[4].parse::<f64>().ok()
        } else {
            None
        };
        rows.push((freq, zr, zi, z_mod, phase));
    }
    rows
}

fn split_numeric_line(line: &str) -> Vec<String> {
    line.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
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
    fn parse_plain_numeric_file() {
        let path = fixture_dir().join("ag_eis_ocv.txt");
        let spectrum = parse_chi_txt(&path).expect("failed to parse plain numeric fixture");
        assert_eq!(spectrum.n_points(), 85);
        // First point: 96680 5.74 -0.4091
        assert!((spectrum.freq_hz[0] - 96680.0).abs() < 1.0);
        assert!((spectrum.z_real_ohm[0] - 5.74).abs() < 1e-6);
        assert!((spectrum.z_imag_ohm[0] - (-0.4091)).abs() < 1e-6);
        // Last point: 0.01 430.3 -2879
        let last = spectrum.n_points() - 1;
        assert!((spectrum.freq_hz[last] - 0.01).abs() < 1e-6);
        assert!((spectrum.z_real_ohm[last] - 430.3).abs() < 0.1);
        assert!((spectrum.z_imag_ohm[last] - (-2879.0)).abs() < 1.0);
        // z_mod should be computed from real/imag
        let expected_mod = (5.74_f64).hypot(-0.4091);
        assert!((spectrum.z_mod_ohm[0] - expected_mod).abs() < 1e-9);
    }

    #[test]
    fn parse_chi_header_format() {
        let path = fixture_dir().join("chi_header_sample.txt");
        let spectrum = parse_chi_txt(&path).expect("failed to parse CHI header fixture");
        assert_eq!(spectrum.n_points(), 5);
        assert_eq!(spectrum.metadata.technique, "A.C. Impedance");
        assert_eq!(spectrum.metadata.instrument_model, "CHI660F");
        assert!((spectrum.freq_hz[0] - 100000.0).abs() < 1.0);
        assert!((spectrum.z_real_ohm[0] - 10.0).abs() < 0.1);
        assert!((spectrum.z_imag_ohm[0] - (-0.5)).abs() < 0.1);
        // z_mod and phase from file columns
        assert!((spectrum.z_mod_ohm[0] - 10.012).abs() < 0.01);
        assert!((spectrum.phase_deg[0] - (-2.86)).abs() < 0.1);
    }

    #[test]
    fn parse_empty_file_errors() {
        let path = fixture_dir().join("empty.txt");
        let result = parse_chi_txt(&path);
        assert!(result.is_err());
    }
}
