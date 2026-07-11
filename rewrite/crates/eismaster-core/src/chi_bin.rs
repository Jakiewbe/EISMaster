//! CHI .bin binary impedance file parser.
//!
//! Ported from Python `src/eismaster/io/chi.py:parse_chi_bin()`.
//!
//! CHI binary format:
//! - Each record is 16 bytes: 4 x f32 little-endian (freq_1, freq_2, z_real, z_imag).
//! - freq_1 and freq_2 should agree within 1e-3 relative tolerance (sanity check).
//! - Record count stored as u16 at offsets 0x25E or 0x266.
//! - DateTime stored as 6 x u16 at offsets 0x26A..0x27E (year, month, day, hour, minute, second).
//! - Header identifier "IMP" should appear in first 32 bytes.

use std::collections::HashMap;
use std::path::Path;

use chrono::NaiveDateTime;

use crate::models::{SpectrumData, SpectrumMetadata};

/// Parse a CHI .bin binary impedance file.
pub fn parse_chi_bin(path: &Path) -> Result<SpectrumData, String> {
    let raw = std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    // Sanity: "IMP" should appear in first 32 bytes
    if !raw.windows(3).take(32).any(|w| w == b"IMP") {
        return Err(format!(
            "{} does not look like a CHI impedance binary file",
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
        ));
    }

    let count = extract_record_count(&raw);
    let (data_start, data_count) = find_trailing_binary_record_run(&raw);

    let (start, count) =
        if data_count.is_some() && (count.is_none() || data_count.unwrap() > count.unwrap_or(0)) {
            (data_start.unwrap(), data_count.unwrap())
        } else if let Some(c) = count {
            let s = raw.len() as i64 - c as i64 * 16;
            if s < 0 {
                return Err(format!("invalid record count for {}", path.display()));
            }
            (s as usize, c)
        } else {
            return Err(format!(
                "unable to determine record count for {}",
                path.display()
            ));
        };

    let mut rows: Vec<(f64, f64, f64, Option<f64>, Option<f64>)> = Vec::new();
    let mut prev_freq = f64::INFINITY;

    for idx in 0..count {
        let offset = start + idx * 16;
        if offset + 16 > raw.len() {
            return Err(format!(
                "binary record at index {idx} exceeds file size in {}",
                path.display()
            ));
        }
        let bytes = &raw[offset..offset + 16];
        let freq_1 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64;
        let freq_2 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as f64;
        let z_real = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as f64;
        let z_imag = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as f64;

        if !freq_1.is_finite() || !freq_2.is_finite() || !z_real.is_finite() || !z_imag.is_finite()
        {
            return Err(format!(
                "non-finite binary record at index {idx} in {}",
                path.display()
            ));
        }
        if freq_1 <= 0.0 || (freq_1 - freq_2).abs() / freq_1.max(1.0) > 1e-3 {
            return Err(format!(
                "unexpected binary record layout at index {idx} in {}",
                path.display()
            ));
        }
        if freq_1 > prev_freq * 1.05 {
            return Err(format!(
                "binary frequencies are not monotonic in {}",
                path.display()
            ));
        }
        prev_freq = freq_1;
        rows.push((freq_1, z_real, z_imag, None, None));
    }

    let acquired_at = extract_bin_datetime(&raw);
    let mut header = HashMap::new();
    if let Some(dt) = acquired_at {
        header.insert(
            "Acquired At".to_string(),
            dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        );
    }

    // Detect technique and instrument model from header bytes
    let technique = if raw
        .windows(b"A.C. Impedance".len())
        .take(128.min(raw.len()))
        .any(|w| w == b"A.C. Impedance")
    {
        "A.C. Impedance"
    } else {
        "Unknown"
    };

    let instrument_model = if raw
        .windows(b"CHI660F".len())
        .take(128.min(raw.len()))
        .any(|w| w == b"CHI660F")
    {
        "CHI660F"
    } else if raw
        .windows(b"CHI660E".len())
        .take(128.min(raw.len()))
        .any(|w| w == b"CHI660E")
    {
        "CHI660E"
    } else if raw
        .windows(b"CHI604E".len())
        .take(128.min(raw.len()))
        .any(|w| w == b"CHI604E")
    {
        "CHI604E"
    } else {
        "Unknown"
    };

    build_spectrum_from_rows(
        path,
        "bin",
        technique,
        instrument_model,
        acquired_at,
        "",
        header,
        &rows,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn extract_record_count(raw: &[u8]) -> Option<usize> {
    let mut candidates = Vec::new();
    for offset in [0x25Eusize, 0x266] {
        if raw.len() >= offset + 2 {
            let count = u16::from_le_bytes([raw[offset], raw[offset + 1]]) as usize;
            if count >= 5 && count <= raw.len() / 16 {
                candidates.push(count);
            }
        }
    }
    if candidates.len() == 1 {
        return Some(candidates[0]);
    }
    // If both agree, return that value
    if candidates.len() == 2 && candidates[0] == candidates[1] {
        return Some(candidates[0]);
    }
    candidates.into_iter().next()
}

fn find_trailing_binary_record_run(raw: &[u8]) -> (Option<usize>, Option<usize>) {
    let max_start = (raw.len().saturating_sub((raw.len() / 16) * 16)).max(0);
    let mut best_start = None;
    let mut best_count = 0usize;

    for start in max_start..=raw.len().saturating_sub(16) {
        let count = count_binary_records_from(raw, start);
        if count > best_count && start + count * 16 == raw.len() {
            best_start = Some(start);
            best_count = count;
        }
    }

    if best_start.is_none() || best_count < 5 {
        return (None, None);
    }
    (best_start, Some(best_count))
}

fn count_binary_records_from(raw: &[u8], start: usize) -> usize {
    let mut count = 0usize;
    let mut prev_freq = f64::INFINITY;

    while start + (count + 1) * 16 <= raw.len() {
        let offset = start + count * 16;
        let bytes = &raw[offset..offset + 16];
        let freq_1 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64;
        let freq_2 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as f64;
        let z_real = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as f64;
        let z_imag = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as f64;

        if !freq_1.is_finite() || !freq_2.is_finite() || !z_real.is_finite() || !z_imag.is_finite()
        {
            break;
        }
        if freq_1 <= 0.0 || (freq_1 - freq_2).abs() / freq_1.max(1.0) > 1e-3 {
            break;
        }
        if freq_1 > prev_freq * 1.05 {
            break;
        }
        prev_freq = freq_1;
        count += 1;
    }
    count
}

fn extract_bin_datetime(raw: &[u8]) -> Option<NaiveDateTime> {
    let offsets = [0x26Ausize, 0x26E, 0x272, 0x276, 0x27A, 0x27E];
    if raw.len() < offsets.iter().max().copied().unwrap_or(0) + 2 {
        return None;
    }
    let values: Vec<u32> = offsets
        .iter()
        .map(|&off| u16::from_le_bytes([raw[off], raw[off + 1]]) as u32)
        .collect();

    // year, month, day, hour, minute, second
    Some(NaiveDateTime::new(
        chrono::NaiveDate::from_ymd_opt(values[0] as i32, values[1], values[2])?,
        chrono::NaiveTime::from_hms_opt(values[3], values[4], values[5])?,
    ))
}

fn build_spectrum_from_rows(
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

    let n = rows.len();
    let mut freq_hz = Vec::with_capacity(n);
    let mut z_real_ohm = Vec::with_capacity(n);
    let mut z_imag_ohm = Vec::with_capacity(n);
    let mut z_mod_ohm = Vec::with_capacity(n);
    let mut phase_deg = Vec::with_capacity(n);

    for &(freq, zr, zi, zm, ph) in rows {
        freq_hz.push(freq);
        z_real_ohm.push(zr);
        z_imag_ohm.push(zi);
        z_mod_ohm.push(zm.unwrap_or_else(|| zr.hypot(zi)));
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid .bin file in memory for parsing tests.
    fn make_minimal_bin(records: &[(f64, f64, f64)]) -> Vec<u8> {
        // Header needs to be at least 0x280 = 640 bytes for datetime offsets at 0x26A-0x27F
        let header_size: usize = 0x280;
        let mut buf = vec![0u8; header_size];
        // Write "IMP" at start
        buf[0] = b'I';
        buf[1] = b'M';
        buf[2] = b'P';
        // Write technique marker
        let tech = b"A.C. Impedance";
        buf[16..16 + tech.len()].copy_from_slice(tech);
        // Write instrument marker
        let inst = b"CHI660F";
        buf[48..48 + inst.len()].copy_from_slice(inst);

        // Write record count (u16) at 0x25E
        let count = records.len() as u16;
        buf[0x25E] = (count & 0xFF) as u8;
        buf[0x25F] = (count >> 8) as u8;
        // Also at 0x266 for redundancy
        buf[0x266] = (count & 0xFF) as u8;
        buf[0x267] = (count >> 8) as u8;

        // Write datetime: 2024-06-15 14:30:00
        let dt_offsets = [0x26A, 0x26E, 0x272, 0x276, 0x27A, 0x27E];
        let dt_values: [u16; 6] = [2024, 6, 15, 14, 30, 0];
        for (off, val) in dt_offsets.iter().zip(dt_values.iter()) {
            buf[*off] = (*val & 0xFF) as u8;
            buf[*off + 1] = (*val >> 8) as u8;
        }

        // Records go at the end (trailing binary records)
        let record_bytes = records.len() * 16;
        buf.resize(header_size + record_bytes, 0);

        let data_start = header_size;
        for (i, &(freq, zr, zi)) in records.iter().enumerate() {
            let off = data_start + i * 16;
            let fb = (freq as f32).to_le_bytes();
            let zb = (zr as f32).to_le_bytes();
            let ib = (zi as f32).to_le_bytes();
            buf[off..off + 4].copy_from_slice(&fb);
            buf[off + 4..off + 8].copy_from_slice(&fb); // freq_1 == freq_2
            buf[off + 8..off + 12].copy_from_slice(&zb);
            buf[off + 12..off + 16].copy_from_slice(&ib);
        }

        buf
    }

    #[test]
    fn parse_minimal_bin_file() {
        let records = vec![
            (100000.0, 10.0, -0.5),
            (50000.0, 12.0, -2.0),
            (10000.0, 18.0, -8.0),
            (1000.0, 30.0, -25.0),
            (100.0, 50.0, -40.0),
            (10.0, 80.0, -30.0),
            (1.0, 100.0, -10.0),
            (0.1, 110.0, -2.0),
        ];
        let data = make_minimal_bin(&records);
        let tmp = std::env::temp_dir().join("eismaster_test_chi.bin");
        std::fs::write(&tmp, &data).unwrap();

        let spectrum = parse_chi_bin(&tmp).expect("should parse valid .bin");
        assert_eq!(spectrum.n_points(), records.len());
        assert_eq!(spectrum.metadata.source_format, "bin");
        assert_eq!(spectrum.metadata.instrument_model, "CHI660F");
        assert_eq!(spectrum.metadata.technique, "A.C. Impedance");
        assert!(spectrum.metadata.acquired_at.is_some());

        // Check first record
        assert!((spectrum.freq_hz[0] - 100000.0).abs() < 1.0);
        assert!((spectrum.z_real_ohm[0] - 10.0).abs() < 1e-4);
        assert!((spectrum.z_imag_ohm[0] - (-0.5)).abs() < 1e-4);

        // Check last record
        let last = spectrum.n_points() - 1;
        assert!((spectrum.freq_hz[last] - 0.1).abs() < 1e-2);
        assert!((spectrum.z_real_ohm[last] - 110.0).abs() < 1e-4);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bin_missing_imp_marker_fails() {
        let buf = vec![0u8; 256]; // no "IMP"
        let tmp = std::env::temp_dir().join("eismaster_test_no_imp.bin");
        std::fs::write(&tmp, &buf).unwrap();
        let result = parse_chi_bin(&tmp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not look like"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bin_non_finite_record_fails() {
        // Write a valid header with record count >= 5, but one record has NaN
        let header_size: usize = 0x280;
        let records: Vec<(f64, f64, f64)> = vec![
            (100000.0, 10.0, -0.5),
            (50000.0, 12.0, -2.0),
            (10000.0, 18.0, -8.0),
            (1000.0, 30.0, -25.0),
            (100.0, f64::NAN, -40.0), // NaN z_real at index 4
            (10.0, 80.0, -30.0),
        ];
        let n = records.len();
        let mut buf = vec![0u8; header_size + n * 16];
        buf[0] = b'I';
        buf[1] = b'M';
        buf[2] = b'P';
        buf[0x25E] = (n & 0xFF) as u8;
        buf[0x25F] = (n >> 8) as u8;

        for (i, &(freq, zr, zi)) in records.iter().enumerate() {
            let off = header_size + i * 16;
            let fb = (freq as f32).to_le_bytes();
            let zb = (zr as f32).to_le_bytes();
            let ib = (zi as f32).to_le_bytes();
            buf[off..off + 4].copy_from_slice(&fb);
            buf[off + 4..off + 8].copy_from_slice(&fb);
            buf[off + 8..off + 12].copy_from_slice(&zb);
            buf[off + 12..off + 16].copy_from_slice(&ib);
        }

        let tmp = std::env::temp_dir().join("eismaster_test_nan.bin");
        std::fs::write(&tmp, &buf).unwrap();
        let result = parse_chi_bin(&tmp);
        assert!(result.is_err(), "expected error, got {:?}", result.ok());
        let err = result.unwrap_err();
        assert!(
            err.contains("non-finite"),
            "expected 'non-finite', got: {err}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn bin_freq_mismatch_fails() {
        // Write valid header but freq_1 != freq_2
        let header_size: usize = 0x280;
        let n: usize = 5;
        let mut buf = vec![0u8; header_size + n * 16];
        buf[0] = b'I';
        buf[1] = b'M';
        buf[2] = b'P';
        buf[0x25E] = (n & 0xFF) as u8;
        buf[0x25F] = (n >> 8) as u8;

        let f1 = (1000.0f32).to_le_bytes();
        let f2 = (500.0f32).to_le_bytes();
        let z = (10.0f32).to_le_bytes();
        // First record: freq mismatch
        buf[header_size..header_size + 4].copy_from_slice(&f1);
        buf[header_size + 4..header_size + 8].copy_from_slice(&f2);
        buf[header_size + 8..header_size + 12].copy_from_slice(&z);
        buf[header_size + 12..header_size + 16].copy_from_slice(&z);
        // Remaining records: valid
        for i in 1..n {
            let off = header_size + i * 16;
            let f = ((50000.0 / i as f64) as f32).to_le_bytes();
            buf[off..off + 4].copy_from_slice(&f);
            buf[off + 4..off + 8].copy_from_slice(&f);
            buf[off + 8..off + 12].copy_from_slice(&z);
            buf[off + 12..off + 16].copy_from_slice(&z);
        }

        let tmp = std::env::temp_dir().join("eismaster_test_mismatch.bin");
        std::fs::write(&tmp, &buf).unwrap();
        let result = parse_chi_bin(&tmp);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("unexpected binary record layout"));
        let _ = std::fs::remove_file(&tmp);
    }
}
