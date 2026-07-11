use serde::Serialize;

use crate::models::SpectrumData;

/// Result of arc segmentation on an EIS spectrum.
#[derive(Debug, Clone, Serialize)]
pub struct SegmentDetection {
    pub requested_mode: String,
    pub resolved_mode: String,
    pub peak_indices: Vec<usize>,
    pub split_indices: Vec<usize>,
}

/// Detect arc segments in an EIS spectrum.
///
/// Uses `-z_imag` (positive for capacitive arcs) to find peaks and valleys.
/// Modes: "auto", "single", "double".
/// Manual overrides: optional peak/split indices.
pub fn detect_segments(
    spectrum: &SpectrumData,
    mode: &str,
    manual_split1: Option<usize>,
    manual_split2: Option<usize>,
    manual_peak1: Option<usize>,
    manual_peak2: Option<usize>,
) -> SegmentDetection {
    let n = spectrum.n_points();
    let raw_y: Vec<f64> = spectrum.z_imag_ohm.iter().map(|&v| -v).collect();
    let y = smooth_trace(&raw_y);
    let mut peaks = significant_peaks(&y);

    if mode == "double" && peaks.len() < 2 {
        peaks = fallback_double_peaks(&raw_y);
    } else if mode == "auto" && peaks.len() < 2 {
        let fallback = fallback_double_peaks(&raw_y);
        if fallback.len() >= 2 {
            peaks = fallback;
        }
    }

    let resolved_mode = if mode == "single" {
        "single"
    } else if mode == "double" {
        "double"
    } else if peaks.len() >= 2 {
        "double"
    } else {
        "single"
    };

    if resolved_mode == "double" && peaks.len() < 2 {
        if peaks.len() == 1 {
            let p = peaks[0];
            peaks.push((p + 5).min(n.saturating_sub(2)));
        } else {
            peaks = vec![n / 3, 2 * n / 3];
        }
    }

    if resolved_mode == "single" {
        let mut peak = if peaks.is_empty() {
            argmax(&y)
        } else {
            peaks[0]
        };
        let mut split1 = valley_after(&y, peak, n.saturating_sub(4));
        if let Some(mp) = manual_peak1 {
            if mp < n {
                peak = mp;
            }
        }
        if let Some(ms) = manual_split1 {
            if ms >= 1 && ms < n.saturating_sub(3) {
                split1 = ms;
            }
        }
        let (peak, split1) = sanitize_single_controls(n, peak, split1);
        return SegmentDetection {
            requested_mode: mode.to_string(),
            resolved_mode: "single".to_string(),
            peak_indices: vec![peak],
            split_indices: vec![split1],
        };
    }

    // Double mode
    let mut peak1 = peaks[0];
    let mut peak2 = peaks[1];
    let mut split1 = valley_between(&y, peak1, peak2);
    let mut split2 = valley_after(&y, peak2, n.saturating_sub(4));
    if let Some(mp) = manual_peak1 {
        if mp < n {
            peak1 = mp;
        }
    }
    if let Some(mp) = manual_peak2 {
        if mp < n {
            peak2 = mp;
        }
    }
    if let Some(ms) = manual_split1 {
        if ms >= 1 && ms < n.saturating_sub(2) {
            split1 = ms;
        }
    }
    if let Some(ms) = manual_split2 {
        if ms >= 1 && ms < n.saturating_sub(1) {
            split2 = ms;
        }
    }
    let (peak1, split1, peak2, split2) = sanitize_double_controls(n, peak1, split1, peak2, split2);
    SegmentDetection {
        requested_mode: mode.to_string(),
        resolved_mode: "double".to_string(),
        peak_indices: vec![peak1, peak2],
        split_indices: vec![split1, split2],
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Weighted 5-point smoothing: kernel [1,2,3,2,1]/9 with edge padding.
fn smooth_trace(values: &[f64]) -> Vec<f64> {
    if values.len() < 5 {
        return values.to_vec();
    }
    let n = values.len();
    let kernel = [1.0, 2.0, 3.0, 2.0, 1.0];
    let ksum: f64 = kernel.iter().sum(); // 9.0

    // Edge-pad: repeat first value 2x, last value 2x
    let mut padded = Vec::with_capacity(n + 4);
    padded.push(values[0]);
    padded.push(values[0]);
    padded.extend_from_slice(values);
    padded.push(values[n - 1]);
    padded.push(values[n - 1]);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = 0.0;
        for (k, &kw) in kernel.iter().enumerate() {
            s += padded[i + k] * kw;
        }
        out.push(s / ksum);
    }
    out
}

/// Find local maxima with prominence filtering.
///
/// A point is a peak if y[i] >= y[i-1] and y[i] >= y[i+1].
/// Keeps peaks whose prominence >= max(y[i]*0.01, 0.2).
/// Nearby peaks (< 4 apart) are merged by keeping the taller one.
fn significant_peaks(y: &[f64]) -> Vec<usize> {
    if y.len() < 5 {
        return if y.is_empty() {
            vec![]
        } else {
            vec![argmax(y)]
        };
    }

    // Local maxima indices
    let mut indices: Vec<usize> = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] >= y[i - 1] && y[i] >= y[i + 1] {
            indices.push(i);
        }
    }
    if indices.is_empty() {
        return vec![argmax(y)];
    }

    // Prominence filtering
    let mut kept: Vec<usize> = Vec::new();
    for &idx in &indices {
        let left_min = y[idx.saturating_sub(5)..=idx]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let right_end = (idx + 6).min(y.len());
        let right_min = y[idx..right_end]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let baseline = left_min.max(right_min);
        let prominence = y[idx] - baseline;
        let threshold = (y[idx] * 0.01).max(0.2);
        if prominence < threshold {
            continue;
        }
        if let Some(&last) = kept.last() {
            if idx - last < 4 {
                if y[idx] > y[last] {
                    *kept.last_mut().unwrap() = idx;
                }
                continue;
            }
        }
        kept.push(idx);
    }
    if !kept.is_empty() {
        return kept;
    }

    // Fallback: just enforce min spacing of 4, no prominence filter
    let mut fallback: Vec<usize> = Vec::new();
    for &idx in &indices {
        if let Some(&last) = fallback.last() {
            if idx - last < 4 {
                if y[idx] > y[last] {
                    *fallback.last_mut().unwrap() = idx;
                }
                continue;
            }
        }
        fallback.push(idx);
    }
    if !fallback.is_empty() {
        fallback
    } else {
        vec![argmax(y)]
    }
}

/// Find the index of minimum value between `start` and `stop` (inclusive).
fn valley_between(y: &[f64], start: usize, stop: usize) -> usize {
    if stop <= start + 1 {
        return (start + 1).min(y.len().saturating_sub(2));
    }
    let mut min_val = f64::INFINITY;
    let mut min_idx = start;
    for i in start..=stop.min(y.len() - 1) {
        if y[i] < min_val {
            min_val = y[i];
            min_idx = i;
        }
    }
    min_idx
}

/// Find the index of minimum value after `start` up to `stop`.
fn valley_after(y: &[f64], start: usize, stop: usize) -> usize {
    let search_end = stop.min(y.len().saturating_sub(1));
    if search_end <= start + 1 {
        return (start + 1).min(y.len().saturating_sub(2));
    }
    let seg_start = start + 1;
    let seg_end = search_end + 1;
    let mut min_val = f64::INFINITY;
    let mut min_idx = seg_start;
    for i in seg_start..seg_end {
        if y[i] < min_val {
            min_val = y[i];
            min_idx = i;
        }
    }
    let candidate = min_idx;

    // If valley is right at peak edge and tail is too small, push further
    let tail_needed = 6;
    if candidate <= start + 2 && (search_end - candidate) < tail_needed {
        let fallback = search_end.saturating_sub(tail_needed);
        if fallback > start + 2 {
            return fallback;
        }
    }
    candidate
}

/// Find two peaks for weak signals where significant_peaks returned < 2.
fn fallback_double_peaks(y: &[f64]) -> Vec<usize> {
    if y.len() < 12 {
        return if y.is_empty() {
            vec![]
        } else {
            vec![argmax(y)]
        };
    }

    let mut indices: Vec<usize> = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] >= y[i - 1] && y[i] >= y[i + 1] {
            indices.push(i);
        }
    }
    if indices.is_empty() {
        return vec![argmax(y)];
    }

    let stop = ((y.len() as f64 * 0.85) as usize).max(2);
    let mut candidates: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&i| i >= 2 && i <= stop)
        .collect();
    if candidates.len() < 2 {
        candidates = indices;
    }

    // Rank by y value descending
    candidates.sort_by(|&a, &b| y[b].partial_cmp(&y[a]).unwrap());
    let mut selected: Vec<usize> = Vec::new();
    for idx in candidates {
        if selected.is_empty() {
            selected.push(idx);
            continue;
        }
        if selected.iter().all(|&prev| idx.abs_diff(prev) >= 6) {
            selected.push(idx);
        }
        if selected.len() == 2 {
            break;
        }
    }
    if selected.len() >= 2 {
        selected.sort();
        selected
    } else if !y.is_empty() {
        vec![argmax(y)]
    } else {
        vec![]
    }
}

/// Clamp single-arc controls to valid ranges.
fn sanitize_single_controls(n_points: usize, peak: usize, split1: usize) -> (usize, usize) {
    if n_points <= 6 {
        let peak = clamp(peak, 0, n_points.saturating_sub(2));
        let split1 = clamp(
            split1,
            (peak + 1).min(n_points.saturating_sub(1)),
            n_points.saturating_sub(2),
        );
        return (peak, split1);
    }
    let peak = clamp(peak, 1, n_points - 5);
    let split1 = clamp(split1, peak + 1, n_points - 4);
    (peak, split1)
}

/// Clamp double-arc controls to valid ranges.
fn sanitize_double_controls(
    n_points: usize,
    peak1: usize,
    split1: usize,
    peak2: usize,
    split2: usize,
) -> (usize, usize, usize, usize) {
    if n_points <= 10 {
        let peak1 = clamp(peak1, 0, n_points.saturating_sub(4));
        let split1 = clamp(split1, peak1 + 1, n_points.saturating_sub(3).max(1));
        let peak2 = clamp(
            peak2,
            split1 + 1,
            n_points.saturating_sub(2).max(split1 + 1),
        );
        let split2 = clamp(split2, peak2 + 1, n_points.saturating_sub(1).max(peak2 + 1));
        return (peak1, split1, peak2, split2);
    }
    let peak1 = clamp(peak1, 1, n_points - 7);
    let split1 = clamp(split1, peak1 + 1, n_points - 6);
    let peak2 = clamp(peak2, split1 + 1, n_points - 5);
    let split2 = clamp(split2, peak2 + 1, n_points - 4);
    (peak1, split1, peak2, split2)
}

fn argmax(y: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..y.len() {
        if y[i] > y[best] {
            best = i;
        }
    }
    best
}

fn clamp(v: usize, lo: usize, hi: usize) -> usize {
    v.max(lo).min(hi)
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

    fn make_spectrum(z_imag: Vec<f64>) -> SpectrumData {
        let n = z_imag.len();
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
            freq_hz: (0..n).map(|i| 10f64.powi(6 - i as i32)).collect(),
            z_real_ohm: vec![0.0; n],
            z_imag_ohm: z_imag,
            z_mod_ohm: vec![0.0; n],
            phase_deg: vec![0.0; n],
        }
    }

    // A simple single semicircle: -z_imag rises then falls
    fn single_arc_imag() -> Vec<f64> {
        // 20 points, -z_imag peaks around index 9
        vec![
            -1.0, -5.0, -15.0, -35.0, -60.0, -90.0, -120.0, -148.0, -170.0, -185.0, -180.0, -165.0,
            -140.0, -110.0, -80.0, -52.0, -30.0, -15.0, -5.0, -1.0,
        ]
    }

    // Two overlapping semicircles
    fn double_arc_imag() -> Vec<f64> {
        vec![
            -1.0, -5.0, -20.0, -45.0, -70.0, -80.0, -70.0, -50.0, -35.0, -30.0, -40.0, -60.0,
            -85.0, -100.0, -95.0, -75.0, -50.0, -30.0, -15.0, -5.0, -1.0,
        ]
    }

    #[test]
    fn single_arc_auto_mode() {
        let spec = make_spectrum(single_arc_imag());
        let det = detect_segments(&spec, "auto", None, None, None, None);
        assert_eq!(det.resolved_mode, "single");
        assert_eq!(det.peak_indices.len(), 1);
        assert_eq!(det.split_indices.len(), 1);
        // Peak should be near index 9 (the most negative z_imag)
        assert!((9..=10).contains(&det.peak_indices[0]));
    }

    #[test]
    fn single_arc_explicit_mode() {
        let spec = make_spectrum(single_arc_imag());
        let det = detect_segments(&spec, "single", None, None, None, None);
        assert_eq!(det.resolved_mode, "single");
        assert_eq!(det.requested_mode, "single");
    }

    #[test]
    fn double_arc_manual_peaks() {
        let spec = make_spectrum(double_arc_imag());
        let det = detect_segments(&spec, "double", None, None, Some(5), Some(13));
        assert_eq!(det.resolved_mode, "double");
        assert_eq!(det.peak_indices.len(), 2);
        assert_eq!(det.peak_indices[0], 5);
        assert_eq!(det.peak_indices[1], 13);
        assert!(det.split_indices[0] > det.peak_indices[0]);
        assert!(det.split_indices[0] < det.peak_indices[1]);
        assert!(det.split_indices[1] > det.peak_indices[1]);
    }

    #[test]
    fn auto_mode_picks_double_for_two_arcs() {
        let spec = make_spectrum(double_arc_imag());
        let det = detect_segments(&spec, "auto", None, None, None, None);
        // The double_arc_imag has two clear peaks, auto should detect double
        assert_eq!(det.resolved_mode, "double");
    }

    #[test]
    fn split_indices_within_bounds() {
        let spec = make_spectrum(single_arc_imag());
        let det = detect_segments(&spec, "auto", None, None, None, None);
        let n = spec.n_points();
        for &s in &det.split_indices {
            assert!(s < n, "split index {} out of bounds (n={})", s, n);
        }
        for &p in &det.peak_indices {
            assert!(p < n, "peak index {} out of bounds (n={})", p, n);
        }
    }

    #[test]
    fn terminal_points_never_crash() {
        // Spectrum with all zeros
        let spec = make_spectrum(vec![0.0; 5]);
        let det = detect_segments(&spec, "auto", None, None, None, None);
        assert!(det.peak_indices[0] < 5);
    }

    #[test]
    fn very_short_spectrum() {
        let spec = make_spectrum(vec![-1.0, -2.0, -1.0]);
        let det = detect_segments(&spec, "single", None, None, None, None);
        assert_eq!(det.resolved_mode, "single");
    }

    #[test]
    fn smooth_trace_matches_expected() {
        let raw = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let smoothed = smooth_trace(&raw);
        assert_eq!(smoothed.len(), 5);
        // With edge padding [10,10,10,20,30,40,50,50,50] and kernel [1,2,3,2,1]/9
        // smoothed[0] = (10*1 + 10*2 + 10*3 + 20*2 + 30*1) / 9 = (10+20+30+40+30)/9 = 130/9
        let expected_0 = (10.0 + 20.0 + 30.0 + 40.0 + 30.0) / 9.0;
        assert!((smoothed[0] - expected_0).abs() < 1e-10);
    }

    #[test]
    fn valley_between_finds_minimum() {
        let y = vec![5.0, 3.0, 1.0, 2.0, 4.0];
        let v = valley_between(&y, 0, 4);
        assert_eq!(v, 2); // y[2] = 1.0 is minimum
    }

    #[test]
    fn sanitize_single_clamps_correctly() {
        let (p, s) = sanitize_single_controls(20, 25, 30);
        assert!(p <= 15); // n_points - 5
        assert!(s >= p + 1);
        assert!(s <= 16); // n_points - 4
    }

    #[test]
    fn sanitize_double_clamps_correctly() {
        let (p1, s1, p2, s2) = sanitize_double_controls(20, 0, 1, 2, 3);
        assert!(p1 >= 1);
        assert!(s1 > p1);
        assert!(p2 > s1);
        assert!(s2 > p2);
        assert!(s2 <= 16); // n_points - 4
    }
}
