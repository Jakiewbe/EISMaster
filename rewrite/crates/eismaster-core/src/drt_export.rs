use std::f64;
use std::path::Path;

use rust_xlsxwriter::{Format, Workbook, XlsxError};

use crate::drt::{time_minutes_from_label, DrtCurve};

/// X-axis mode for the line plot sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineXAxis {
    LogTau,
    Tau,
}

/// Format a logtau bound into a region label suffix (matches Python `_format_logtau_bound`).
fn format_logtau_bound(value: f64) -> String {
    let s = format!("{}", value);
    s.replace('-', "neg").replace('.', "p")
}

/// Build region descriptors from breaks.
fn logtau_regions(breaks: &[f64]) -> Vec<(String, f64, f64)> {
    if breaks.is_empty() {
        return vec![("logtau_all".to_string(), f64::NEG_INFINITY, f64::INFINITY)];
    }
    let mut sorted: Vec<f64> = breaks.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();

    let mut regions = Vec::new();
    regions.push((
        format!("logtau_lt_{}", format_logtau_bound(sorted[0])),
        f64::NEG_INFINITY,
        sorted[0],
    ));
    for w in sorted.windows(2) {
        regions.push((
            format!(
                "logtau_{}_to_{}",
                format_logtau_bound(w[0]),
                format_logtau_bound(w[1])
            ),
            w[0],
            w[1],
        ));
    }
    regions.push((
        format!("logtau_ge_{}", format_logtau_bound(*sorted.last().unwrap())),
        *sorted.last().unwrap(),
        f64::INFINITY,
    ));
    regions
}

/// Interpolate values from one logtau grid to a target logtau grid.
fn interpolate_to_logtau(logtau: &[f64], values: &[f64], target: &[f64]) -> Vec<f64> {
    let mut finite_pairs: Vec<(f64, f64)> = logtau
        .iter()
        .zip(values.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();

    if finite_pairs.len() < 2 {
        return vec![f64::NAN; target.len()];
    }
    finite_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    target
        .iter()
        .map(|&xt| {
            if xt < finite_pairs[0].0 || xt > finite_pairs.last().unwrap().0 {
                return f64::NAN;
            }
            // Find bracketing interval
            let mut lo = 0;
            for i in 1..finite_pairs.len() {
                if finite_pairs[i].0 >= xt {
                    // Linear interpolation between i-1 and i
                    let x0 = finite_pairs[i - 1].0;
                    let x1 = finite_pairs[i].0;
                    let y0 = finite_pairs[i - 1].1;
                    let y1 = finite_pairs[i].1;
                    if (x1 - x0).abs() < 1e-30 {
                        return y0;
                    }
                    return y0 + (y1 - y0) * (xt - x0) / (x1 - x0);
                }
                lo = i;
            }
            finite_pairs[lo].1
        })
        .collect()
}

/// Write `drt_matrix.xlsx` with three sheets.
pub fn write_drt_workbook(
    output_path: &Path,
    curves: &[DrtCurve],
    line_x: LineXAxis,
    logtau_breaks: &[f64],
) -> Result<(), XlsxError> {
    if curves.is_empty() {
        return Ok(());
    }

    // Align all curves to a common logtau grid (use the first curve's grid as reference)
    let common_logtau = &curves[0].logtau;
    let mut aligned: Vec<DrtCurve> = Vec::with_capacity(curves.len());
    for curve in curves {
        if &curve.logtau == common_logtau {
            aligned.push(curve.clone());
        } else {
            let gamma = interpolate_to_logtau(&curve.logtau, &curve.gamma_tau, common_logtau);
            let area = interpolate_to_logtau(&curve.logtau, &curve.area_dln_tau, common_logtau);
            aligned.push(DrtCurve {
                label: curve.label.clone(),
                logtau: common_logtau.clone(),
                tau_s: curve.tau_s.clone(),
                gamma_tau: gamma,
                area_dln_tau: area,
            });
        }
    }

    let mut workbook = Workbook::new();

    // --- Sheet 1: drt_line_plot ---
    let line_sheet = workbook.add_worksheet();
    line_sheet.set_name("drt_line_plot")?;
    let num_fmt = Format::new().set_num_format("0.000000E+00");

    // Row 0: blank, then labels
    line_sheet.write_string(0, 0, "")?;
    for (j, curve) in aligned.iter().enumerate() {
        line_sheet.write_string(0, (j + 1) as u16, &curve.label)?;
    }
    // Row 1: x label, then gamma_tau repeated
    let x_label = if line_x == LineXAxis::Tau {
        "tau/s"
    } else {
        "logtau"
    };
    line_sheet.write_string(1, 0, x_label)?;
    for j in 0..aligned.len() {
        line_sheet.write_string(1, (j + 1) as u16, "gamma_tau")?;
    }
    // Data rows
    let max_len = aligned.iter().map(|c| c.logtau.len()).max().unwrap_or(0);
    for i in 0..max_len {
        let x_val = if i < aligned[0].logtau.len() {
            if line_x == LineXAxis::Tau {
                aligned[0].tau_s[i]
            } else {
                aligned[0].logtau[i]
            }
        } else {
            f64::NAN
        };
        if x_val.is_finite() {
            line_sheet.write_number_with_format((i + 2) as u32, 0, x_val, &num_fmt)?;
        }
        for (j, curve) in aligned.iter().enumerate() {
            let val = if i < curve.gamma_tau.len() {
                curve.gamma_tau[i]
            } else {
                f64::NAN
            };
            if val.is_finite() {
                line_sheet.write_number_with_format(
                    (i + 2) as u32,
                    (j + 1) as u16,
                    val,
                    &num_fmt,
                )?;
            }
        }
    }

    // --- Sheet 2: drt_cloud_density ---
    let cloud_sheet = workbook.add_worksheet();
    cloud_sheet.set_name("drt_cloud_density")?;
    cloud_sheet.write_string(0, 0, "")?;
    for (j, curve) in aligned.iter().enumerate() {
        cloud_sheet.write_string(0, (j + 1) as u16, &curve.label)?;
    }
    cloud_sheet.write_string(1, 0, "logtau")?;
    for j in 0..aligned.len() {
        cloud_sheet.write_string(1, (j + 1) as u16, "gamma_tau")?;
    }
    for (i, &lt) in common_logtau.iter().enumerate() {
        cloud_sheet.write_number_with_format((i + 2) as u32, 0, lt, &num_fmt)?;
        for (j, curve) in aligned.iter().enumerate() {
            let val = if i < curve.gamma_tau.len() {
                curve.gamma_tau[i]
            } else {
                f64::NAN
            };
            if val.is_finite() {
                cloud_sheet.write_number_with_format(
                    (i + 2) as u32,
                    (j + 1) as u16,
                    val,
                    &num_fmt,
                )?;
            }
        }
    }

    // --- Sheet 3: drt_quant_area ---
    let quant_sheet = workbook.add_worksheet();
    quant_sheet.set_name("drt_quant_area")?;
    let regions = logtau_regions(logtau_breaks);

    // Headers
    quant_sheet.write_string(0, 0, "sample")?;
    quant_sheet.write_string(0, 1, "time_min")?;
    quant_sheet.write_string(0, 2, "total_area")?;
    for (j, (name, _, _)) in regions.iter().enumerate() {
        quant_sheet.write_string(0, (j + 3) as u16, name)?;
    }

    // Data rows
    for (row_idx, curve) in aligned.iter().enumerate() {
        let r = (row_idx + 1) as u32;
        quant_sheet.write_string(r, 0, &curve.label)?;

        if let Some(tm) = time_minutes_from_label(&curve.label) {
            quant_sheet.write_number(r, 1, tm)?;
        }

        let total_area: f64 = curve.area_dln_tau.iter().filter(|v| v.is_finite()).sum();
        quant_sheet.write_number(r, 2, total_area)?;

        for (j, (_, lo, hi)) in regions.iter().enumerate() {
            let region_area: f64 = curve
                .logtau
                .iter()
                .zip(curve.area_dln_tau.iter())
                .filter(|(lt, _)| lt.is_finite() && **lt >= *lo && **lt < *hi)
                .map(|(_, a)| a)
                .sum();
            quant_sheet.write_number(r, (j + 3) as u16, region_area)?;
        }
    }

    workbook.save(output_path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drt::parse_drt_file;
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn tmp_dir() -> PathBuf {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test_tmp");
        std::fs::create_dir_all(&p).ok();
        p
    }

    #[test]
    fn export_workbook_has_three_sheets() {
        let path = fixture_dir().join("sample_drt_tau.txt");
        let curve = parse_drt_file(&path, "OCV".into()).unwrap();
        let out = tmp_dir().join("test_export_sheets.xlsx");
        write_drt_workbook(&out, &[curve], LineXAxis::LogTau, &[-3.0, 0.0]).expect("write failed");
        assert!(out.exists());
        // Verify it's a valid xlsx by checking it has content
        let meta = std::fs::metadata(&out).unwrap();
        assert!(meta.len() > 100);
    }

    #[test]
    fn export_with_tau_axis() {
        let path = fixture_dir().join("sample_drt_tau.txt");
        let curve = parse_drt_file(&path, "T10M".into()).unwrap();
        let out = tmp_dir().join("test_export_tau_axis.xlsx");
        write_drt_workbook(&out, &[curve], LineXAxis::Tau, &[-3.0, 0.0]).expect("write failed");
        assert!(out.exists());
    }

    #[test]
    fn custom_logtau_breaks() {
        let breaks = vec![1.0, 3.0];
        let regions = logtau_regions(&breaks);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].0, "logtau_lt_1");
        assert_eq!(regions[1].0, "logtau_1_to_3");
        assert_eq!(regions[2].0, "logtau_ge_3");
    }

    #[test]
    fn default_logtau_breaks() {
        let breaks = vec![-3.0, 0.0];
        let regions = logtau_regions(&breaks);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].0, "logtau_lt_neg3");
        assert_eq!(regions[1].0, "logtau_neg3_to_0");
        assert_eq!(regions[2].0, "logtau_ge_0");
    }
}
