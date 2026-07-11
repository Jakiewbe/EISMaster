use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::models::SpectrumData;

/// Format a float with up to 12 significant digits (g-style).
///
/// Strips trailing zeros and unnecessary decimal point.
/// Matches Python's `f"{value:.12g}"` behavior.
fn format_g12(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.12e}", value);
    // Parse "1.234567890000e-03" -> trim trailing zeros in mantissa
    if let Some(exp_pos) = s.find('e') {
        let (mantissa, exp) = s.split_at(exp_pos);
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{}{}", mantissa, exp)
    } else {
        s
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// MATLAB DRT execution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatlabDrtConfig {
    pub matlab_exe: String,
    pub drttools_dir: String,
    pub matlab_bridge_dir: String,
    pub method_tag: String,
    pub drt_type: u32,
    pub lambda_value: f64,
    pub coeff_value: f64,
    pub derivative_order: String,
    pub data_used: String,
    pub inductance_mode: u32,
    pub shape_control: String,
}

impl Default for MatlabDrtConfig {
    fn default() -> Self {
        Self {
            matlab_exe: String::new(),
            drttools_dir: String::new(),
            matlab_bridge_dir: String::new(),
            method_tag: "simple".to_string(),
            drt_type: 2,
            lambda_value: 1e-3,
            coeff_value: 0.5,
            derivative_order: "1st-order".to_string(),
            data_used: "Combined Re-Im Data".to_string(),
            inductance_mode: 1,
            shape_control: "FWHM Coefficient".to_string(),
        }
    }
}

/// Result of a MATLAB DRT run.
#[derive(Debug, Clone, Serialize)]
pub struct MatlabDrtResult {
    pub command: String,
    pub return_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub staging_dir: String,
    pub output_dir: String,
    pub output_files: Vec<String>,
    pub workbook_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

/// Stage selected spectra into tab-separated input files for MATLAB.
///
/// Writes `freq\tz_real\tz_imag` per line (raw z_imag, no sign flip).
/// Returns the staging directory path.
pub fn stage_matlab_drt_inputs(
    spectra: &[SpectrumData],
    base_output_dir: &Path,
) -> Result<PathBuf, String> {
    let staging_dir = base_output_dir.join("matlab_drt_inputs");
    if staging_dir.exists() {
        std::fs::remove_dir_all(&staging_dir)
            .map_err(|e| format!("failed to clean staging dir: {e}"))?;
    }
    std::fs::create_dir_all(&staging_dir)
        .map_err(|e| format!("failed to create staging dir: {e}"))?;

    for spectrum in spectra {
        let stem = spectrum
            .metadata
            .file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let target = staging_dir.join(format!("{stem}.txt"));
        write_raw_impedance_input(&target, spectrum)?;
    }

    Ok(staging_dir)
}

/// Write a single spectrum as tab-separated freq/z_real/z_imag.
pub fn write_raw_impedance_input(path: &Path, spectrum: &SpectrumData) -> Result<(), String> {
    let mut lines = Vec::with_capacity(spectrum.freq_hz.len());
    for i in 0..spectrum.freq_hz.len() {
        lines.push(format!(
            "{}\t{}\t{}",
            format_g12(spectrum.freq_hz[i]),
            format_g12(spectrum.z_real_ohm[i]),
            format_g12(spectrum.z_imag_ohm[i])
        ));
    }
    let content = lines.join("\n") + "\n";
    std::fs::write(path, content).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// MATLAB command building
// ---------------------------------------------------------------------------

/// Quote a string for MATLAB single-quoted literal.
///
/// `'` becomes `''` (MATLAB escaping convention).
pub fn matlab_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Build the `-batch` argument string for MATLAB.
///
/// Returns: `addpath('<runner_dir>'); eismaster_batch_drt('<arg1>', ...);`
pub fn build_matlab_batch_string(
    runner_dir: &Path,
    input_dir: &Path,
    output_dir: &Path,
    config: &MatlabDrtConfig,
) -> String {
    let runner_dir_s = runner_dir.to_string_lossy().to_string();
    let runner_stem = "eismaster_batch_drt";

    let args: Vec<String> = vec![
        input_dir.to_string_lossy().to_string(),
        output_dir.to_string_lossy().to_string(),
        config.drttools_dir.clone(),
        config.method_tag.clone(),
        config.drt_type.to_string(),
        format_g12(config.lambda_value),
        format_g12(config.coeff_value),
        config.derivative_order.clone(),
        config.data_used.clone(),
        config.inductance_mode.to_string(),
        config.shape_control.clone(),
    ];

    let quoted_args: Vec<String> = args.iter().map(|a| matlab_quote(a)).collect();
    format!(
        "addpath('{}'); {}({});",
        runner_dir_s.replace('\'', "''"),
        runner_stem,
        quoted_args.join(", ")
    )
}

/// Build the full command vector: [matlab_exe, "-batch", batch_string]
pub fn build_matlab_command(matlab_exe: &str, batch_string: &str) -> Vec<String> {
    vec![
        matlab_exe.to_string(),
        "-batch".to_string(),
        batch_string.to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Run MATLAB and collect output.
///
/// Uses `std::process::Command` with the `-batch` flag. Blocks until MATLAB exits.
pub fn run_matlab_process(
    matlab_exe: &str,
    batch_string: &str,
) -> Result<(Option<i32>, String, String), String> {
    let command = build_matlab_command(matlab_exe, batch_string);

    let output = std::process::Command::new(&command[0])
        .args(&command[1..])
        .output()
        .map_err(|e| format!("failed to execute '{}': {e}", matlab_exe))?;

    Ok((
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Output discovery
// ---------------------------------------------------------------------------

/// Discover `*_DRT.txt` files in the output directory.
///
/// Returns sorted list of file paths.
pub fn discover_drt_output_files(output_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(output_dir)
        .map_err(|e| format!("cannot read output dir: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        })
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.ends_with("_DRT"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

/// Run the complete MATLAB DRT pipeline: stage, execute, discover, export workbook.
///
/// Returns a `MatlabDrtResult` with all details.
pub fn execute_matlab_drt_pipeline(
    config: &MatlabDrtConfig,
    spectra: &[SpectrumData],
    output_dir: &Path,
    line_x: &str,
    logtau_breaks: &[f64],
) -> Result<MatlabDrtResult, String> {
    if config.matlab_exe.is_empty() {
        return Err("MATLAB executable path is empty".to_string());
    }
    if config.drttools_dir.is_empty() {
        return Err("DRTtools directory path is empty".to_string());
    }
    if config.matlab_bridge_dir.is_empty() {
        return Err("MATLAB bridge directory path is empty".to_string());
    }
    if spectra.is_empty() {
        return Err("no spectra provided for MATLAB DRT".to_string());
    }

    // Ensure output dir exists
    std::fs::create_dir_all(output_dir).map_err(|e| format!("failed to create output dir: {e}"))?;

    // Stage inputs
    let staging_dir = stage_matlab_drt_inputs(spectra, output_dir)?;

    // Build MATLAB command
    let runner_dir = Path::new(&config.matlab_bridge_dir);
    let batch_string = build_matlab_batch_string(runner_dir, &staging_dir, output_dir, config);
    let command_vec = build_matlab_command(&config.matlab_exe, &batch_string);
    let command_display = command_vec.join(" ");

    // Run MATLAB
    let (return_code, stdout, stderr) = run_matlab_process(&config.matlab_exe, &batch_string)?;

    // Discover output files
    let output_files = match discover_drt_output_files(output_dir) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("output discovery warning: {e}");
            Vec::new()
        }
    };

    // Export workbook if we have DRT files
    let mut workbook_path: Option<String> = None;
    if !output_files.is_empty() {
        match export_workbook_from_drt_files(output_dir, line_x, logtau_breaks) {
            Ok(path) => workbook_path = Some(path),
            Err(e) => eprintln!("workbook export warning: {e}"),
        }
    }

    Ok(MatlabDrtResult {
        command: command_display,
        return_code,
        stdout,
        stderr,
        staging_dir: staging_dir.to_string_lossy().to_string(),
        output_dir: output_dir.to_string_lossy().to_string(),
        output_files: output_files
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        workbook_path,
    })
}

/// Export drt_matrix.xlsx from discovered DRT files using existing core functions.
fn export_workbook_from_drt_files(
    output_dir: &Path,
    line_x: &str,
    logtau_breaks: &[f64],
) -> Result<String, String> {
    use crate::drt::{label_from_stem, parse_drt_file};
    use crate::drt_export::{write_drt_workbook, LineXAxis};

    let drt_files = discover_drt_output_files(output_dir)?;

    let mut curves = Vec::new();
    for path in &drt_files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        // Remove _DRT suffix to get original stem for label
        let orig_stem = stem.strip_suffix("_DRT").unwrap_or(stem);
        let label = label_from_stem(orig_stem);
        match parse_drt_file(path, label) {
            Ok(curve) => curves.push(curve),
            Err(e) => eprintln!("skipping {}: {e}", path.display()),
        }
    }

    if curves.is_empty() {
        return Err("no valid DRT curves for workbook export".to_string());
    }

    let x_axis = match line_x {
        "tau" => LineXAxis::Tau,
        _ => LineXAxis::LogTau,
    };

    let workbook_path = output_dir.join("drt_matrix.xlsx");
    write_drt_workbook(&workbook_path, &curves, x_axis, logtau_breaks)
        .map_err(|e| format!("xlsx write error: {e}"))?;

    Ok(workbook_path.to_string_lossy().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SpectrumData, SpectrumMetadata};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn make_spectrum(path: &str, freq: Vec<f64>, zr: Vec<f64>, zi: Vec<f64>) -> SpectrumData {
        let n = freq.len();
        SpectrumData {
            metadata: SpectrumMetadata {
                file_path: PathBuf::from(path),
                technique: "A.C. Impedance".to_string(),
                instrument_model: "CHI660F".to_string(),
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

    fn tmp_dir() -> PathBuf {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test_tmp_matlab");
        std::fs::create_dir_all(&p).ok();
        p
    }

    #[test]
    fn staging_writes_raw_z_imag() {
        let dir = tmp_dir().join("staging_test");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }

        let spectrum = make_spectrum(
            r"C:\data\Ag_EIS_OCV.txt",
            vec![100000.0, 10.0, 0.01],
            vec![5.74, 200.0, 430.3],
            vec![-0.4091, -500.0, -2879.0],
        );

        let staging = stage_matlab_drt_inputs(&[spectrum], &dir).unwrap();
        assert!(staging.is_dir());

        let input_file = staging.join("Ag_EIS_OCV.txt");
        assert!(input_file.exists());

        let content = std::fs::read_to_string(&input_file).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 3);

        // First line: freq=100000, z_real=5.74, z_imag=-0.4091
        let parts: Vec<&str> = lines[0].split('\t').collect();
        assert_eq!(parts.len(), 3);
        let z_imag: f64 = parts[2].parse().unwrap();
        assert!((z_imag - (-0.4091)).abs() < 1e-10);

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn staging_preserves_file_stems() {
        let dir = tmp_dir().join("staging_stems_test");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }

        let s1 = make_spectrum(r"C:\data\sample_A.txt", vec![1.0], vec![1.0], vec![1.0]);
        let s2 = make_spectrum(r"C:\data\sample_B.txt", vec![2.0], vec![2.0], vec![2.0]);

        let staging = stage_matlab_drt_inputs(&[s1, s2], &dir).unwrap();
        assert!(staging.join("sample_A.txt").exists());
        assert!(staging.join("sample_B.txt").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matlab_quote_escapes_single_quotes() {
        assert_eq!(matlab_quote("hello"), "'hello'");
        assert_eq!(matlab_quote("it's"), "'it''s'");
        assert_eq!(matlab_quote("a'b'c"), "'a''b''c'");
        assert_eq!(matlab_quote(""), "''");
    }

    #[test]
    fn matlab_quote_handles_paths_with_spaces() {
        let path = r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe";
        let quoted = matlab_quote(path);
        assert_eq!(
            quoted,
            "'C:\\Program Files\\MATLAB\\R2024b\\bin\\matlab.exe'"
        );
    }

    #[test]
    fn matlab_quote_handles_chinese_chars() {
        let path = r"C:\用户\数据\spectrum.txt";
        let quoted = matlab_quote(path);
        assert!(quoted.starts_with('\''));
        assert!(quoted.ends_with('\''));
        assert!(quoted.contains("用户"));
    }

    #[test]
    fn batch_command_contains_batch_flag() {
        let config = MatlabDrtConfig {
            matlab_exe: r"D:\Matlabs\bin\matlab.EXE".to_string(),
            drttools_dir: r"C:\tools\DRTtools".to_string(),
            matlab_bridge_dir: r"C:\repos\matlab_bridge".to_string(),
            ..Default::default()
        };

        let batch = build_matlab_batch_string(
            Path::new(&config.matlab_bridge_dir),
            Path::new(r"C:\tmp\inputs"),
            Path::new(r"C:\tmp\outputs"),
            &config,
        );

        let cmd = build_matlab_command(&config.matlab_exe, &batch);
        assert_eq!(cmd.len(), 3);
        assert_eq!(cmd[1], "-batch");
        assert!(cmd[2].contains("addpath"));
        assert!(cmd[2].contains("eismaster_batch_drt"));
    }

    #[test]
    fn batch_command_has_11_args() {
        let config = MatlabDrtConfig::default();
        let batch = build_matlab_batch_string(
            Path::new("/bridge"),
            Path::new("/input"),
            Path::new("/output"),
            &config,
        );
        // Count commas between parens of eismaster_batch_drt(...)
        let inner = batch.split("eismaster_batch_drt(").nth(1).unwrap();
        let inner = inner.split(");").next().unwrap();
        let arg_count = inner.matches("', '").count() + 1;
        assert_eq!(arg_count, 11, "MATLAB bridge expects exactly 11 arguments");
    }

    #[test]
    fn discover_drt_output_files_finds_correct_pattern() {
        let dir = tmp_dir().join("discover_test");
        if dir.exists() {
            std::fs::remove_dir_all(&dir).ok();
        }
        std::fs::create_dir_all(&dir).unwrap();

        // Create test files
        std::fs::write(dir.join("sample1_DRT.txt"), "data").unwrap();
        std::fs::write(dir.join("sample2_DRT.txt"), "data").unwrap();
        std::fs::write(dir.join("not_drt.txt"), "data").unwrap();
        std::fs::write(dir.join("summary.txt"), "data").unwrap();

        let files = discover_drt_output_files(&dir).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].to_string_lossy().contains("sample1_DRT.txt"));
        assert!(files[1].to_string_lossy().contains("sample2_DRT.txt"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_matlab_command_with_spaces_in_path() {
        let exe = r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe";
        let batch = "addpath('C:\\bridge'); eismaster_batch_drt('arg1');";
        let cmd = build_matlab_command(exe, batch);
        assert_eq!(cmd[0], exe);
        assert_eq!(cmd[1], "-batch");
        assert_eq!(cmd[2], batch);
    }

    #[test]
    fn no_matlab_exe_gives_clear_error() {
        let config = MatlabDrtConfig {
            matlab_exe: String::new(),
            ..Default::default()
        };
        let spectrum = make_spectrum("test.txt", vec![1.0], vec![1.0], vec![1.0]);
        let dir = tmp_dir().join("error_test");
        let result =
            execute_matlab_drt_pipeline(&config, &[spectrum], &dir, "logtau", &[-3.0, 0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("MATLAB executable"));
    }

    #[test]
    fn no_drttools_dir_gives_clear_error() {
        let config = MatlabDrtConfig {
            matlab_exe: "matlab".to_string(),
            drttools_dir: String::new(),
            ..Default::default()
        };
        let spectrum = make_spectrum("test.txt", vec![1.0], vec![1.0], vec![1.0]);
        let dir = tmp_dir().join("error_test2");
        let result =
            execute_matlab_drt_pipeline(&config, &[spectrum], &dir, "logtau", &[-3.0, 0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("DRTtools"));
    }

    #[test]
    fn no_spectra_gives_clear_error() {
        let config = MatlabDrtConfig {
            matlab_exe: "matlab".to_string(),
            drttools_dir: "/tools".to_string(),
            matlab_bridge_dir: "/bridge".to_string(),
            ..Default::default()
        };
        let dir = tmp_dir().join("error_test3");
        let result = execute_matlab_drt_pipeline(&config, &[], &dir, "logtau", &[-3.0, 0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no spectra"));
    }
}
