use std::path::{Path, PathBuf};

use eismaster_core::chi_bin::parse_chi_bin;
use eismaster_core::chi_txt::parse_chi_txt;
use eismaster_core::drt::{label_from_stem, parse_drt_file};
use eismaster_core::drt_export::{write_drt_workbook, LineXAxis};
use eismaster_core::matlab_drt::{execute_matlab_drt_pipeline, MatlabDrtConfig, MatlabDrtResult};
use eismaster_core::models::SpectrumData;
use serde::Deserialize;
use tauri::command;
use tauri::Manager;

/// Serializable wrapper for DRT export settings.
#[derive(Debug, Deserialize)]
pub struct DrtExportRequest {
    pub spectra_dir: PathBuf,
    pub drt_dir: PathBuf,
    pub output_path: PathBuf,
    #[serde(default = "default_line_x")]
    pub line_x: String,
    #[serde(default = "default_logtau_breaks")]
    pub logtau_breaks: Vec<f64>,
}

fn default_line_x() -> String {
    "logtau".to_string()
}

fn default_logtau_breaks() -> Vec<f64> {
    vec![-3.0, 0.0]
}

/// Parse a single EIS file (.txt, .csv, .bin) and return its data as JSON.
#[command]
pub fn parse_file(path: String) -> Result<SpectrumData, String> {
    let p = Path::new(&path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "bin" => parse_chi_bin(p),
        _ => parse_chi_txt(p),
    }
}

/// Parse all EIS files (.txt, .csv, .bin) in a folder and return their data,
/// sorted by acquisition timestamp then filename (matching Python sort_key_for_spectrum).
#[command]
pub fn parse_folder(dir: String) -> Result<Vec<SpectrumData>, String> {
    let dir_path = Path::new(&dir);
    if !dir_path.is_dir() {
        return Err(format!("not a directory: {dir}"));
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir_path)
        .map_err(|e| format!("cannot read directory: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "txt" | "csv" | "bin"))
                .unwrap_or(false)
        })
        .collect();

    let mut results = Vec::new();
    for path in &entries {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        let result = match ext.as_str() {
            "bin" => parse_chi_bin(path),
            _ => parse_chi_txt(path),
        };
        match result {
            Ok(spectrum) => results.push(spectrum),
            Err(e) => eprintln!("skipping {}: {e}", path.display()),
        }
    }

    if results.is_empty() {
        return Err("no valid EIS files found in directory".to_string());
    }

    // Sort by (acquired_at, file_name.to_lowercase()) — matches Python sort_key_for_spectrum
    // None timestamps sort first (Python uses datetime.min)
    results.sort_by(|a, b| {
        let dt_cmp = match (a.metadata.acquired_at, b.metadata.acquired_at) {
            (Some(a_dt), Some(b_dt)) => a_dt.cmp(&b_dt),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        dt_cmp.then_with(|| {
            a.metadata
                .file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase()
                .cmp(
                    &b.metadata
                        .file_path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_lowercase(),
                )
        })
    });

    Ok(results)
}

/// Export DRT workbook from matched spectra/DRT file pairs.
#[command]
pub fn export_drt_workbook(req: DrtExportRequest) -> Result<String, String> {
    let spectra_dir = &req.spectra_dir;
    let drt_dir = &req.drt_dir;

    if !spectra_dir.is_dir() {
        return Err(format!("spectra dir not found: {}", spectra_dir.display()));
    }
    if !drt_dir.is_dir() {
        return Err(format!("DRT dir not found: {}", drt_dir.display()));
    }

    // Collect spectrum stems: stem.txt -> look for stem_DRT.txt
    let mut stems: Vec<String> = std::fs::read_dir(spectra_dir)
        .map_err(|e| format!("cannot read spectra dir: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        })
        .filter_map(|e| {
            e.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .collect();

    stems.sort();

    let mut curves = Vec::new();
    for stem in &stems {
        let drt_path = drt_dir.join(format!("{stem}_DRT.txt"));
        if !drt_path.exists() {
            eprintln!("no DRT file for {stem}, skipping");
            continue;
        }
        let label = label_from_stem(stem);
        match parse_drt_file(&drt_path, label) {
            Ok(curve) => curves.push(curve),
            Err(e) => eprintln!("failed to parse {}: {e}", drt_path.display()),
        }
    }

    if curves.is_empty() {
        return Err("no DRT curves found to export".to_string());
    }

    let line_x = match req.line_x.as_str() {
        "tau" => LineXAxis::Tau,
        _ => LineXAxis::LogTau,
    };

    write_drt_workbook(&req.output_path, &curves, line_x, &req.logtau_breaks)
        .map_err(|e| format!("xlsx write error: {e}"))?;

    Ok(format!(
        "Exported {} curves to {}",
        curves.len(),
        req.output_path.display()
    ))
}

// ---------------------------------------------------------------------------
// MATLAB DRT
// ---------------------------------------------------------------------------

/// Frontend request shape for MATLAB DRT execution.
#[derive(Debug, Deserialize)]
pub struct MatlabDrtRequest {
    pub spectra_paths: Vec<String>,
    pub output_dir: String,
    pub matlab_exe: String,
    pub drttools_dir: String,
    pub matlab_bridge_dir: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_drt_type")]
    pub drt_type: u32,
    #[serde(default = "default_lambda")]
    pub lambda_value: f64,
    #[serde(default = "default_coeff")]
    pub coeff_value: f64,
    #[serde(default = "default_inductance")]
    pub inductance_mode: u32,
    #[serde(default = "default_derivative")]
    pub derivative_order: String,
    #[serde(default = "default_data_used")]
    pub data_used: String,
    #[serde(default = "default_shape")]
    pub shape_control: String,
    #[serde(default = "default_line_x")]
    pub line_x_axis: String,
    pub logtau_breaks: Option<Vec<f64>>,
}

fn default_method() -> String {
    "simple".to_string()
}
fn default_drt_type() -> u32 {
    2
}
fn default_lambda() -> f64 {
    1e-3
}
fn default_coeff() -> f64 {
    0.5
}
fn default_inductance() -> u32 {
    1
}
fn default_derivative() -> String {
    "1st-order".to_string()
}
fn default_data_used() -> String {
    "Combined Re-Im Data".to_string()
}
fn default_shape() -> String {
    "FWHM Coefficient".to_string()
}

/// Run MATLAB DRT on selected spectra and export workbook.
#[command]
pub fn run_matlab_drt(req: MatlabDrtRequest) -> Result<MatlabDrtResult, String> {
    // Parse input spectra
    let mut spectra = Vec::new();
    for path_str in &req.spectra_paths {
        match parse_chi_txt(Path::new(path_str)) {
            Ok(s) => spectra.push(s),
            Err(e) => return Err(format!("failed to parse {}: {e}", path_str)),
        }
    }

    let config = MatlabDrtConfig {
        matlab_exe: req.matlab_exe,
        drttools_dir: req.drttools_dir,
        matlab_bridge_dir: req.matlab_bridge_dir,
        method_tag: req.method,
        drt_type: req.drt_type,
        lambda_value: req.lambda_value,
        coeff_value: req.coeff_value,
        derivative_order: req.derivative_order,
        data_used: req.data_used,
        inductance_mode: req.inductance_mode,
        shape_control: req.shape_control,
    };

    let output_dir = Path::new(&req.output_dir);
    let breaks = req.logtau_breaks.unwrap_or_else(|| vec![-3.0, 0.0]);

    execute_matlab_drt_pipeline(&config, &spectra, output_dir, &req.line_x_axis, &breaks)
}

/// Discover DRT output files in a directory (for manual re-export).
#[command]
pub fn discover_drt_files(dir: String) -> Result<Vec<String>, String> {
    let files = eismaster_core::matlab_drt::discover_drt_output_files(Path::new(&dir))?;
    Ok(files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect())
}

/// Request for re-exporting workbook from DRT dir.
#[derive(Debug, Deserialize)]
pub struct ExportFromDrtDirRequest {
    pub drt_dir: String,
    pub output_path: String,
    #[serde(default = "default_line_x")]
    pub line_x: String,
    #[serde(default = "default_logtau_breaks")]
    pub logtau_breaks: Vec<f64>,
}

/// Re-export drt_matrix.xlsx from existing DRT files.
#[command]
pub fn export_workbook_from_drt_dir(req: ExportFromDrtDirRequest) -> Result<String, String> {
    let dir = Path::new(&req.drt_dir);
    let drt_files = eismaster_core::matlab_drt::discover_drt_output_files(dir)?;

    let mut curves = Vec::new();
    for path in &drt_files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let orig_stem = stem.strip_suffix("_DRT").unwrap_or(stem);
        let label = label_from_stem(orig_stem);
        match parse_drt_file(path, label) {
            Ok(curve) => curves.push(curve),
            Err(e) => eprintln!("skipping {}: {e}", path.display()),
        }
    }

    if curves.is_empty() {
        return Err("no valid DRT curves found".to_string());
    }

    let x_axis = match req.line_x.as_str() {
        "tau" => LineXAxis::Tau,
        _ => LineXAxis::LogTau,
    };

    let out = Path::new(&req.output_path);
    write_drt_workbook(out, &curves, x_axis, &req.logtau_breaks)
        .map_err(|e| format!("xlsx write error: {e}"))?;

    Ok(format!(
        "Exported {} curves to {}",
        curves.len(),
        out.display()
    ))
}

/// Get resource paths for matlab_bridge and matlab-DRTtools-local.
///
/// In dev mode: resolves relative to CARGO_MANIFEST_DIR.
/// In packaged mode: resolves relative to the Tauri resource directory.
#[command]
pub fn get_dev_resource_paths(app: tauri::AppHandle) -> Result<DevResourcePaths, String> {
    // Try packaged resource dir first
    if let Ok(res_dir) = app.path().resource_dir() {
        let bridge = res_dir.join("matlab_bridge");
        let drttools = res_dir.join("matlab-DRTtools-local");
        if bridge.is_dir() || drttools.is_dir() {
            return Ok(DevResourcePaths {
                matlab_bridge_dir: bridge
                    .is_dir()
                    .then(|| bridge.to_string_lossy().to_string()),
                drttools_dir: drttools
                    .is_dir()
                    .then(|| drttools.to_string_lossy().to_string()),
            });
        }
    }

    // Fallback: dev mode paths
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rewrite_dir = crate_dir.join("../..");
    let repo_root = rewrite_dir.join("../..");

    let matlab_bridge = repo_root.join("matlab_bridge");
    let drttools = repo_root.join("matlab-DRTtools-local");

    Ok(DevResourcePaths {
        matlab_bridge_dir: if matlab_bridge.is_dir() {
            matlab_bridge
                .canonicalize()
                .map(|p| p.to_string_lossy().to_string())
                .ok()
        } else {
            None
        },
        drttools_dir: if drttools.is_dir() {
            drttools
                .canonicalize()
                .map(|p| p.to_string_lossy().to_string())
                .ok()
        } else {
            None
        },
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DevResourcePaths {
    pub matlab_bridge_dir: Option<String>,
    pub drttools_dir: Option<String>,
}

// ---------------------------------------------------------------------------
// Spectrum inspection
// ---------------------------------------------------------------------------

use eismaster_core::circuits::CircuitTemplate as CoreCircuitTemplate;
use eismaster_core::quality::QualityReport;
use eismaster_core::segmentation::SegmentDetection;

#[derive(Debug, Clone, serde::Serialize)]
pub struct InspectResult {
    pub file: String,
    pub n_points: usize,
    pub quality: QualityReport,
    pub segmentation: SegmentDetection,
    pub circuit_templates: Vec<TemplateInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TemplateInfo {
    pub key: String,
    pub label: String,
    pub parameter_names: Vec<String>,
}

/// Inspect a single EIS file: quality, segmentation, circuit templates.
#[command]
pub fn inspect_spectrum(path: String) -> Result<InspectResult, String> {
    let spectrum = parse_chi_txt(Path::new(&path))?;
    let quality = eismaster_core::quality::assess_spectrum_quality(&spectrum);
    let segmentation =
        eismaster_core::segmentation::detect_segments(&spectrum, "auto", None, None, None, None);
    let templates: Vec<TemplateInfo> = eismaster_core::circuits::templates()
        .iter()
        .map(|t: &CoreCircuitTemplate| TemplateInfo {
            key: t.key.to_string(),
            label: t.label.to_string(),
            parameter_names: t.parameter_names.iter().map(|s| s.to_string()).collect(),
        })
        .collect();

    Ok(InspectResult {
        file: path,
        n_points: spectrum.n_points(),
        quality,
        segmentation,
        circuit_templates: templates,
    })
}

// ---------------------------------------------------------------------------
// Circuit fitting
// ---------------------------------------------------------------------------

use eismaster_core::fitting::BatchSummary as CoreBatchSummary;
use eismaster_core::fitting::FitOutcome as CoreFitOutcome;

/// Fit a circuit model to a spectrum.
#[command]
pub fn fit_spectrum(path: String, model_key: String) -> Result<CoreFitOutcome, String> {
    let spectrum = parse_chi_txt(Path::new(&path))?;
    Ok(eismaster_core::fitting::fit_spectrum(&spectrum, &model_key))
}

/// Fit a circuit model to all .txt/.csv spectra in a folder.
#[command]
pub fn fit_batch_folder(dir: String, model_key: String) -> Result<CoreBatchSummary, String> {
    eismaster_core::fitting::fit_batch(Path::new(&dir), &model_key)
}

/// Fit a circuit model to a list of spectrum file paths.
///
/// Matches Python behavior where batch operates on already-loaded spectra,
/// not re-scanning a directory. Supports .txt/.csv/.bin via extension dispatch.
#[command]
pub fn fit_batch_paths(paths: Vec<String>, model_key: String) -> Result<CoreBatchSummary, String> {
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    eismaster_core::fitting::fit_batch_paths(&path_bufs, &model_key)
}

/// Export a batch fit summary to an XLSX workbook.
#[command]
pub fn export_batch_workbook(
    output_path: String,
    dir: String,
    model_key: String,
) -> Result<String, String> {
    let summary = eismaster_core::fitting::fit_batch(Path::new(&dir), &model_key)?;
    eismaster_core::fitting::write_batch_workbook(Path::new(&output_path), &summary)
        .map_err(|e| format!("xlsx write error: {e}"))?;
    Ok(format!(
        "Exported {} items to {}",
        summary.n_total, output_path
    ))
}

/// Export a single spectrum's fit result to an XLSX workbook.
#[command]
pub fn export_single_fit(
    output_path: String,
    spectrum_path: String,
    model_key: String,
) -> Result<String, String> {
    let path = Path::new(&spectrum_path);
    let spectrum = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("bin") => parse_chi_bin(path),
        _ => parse_chi_txt(path),
    }
    .map_err(|e| format!("解析失败: {e}"))?;

    let outcome = eismaster_core::fitting::fit_spectrum(&spectrum, &model_key);
    eismaster_core::fitting::write_single_fit_workbook(
        Path::new(&output_path),
        &spectrum,
        &outcome,
    )
    .map_err(|e| format!("xlsx write error: {e}"))?;
    Ok(format!("已导出拟合结果到 {}", output_path))
}

/// Export a batch fit from explicit paths (for loaded spectra).
#[command]
pub fn export_batch_workbook_from_paths(
    output_path: String,
    paths: Vec<String>,
    model_key: String,
) -> Result<String, String> {
    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let summary = eismaster_core::fitting::fit_batch_paths(&path_bufs, &model_key)?;
    eismaster_core::fitting::write_batch_workbook(Path::new(&output_path), &summary)
        .map_err(|e| format!("xlsx write error: {e}"))?;
    Ok(format!(
        "Exported {} items to {}",
        summary.n_total, output_path
    ))
}
