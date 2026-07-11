use std::path::{Path, PathBuf};
use std::process;

use serde::Serialize;

#[cfg(test)]
mod tests {
    use super::wants_help;

    #[test]
    fn help_flag_is_recognized() {
        assert!(wants_help(Some("--help")));
        assert!(wants_help(Some("-h")));
        assert!(!wants_help(Some("parse")));
    }
}

fn wants_help(arg: Option<&str>) -> bool {
    matches!(arg, Some("--help") | Some("-h"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if wants_help(args.get(1).map(String::as_str)) {
        print_usage();
        return;
    }

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "parse" => cmd_parse(&args),
        "drt-export" => cmd_drt_export(&args),
        "inspect" => cmd_inspect(&args),
        "inspect-folder" => cmd_inspect_folder(&args),
        "fit" => cmd_fit(&args),
        "fit-folder" => cmd_fit_folder(&args),
        "list-models" => cmd_list_models(),
        _ => {
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  eismaster-cli parse <path>");
    eprintln!("  eismaster-cli drt-export <spectra-stems-dir> <drt-results-dir> <output.xlsx> [--line-x logtau|tau] [--logtau-breaks \"-3,0\"]");
    eprintln!(
        "  eismaster-cli inspect <file>           — quality + segmentation + circuit templates"
    );
    eprintln!(
        "  eismaster-cli inspect-folder <dir>      — inspect all .txt/.csv files in a folder"
    );
    eprintln!("  eismaster-cli fit <file> --model <key>  — fit a circuit model");
    eprintln!(
        "  eismaster-cli fit-folder <dir> --model <key> [--output <json>] [--export-xlsx <path>]"
    );
    eprintln!("  eismaster-cli list-models               — list available models");
}

fn cmd_parse(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: parse requires a file path");
        process::exit(1);
    }
    let path = PathBuf::from(&args[2]);
    match eismaster_core::chi_txt::parse_chi_txt(&path) {
        Ok(spectrum) => {
            let json = serde_json::to_string_pretty(&spectrum).expect("failed to serialize JSON");
            println!("{json}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn cmd_drt_export(args: &[String]) {
    if args.len() < 5 {
        eprintln!("error: drt-export requires <spectra-stems-dir> <drt-results-dir> <output.xlsx>");
        process::exit(1);
    }

    let spectra_dir = Path::new(&args[2]);
    let drt_dir = Path::new(&args[3]);
    let output_path = Path::new(&args[4]);

    let mut line_x = eismaster_core::drt_export::LineXAxis::LogTau;
    let mut logtau_breaks: Vec<f64> = vec![-3.0, 0.0];

    // Parse optional flags
    let mut i = 5;
    while i < args.len() {
        match args[i].as_str() {
            "--line-x" => {
                i += 1;
                match args.get(i).map(|s| s.as_str()) {
                    Some("tau") => line_x = eismaster_core::drt_export::LineXAxis::Tau,
                    Some("logtau") => line_x = eismaster_core::drt_export::LineXAxis::LogTau,
                    other => {
                        eprintln!("error: unknown --line-x value: {:?}", other);
                        process::exit(1);
                    }
                }
            }
            "--logtau-breaks" => {
                i += 1;
                let raw = args.get(i).unwrap_or_else(|| {
                    eprintln!("error: --logtau-breaks requires a value like \"-3,0\"");
                    process::exit(1);
                });
                logtau_breaks = raw
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
                if logtau_breaks.is_empty() {
                    logtau_breaks = vec![-3.0, 0.0];
                }
            }
            other => {
                eprintln!("error: unknown flag: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    // Collect stems from spectra directory
    let mut stems: Vec<String> = Vec::new();
    if spectra_dir.is_dir() {
        for entry in std::fs::read_dir(spectra_dir).unwrap_or_else(|e| {
            eprintln!("error: cannot read spectra dir: {e}");
            process::exit(1);
        }) {
            if let Ok(entry) = entry {
                let path = entry.path();
                if path.is_file() {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        stems.push(stem.to_string());
                    }
                }
            }
        }
    }

    stems.sort();

    let mut curves = Vec::new();
    for stem in &stems {
        let drt_path = drt_dir.join(format!("{stem}_DRT.txt"));
        if !drt_path.exists() {
            continue;
        }
        let label = eismaster_core::drt::label_from_stem(stem);
        match eismaster_core::drt::parse_drt_file(&drt_path, label) {
            Ok(curve) => curves.push(curve),
            Err(e) => eprintln!("warning: skipping {stem}: {e}"),
        }
    }

    if curves.is_empty() {
        eprintln!("error: no DRT files found matching spectra stems");
        process::exit(1);
    }

    match eismaster_core::drt_export::write_drt_workbook(
        output_path,
        &curves,
        line_x,
        &logtau_breaks,
    ) {
        Ok(()) => {
            eprintln!("wrote {} to {}", curves.len(), output_path.display());
        }
        Err(e) => {
            eprintln!("error: failed to write workbook: {e}");
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Inspect commands
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct InspectResult {
    file: String,
    n_points: usize,
    quality: eismaster_core::quality::QualityReport,
    segmentation: eismaster_core::segmentation::SegmentDetection,
    circuit_templates: Vec<TemplateInfo>,
}

#[derive(Serialize)]
struct TemplateInfo {
    key: String,
    label: String,
    parameter_names: Vec<String>,
}

fn inspect_spectrum(path: &Path) -> Result<InspectResult, String> {
    let spectrum = eismaster_core::chi_txt::parse_chi_txt(path)?;
    let quality = eismaster_core::quality::assess_spectrum_quality(&spectrum);
    let segmentation =
        eismaster_core::segmentation::detect_segments(&spectrum, "auto", None, None, None, None);
    let templates: Vec<TemplateInfo> = eismaster_core::circuits::templates()
        .iter()
        .map(|t| TemplateInfo {
            key: t.key.to_string(),
            label: t.label.to_string(),
            parameter_names: t.parameter_names.iter().map(|s| s.to_string()).collect(),
        })
        .collect();

    Ok(InspectResult {
        file: path.to_string_lossy().to_string(),
        n_points: spectrum.n_points(),
        quality,
        segmentation,
        circuit_templates: templates,
    })
}

fn cmd_inspect(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: inspect requires a file path");
        process::exit(1);
    }
    let path = PathBuf::from(&args[2]);
    match inspect_spectrum(&path) {
        Ok(result) => {
            let json = serde_json::to_string_pretty(&result).expect("failed to serialize JSON");
            println!("{json}");
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn cmd_inspect_folder(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: inspect-folder requires a directory path");
        process::exit(1);
    }
    let dir = Path::new(&args[2]);
    if !dir.is_dir() {
        eprintln!("error: not a directory: {}", dir.display());
        process::exit(1);
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| {
            eprintln!("error: cannot read directory: {e}");
            process::exit(1);
        })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_lowercase().as_str(), "txt" | "csv"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    if entries.is_empty() {
        eprintln!("error: no .txt/.csv files found in {}", dir.display());
        process::exit(1);
    }

    let mut results = Vec::new();
    for path in &entries {
        match inspect_spectrum(path) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("warning: skipping {}: {e}", path.display()),
        }
    }

    if results.is_empty() {
        eprintln!("error: no files could be inspected");
        process::exit(1);
    }

    let json = serde_json::to_string_pretty(&results).expect("failed to serialize JSON");
    println!("{json}");
}

// ---------------------------------------------------------------------------
// Fit commands
// ---------------------------------------------------------------------------

fn cmd_fit(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: fit requires a file path");
        process::exit(1);
    }

    let path = PathBuf::from(&args[2]);
    let mut model_key = String::new();

    // Parse --model flag
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_key = args.get(i).cloned().unwrap_or_default();
            }
            other => {
                eprintln!("error: unknown flag: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    if model_key.is_empty() {
        eprintln!("error: --model <key> is required");
        eprintln!("Available models:");
        for t in eismaster_core::circuits::templates() {
            eprintln!("  {} — {}", t.key, t.label);
        }
        process::exit(1);
    }

    let spectrum = match eismaster_core::chi_txt::parse_chi_txt(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to parse {path:?}: {e}");
            process::exit(1);
        }
    };

    let outcome = eismaster_core::fitting::fit_spectrum(&spectrum, &model_key);
    let json = serde_json::to_string_pretty(&outcome).expect("failed to serialize JSON");
    println!("{json}");
}

fn cmd_fit_folder(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: fit-folder requires a directory path");
        process::exit(1);
    }

    let dir = PathBuf::from(&args[2]);
    let mut model_key = String::new();
    let mut output_json: Option<PathBuf> = None;
    let mut export_xlsx: Option<PathBuf> = None;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_key = args.get(i).cloned().unwrap_or_default();
            }
            "--output" => {
                i += 1;
                output_json = args.get(i).map(PathBuf::from);
            }
            "--export-xlsx" => {
                i += 1;
                export_xlsx = args.get(i).map(PathBuf::from);
            }
            other => {
                eprintln!("error: unknown flag: {other}");
                process::exit(1);
            }
        }
        i += 1;
    }

    if model_key.is_empty() {
        eprintln!("error: --model <key> is required");
        process::exit(1);
    }

    let summary = match eismaster_core::fitting::fit_batch(&dir, &model_key) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    eprintln!(
        "Batch fit: {} total, {} ok, {} warn, {} failed",
        summary.n_total, summary.n_ok, summary.n_warn, summary.n_failed
    );

    // Write JSON if requested
    if let Some(ref out_path) = output_json {
        let json = serde_json::to_string_pretty(&summary).expect("failed to serialize JSON");
        std::fs::write(out_path, &json).unwrap_or_else(|e| {
            eprintln!("error: failed to write JSON: {e}");
            process::exit(1);
        });
        eprintln!("JSON written to {}", out_path.display());
    } else {
        let json = serde_json::to_string_pretty(&summary).expect("failed to serialize JSON");
        println!("{json}");
    }

    // Export XLSX if requested
    if let Some(ref xlsx_path) = export_xlsx {
        match eismaster_core::fitting::write_batch_workbook(xlsx_path, &summary) {
            Ok(()) => {
                eprintln!("Workbook written to {}", xlsx_path.display());
            }
            Err(e) => {
                eprintln!("error: failed to write workbook: {e}");
                process::exit(1);
            }
        }
    }
}

fn cmd_list_models() {
    let models: Vec<serde_json::Value> = eismaster_core::circuits::templates()
        .iter()
        .map(|t| {
            serde_json::json!({
                "key": t.key,
                "label": t.label,
                "parameter_names": t.parameter_names,
                "primary_exports": t.primary_exports,
            })
        })
        .collect();
    let json = serde_json::to_string_pretty(&models).expect("failed to serialize JSON");
    println!("{json}");
}
