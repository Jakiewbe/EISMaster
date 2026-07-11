# Rewrite Notes

## Phase 2: DRT Parser + Workbook Export

### What's Implemented
- DRT file parser supporting three formats:
  - Continuous tau/gamma (`tau\tgamma(tau)` header)
  - Continuous freq/gamma (`freq\tgamma(freq)` header) with tau = 1/(2*pi*f) conversion
  - Peak-fit format (Gaussian reconstruction with 160 points, mu in natural-log space)
- Area computation using `dln_tau` bin widths (edge-midpoint scheme, matches Python).
- Label extraction from file stems (OCV, T5M, T100M preferred tokens).
- Time extraction from labels (T10M=10min, T30S=0.5min, T2H=120min, T1D=1440min).
- XLSX workbook export with three sheets: `drt_line_plot`, `drt_cloud_density`, `drt_quant_area`.
- CLI command: `eismaster-cli drt-export <spectra-dir> <drt-dir> <output.xlsx> [--line-x logtau|tau] [--logtau-breaks "-3,0"]`.
- Grid alignment via linear interpolation when curves have different logtau grids.

### Parity With Python Reference
- Area computation verified: Python test expects `total_area = 13.815510557964274` for fixture `tau=1,10,100; gamma=1,2,3`. Rust produces identical value.
- Label parsing matches Python `_export_label_from_stem`: "Ag_EIS_OCV" -> "OCV", "Ag_S01_EIS_T5M" -> "T5M".
- Time extraction matches Python `_time_minutes_from_label`: T10M=10, T30S=0.5, T2H=120, T1D=1440.
- Region naming matches Python `_format_logtau_bound`: -3,0 -> "logtau_lt_neg3", "logtau_neg3_to_0", "logtau_ge_0".
- Peak-fit: Python uses `np.exp(logtau_ln)` for tau, Rust uses same formula. 160-point linspace from mu-4*sigma to mu+4*sigma.
- Validated with 52 real MATLAB DRT output files (Ag_S01 batch): all parse and export successfully.

### Not Yet Implemented (vs Python `exporters.py`)
| Feature | Python support | Rust status |
|---|---|---|
| Credit/BHT DRT formats | 5-column and 3-column exports | Not parsed (only simple method used in batch) |
| `g(tau)` column (gamma*freq) | DRT type 3/4 | Not parsed |
| DRT only export (from source) | `write_drt_only_export()` | Partially: CLI handles matching |
| Interpolation for cloud density | `_interpolate_to_logtau()` | Done (linear interp, same behavior) |

### Test Coverage
14 tests total (11 core + 3 from Phase 1):
- `parse_tau_gamma_file`: tau/gamma header format
- `parse_freq_gamma_file`: freq/gamma header + conversion
- `parse_peak_fit_file`: Gaussian peak reconstruction
- `area_computation_matches_python`: verified against Python numeric reference
- `label_from_stem_ocv`, `label_from_stem_time`: label extraction
- `time_minutes_parsing`: T10M, T30S, T2H, T1D
- `export_workbook_has_three_sheets`: xlsx structure
- `export_with_tau_axis`: line-x tau mode
- `custom_logtau_breaks`, `default_logtau_breaks`: region naming

### Commands Run
```
cargo fmt                                      -> clean
cargo test                                     -> 14/14 passed
cargo run -p eismaster-cli -- drt-export ...   -> 52 curves exported (122KB xlsx)
```

---

## Phase 3: Tauri GUI Shell

### What's Implemented
- Tauri 2 desktop app with React/TypeScript/Vite frontend.
- Seven Tauri commands: `parse_file`, `parse_folder`, `export_drt_workbook`, `run_matlab_drt`, `discover_drt_files`, `export_workbook_from_drt_dir`, `get_dev_resource_paths`.
- Frontend layout: toolbar, left spectrum list, center data table, right panel (MATLAB DRT + manual DRT export), status bar.
- File import: single files (`.txt`/`.csv`) and folder batch import via HTML file input.
- Data preview: scrollable table with Freq, Z', Z'', |Z|, Phase columns.
- Optional Nyquist plot (canvas-based scatter).
- MATLAB DRT panel: full workflow (see Phase 3b below).
- Manual DRT export panel: spectra dir, DRT dir, output path, line-x mode, logtau breaks.
- Workspace: Tauri app added to workspace members.

### Not Yet Implemented
| Feature | Status |
|---|---|
| Native file dialogs (`tauri-plugin-dialog`) | Deferred — using HTML file input for now |
| Electron-style drag-and-drop import | Not started |
| Matplotlib-quality plots | Basic canvas scatter only |
| Spectrum metadata display | Partial (filename + technique in header) |
| Multi-spectrum overlay | Not started |

### How to Run
```
cd rewrite/apps/eismaster-tauri
npm install          # first time only
npx tauri dev        # launches dev window
```

### Build Validation
- `cargo test` → 26/26 passed
- `npx tsc --noEmit` → clean
- `npx vite build` → 156KB JS + 4KB CSS

---

## Phase 3b: MATLAB DRT GUI Integration

### What's Implemented
- `eismaster_core::matlab_drt` module with:
  - `stage_matlab_drt_inputs()` — writes tab-separated freq/z_real/z_imag files (raw z_imag, no sign flip, matches Python).
  - `build_matlab_batch_string()` — constructs `addpath('...'); eismaster_batch_drt('arg1', ...);` with proper MATLAB quoting.
  - `build_matlab_command()` — builds `[matlab_exe, "-batch", batch_string]` vector.
  - `run_matlab_process()` — executes via `std::process::Command`, collects stdout/stderr/exit code.
  - `discover_drt_output_files()` — finds `*_DRT.txt` files in output dir.
  - `execute_matlab_drt_pipeline()` — full pipeline: stage → run → discover → export workbook.
  - `export_workbook_from_drt_dir()` — standalone re-export from existing DRT files.
  - `get_dev_resource_paths()` — resolves `matlab_bridge` and `matlab-DRTtools-local` relative to repo root.
- Frontend MATLAB DRT panel with:
  - MATLAB exe, DRTtools dir, bridge dir, output dir fields.
  - Method selector (simple/credit/BHT/peak) with Chinese labels.
  - DRT type selector (tau/gamma, freq/gamma, tau/g, freq/g).
  - Lambda, coeff text inputs.
  - Inductance mode (保留/忽略/去除).
  - Line X-axis selector.
  - log(tau) breaks input.
  - Run/Clear Log buttons.
  - Running state disables run button.
  - Log panel showing command, return code, stdout, stderr, output files, workbook path.
- Dev-mode auto-detection: bridge and DRTtools dirs populated on app launch if found.

### Parity With Python Reference
- Staging format matches: tab-separated, 12 significant digits via g-style formatting.
- MATLAB command structure matches: `addpath('<dir>'); eismaster_batch_drt('<arg1>', ..., '<arg11>');` with 11 args.
- Quoting matches: single quotes escaped as `''`.
- Pipeline flow matches: stage → `-batch` call → collect `*_DRT.txt` → export `drt_matrix.xlsx`.
- Error handling: missing exe/drttools/spectra give clear error messages.

### Tests (12 new)
- `staging_writes_raw_z_imag` — verifies tab-separated output with correct z_imag sign.
- `staging_preserves_file_stems` — verifies file naming from original paths.
- `matlab_quote_escapes_single_quotes` — `'` → `''`.
- `matlab_quote_handles_paths_with_spaces` — spaces preserved.
- `matlab_quote_handles_chinese_chars` — Chinese characters preserved.
- `batch_command_contains_batch_flag` — command has `-batch` and `eismaster_batch_drt`.
- `batch_command_has_11_args` — exactly 11 arguments.
- `discover_drt_output_files_finds_correct_pattern` — only `*_DRT.txt` matched.
- `build_matlab_command_with_spaces_in_path` — spaces in exe path handled.
- `no_matlab_exe_gives_clear_error` — empty exe → readable error.
- `no_drttools_dir_gives_clear_error` — empty drttools → readable error.
- `no_spectra_gives_clear_error` — empty spectra → readable error.

### Known Limitations
- MATLAB execution blocks the Tauri command thread (acceptable for first version).
- No progress streaming from MATLAB.
- Credit/BHT DRT formats parsed by core but workbook only uses first 2 columns (tau/freq + gamma).
- `g(tau)` column in DRT type 3/4 output files parsed but not separately handled in workbook.
- Not tested with real MATLAB yet (no MATLAB on this machine).

---

## Phase 1: CHI TXT Parser (completed)

### What's Implemented
- CHI TXT header format parsing (date, technique, Instrument Model, Note, key-value headers).
- Plain numeric EIS text parsing (whitespace/tab/comma-separated, no headers).
- `SpectrumMetadata` and `SpectrumData` with all core arrays.
- Computed `z_mod_ohm` and `phase_deg` when columns are absent (matches Python `math.hypot` / `math.degrees(math.atan2)`).
- JSON output via CLI (`eismaster-cli parse <path>`).

### Not Yet Implemented (vs Python `chi.py`)
| Feature | Python support | Rust status |
|---|---|---|
| `.bin` binary format | `parse_chi_bin()` | Not started |
| `.csv` format | `parse_delimited_text()` | Not started |
| Folder batch loading | `load_spectra_from_folder()` | Not started |
| `SpectrumData.impedance` (complex) | property | Not needed for JSON |

### Parser Ambiguities Found
1. **Date line colon collision**: CHI header scan picks up datetime `:` as header key-value. Python has the same behavior.
2. **Units row assumption**: Parser skips 2 lines after `Freq/Hz`. Matches Python and real CHI instrument output.
3. **Phase sign convention**: Python `math.degrees(math.atan2(z_imag, z_real))` matches Rust `z_imag.atan2(z_r).to_degrees()`.

### Decisions
- `chrono::NaiveDateTime` — matches Python's timezone-naive `datetime`.
- `Vec<f64>` instead of numpy — no numeric library dependency.
- `rust_xlsxwriter` for xlsx — lightweight, cross-platform, no C dependencies.

---

## Phase 4: EIS Analysis (segmentation, quality, circuits)

### What's Implemented
- **Segmentation module** (`segmentation.rs`):
  - `detect_segments()` — auto/single/double arc detection from `-z_imag`.
  - `smooth_trace()` — 5-point weighted smoothing [1,2,3,2,1]/9 with edge padding.
  - `significant_peaks()` — local maxima with prominence filtering (max(y*0.01, 0.2)).
  - `valley_between()`, `valley_after()` — split point detection.
  - `fallback_double_peaks()` — for weak signals, finds 2 peaks ≥6 apart.
  - `sanitize_single_controls()`, `sanitize_double_controls()` — bounds clamping.
- **Quality module** (`quality.rs`):
  - `assess_spectrum_quality()` — returns `QualityReport` with status/issues/kk_status.
  - Checks: point count <8, non-finite, non-positive freq, non-descending, duplicates, negative z_real, inductive start, outlier count.
  - Outlier detection: curvature/slope/log-frequency MAD-based voting (matches Python `detect_outliers_common`).
  - KK/Z-HIT always "not_run" (Python-only).
- **Circuits module** (`circuits.rs`):
  - `z_cpe()` — CPE impedance `1/(T*(jw)^P)`.
  - `z_warburg_open()` — finite Warburg with numerical edge cases (|x|>50, |x|<1e-8).
  - `zview_single_model()` — R(QRWo) single-arc, 7 params.
  - `zview_double_model()` — R(QR)(Q(RWo)) double-arc, 10 params.
  - `templates()` — returns template metadata matching Python `CircuitTemplate`.
- **CLI** (`eismaster-cli`):
  - `inspect <file>` — quality + segmentation + circuit templates as JSON.
  - `inspect-folder <dir>` — batch inspect all .txt/.csv files.
- **Tauri GUI**:
  - `inspect_spectrum` command — returns InspectResult with quality/segmentation/templates.
  - "Spectrum Inspection" panel in right sidebar with Inspect Selected button.
  - Quality status (pass/warn/fail) with issue list, segmentation peaks/splits, template list.

### Parity With Python Reference
- Segmentation: smooth kernel, prominence threshold, valley logic all match Python.
- Quality checks: same thresholds, same Chinese messages, same severity levels.
- Outlier detection: curvature (scale=6), slope (scale=6), gradient (scale=7), vote_threshold=2 — matches Python.
- Circuit models: `_zview_full_model` and `_zview_double_model` formulae ported exactly.
- Template metadata: keys, labels, parameter_names match Python `TEMPLATES` dict.

### Not Yet Implemented (vs Python)
| Feature | Python support | Rust status |
|---|---|---|
| KK/Z-HIT via pyimpspec | `perform_kramers_kronig_test()` | Placeholder ("not_run") |
| `.bin` binary format | `parse_chi_bin()` | Not started |
| Fitting/optimizer | `scipy.optimize.least_squares` | Not started (user constraint) |
| FitOutcome model | `FitOutcome` dataclass | Not implemented |
| Confidence intervals | bootstrap | Not started |

### Test Coverage
59 tests total (33 new + 26 from prior phases):
- Segmentation: 11 tests (single/double/auto/manual, sanitize, smooth, valley)
- Quality: 12 tests (all checks + median/MAD utilities)
- Circuits: 10 tests (CPE, Warburg, single/double models, templates)

### Commands Run
```
cargo fmt                                      -> clean
cargo test                                     -> 59/59 passed
cargo run -p eismaster-cli -- inspect ...      -> JSON output correct
npx tsc --noEmit                               -> clean
npx vite build                                 -> 157KB JS + 4.5KB CSS
```

---

## Phase 5: Single-Spectrum Fitting Engine

### What's Implemented
- **Fitting module** (`fitting.rs`):
  - `FitOutcome` struct — matches Python fields: model_key, model_label, status, message, parameters, statistics, predicted_real/imag_ohm, masked_points, preprocess_actions, diagnosis fields.
  - `preprocess_mask()` — removes non-finite and inductive (z_imag > 0) points.
  - `fit_spectrum()` — top-level function: preprocess → initial guess → multi-start LM → statistics → diagnostics.
  - Custom bounded Levenberg-Marquardt optimizer:
    - Finite-difference Jacobian (forward, relative+absolute step).
    - Diagonal-damped normal equations solved via Gaussian elimination with partial pivoting.
    - Parameter bounds clamped after each step.
    - Convergence: relative RSS change < 1e-8.
  - Modulus weighting: divide residuals by max(|Z_exp|, adaptive floor).
  - Multi-start with perturbation factors (5 seeds for single, 4 for double).
  - Initial guess from segmentation:
    - Rs = min(Z_real), span = max - min, peak/split freq from segmentation.
    - CPE n estimated from log-log slope near peak.
    - Q = 1/(span * (2*pi*peak_freq)^n), Warburg from tail.
  - Statistics: RSS, chi2_reduced, AIC, AICc, BIC.
  - Basic diagnostics: convergence status, error messages, suggestions.
- **CLI**:
  - `fit <file> --model <key>` — outputs FitOutcome JSON.
  - `list-models` — outputs available model keys, labels, parameter names.
- **Tauri GUI**:
  - `fit_spectrum(path, model_key)` command.
  - "Circuit Fitting" panel with model dropdown, Fit Selected button.
  - Displays: status, parameter table, statistics (chi2_reduced, RSS), preprocessing info.

### Optimizer Choice
Custom bounded Levenberg-Marquardt implementation. No external optimization crate added.
- Rationale: keeps dependencies small, full control over bounds/clamping, sufficient for 7-10 parameter EIS models.
- Limitation: no trust-region scaling, no analytical Jacobian (finite-diff only). Acceptable for current scope.

### Parity With Python Reference
- Objective function: modulus weighting matches Python `calc-modulus` scheme.
- Bounds: match Python ZView direct bounds (resistances 1e-9..1e6/1e8, CPE n 0.2..1.0).
- Initial guess: segmentation-based approach matches Python ZView direct guess logic.
- Statistics: RSS, AIC, AICc, BIC formulas match Python `_zview_statistics`.
- Preprocessing: non-finite + inductive removal matches Python `preprocess_for_fit`.
- Not ported: pyimpspec backend (broken in Python anyway), multi-weight strategy (only modulus), DRT-guided guess, segment-based initialization.

### Known Gaps vs Python
| Feature | Python | Rust |
|---|---|---|
| pyimpspec fitting path | exists but broken | not ported |
| Multiple weighting schemes | unit, proportional, modulus, data-special | modulus only |
| Two-stage seed selection | AICc + runs test + cond | best RSS |
| Jacobian condition / stderr | SVD-based | not computed |
| Confidence intervals | bootstrap | not implemented |
| DRT-guided initial guess | stub (returns None) | not implemented |
| Batch fitting | full | not started |

### Test Coverage
68 tests total (9 new fitting tests + 59 prior):
- `fit_single_arc_recovers_params`: synthetic Rs=5, Rct=50 → fit recovers within tolerance
- `fit_double_arc_converges`: synthetic 10-param model converges
- `fit_produces_valid_statistics`: RSS, chi2_reduced, AIC, BIC all present
- `fit_unknown_model_fails_gracefully`: returns "failed" status
- `fit_few_points_fails`: < n_params+2 points → clear error message
- `preprocess_removes_inductive`: inductive point flagged
- `perturb_generates_variants`: correct seed count
- `solve_linear_system_basic`: 2x2 system solved correctly
- `estimate_cpe_n_reasonable`: n in [0.4, 1.0]

### Commands Run
```
cargo fmt                                      -> clean
cargo test                                     -> 68/68 passed
cargo check -p eismaster-tauri                -> clean
cargo run -p eismaster-cli -- list-models      -> JSON output
cargo run -p eismaster-cli -- fit ... --model ... -> converges (chi2=0.0013)
npx tsc --noEmit                               -> clean
npx vite build                                 -> 159KB JS + 4.8KB CSS
```

---

## Phase 6: Fit Overlay Plotting + Batch Fitting

### What's Implemented
- **Fit overlay on Nyquist canvas** (`NyquistPlot` component):
  - Accepts optional `fitReal`/`fitImag` props.
  - Fit curve drawn as red line behind blue data points.
  - Axis range auto-scales to include both experimental and fit data.
  - Plot updates immediately when fit result changes.
- **Batch fitting core** (`fitting.rs`):
  - `BatchItemResult` struct — file, label, quality, fit, error, z_real_ohm, z_imag_ohm.
  - `BatchSummary` struct — model_key, items, n_total/ok/warn/failed counts.
  - `fit_batch(dir, model_key)` — parses all .txt/.csv in folder, runs quality + fit per file, collects results.
  - `batch_label_from_stem()` — same label extraction as DRT module (OCV, T5M, etc.).
- **Batch export workbook** (`write_batch_workbook`):
  - Three sheets: `raw_plot`, `rs_rct`, `fit_overlay`.
  - `raw_plot`: 2 columns per spectrum (z_real_ohm, z_imag_ohm_pos), with label + blank header row.
  - `rs_rct`: header (label, file, Rs, Rct, Rsei), one row per spectrum.
  - `fit_overlay`: 4 columns per spectrum (z_real_exp, z_imag_exp_pos, z_real_fit, z_imag_fit_pos), with label + 3 blank header row.
- **CLI** (`eismaster-cli`):
  - `fit-folder <dir> --model <key> [--output <json>] [--export-xlsx <path>]` — batch fit all spectra, optional JSON + XLSX export.
- **Tauri GUI**:
  - `fit_batch_folder` and `export_batch_workbook` Tauri commands.
  - "Batch Fitting" panel in right sidebar with folder input, model selector, Run Batch Fit button.
  - Result table showing label, status (ok/warn/failed), chi2_reduced per spectrum.
  - Summary counts: total, ok, warn, failed.

### Parity With Python Reference
- Batch flow matches Python `batch.py`: parse → quality → fit → collect.
- Label extraction matches Python `_export_label_from_stem`.
- Workbook sheets match Python `exporters.py` layouts:
  - `raw_plot`: 2 cols/spectrum with label + blank header (Python uses `z_real_ohm`, `z_imag_ohm_pos` names).
  - `rs_rct`: header row with label, file, Rs, Rct, Rsei (Rsei only present for double-arc model).
  - `fit_overlay`: 4 cols/spectrum (z_real_exp, -z_imag_exp, z_real_fit, -z_imag_fit).
- Error handling: parse failures recorded with error message, not aborting batch (matches Python).

### Not Yet Implemented (vs Python)
| Feature | Python support | Rust status |
|---|---|---|
| Hysteresis check | `hysteresis_flag`, `hysteresis_ratio` | Not implemented |
| Progress callback | `on_progress` callback | Not implemented (MATLAB blocks) |
| Fit result in spectrum list | click to view fit | Not implemented |
| Export batch workbook from GUI | separate button | Partially (via `export_batch_workbook` command, no GUI button yet) |
| Confidence intervals | bootstrap | Not implemented |

### Test Coverage
72 tests total (4 new batch tests + 68 prior):
- `batch_label_from_stem_extracts_tokens`: OCV, T5M, T100M, fallback
- `batch_workbook_has_three_sheets`: xlsx structure with fit data
- `fit_batch_on_temp_dir`: parses plain numeric file, fits, collects results
- `fit_batch_empty_dir_fails`: empty dir → clear error

### Commands Run
```
cargo fmt                                      -> clean
cargo test                                     -> 72/72 passed
cargo check -p eismaster-tauri                -> clean
cargo check -p eismaster-cli                  -> clean
npx tsc --noEmit                               -> clean
npx vite build                                 -> 162KB JS + 4.9KB CSS
```

---

## Phase 7: Python Parity Hardening

### What's Implemented
- **rs_rct file column fix**: `BatchItemResult.file` now stores file stem without extension (matches Python `file_path.stem`).
- **Workbook header parity**: raw_plot headers changed to `z_real`/`imag_pos`; fit_overlay headers changed to `imag_exp_pos`/`imag_fit_pos` (matching Python `exporters.py`).
- **FitOutcome JSON parity**: Added `confidence_intervals` (empty HashMap) and `correlation_matrix_max` (NaN) placeholder fields. Typescript types updated.
- **GUI: batch row selection**: Clicking a batch result row highlights it and shows a `BatchFitPlot` Nyquist preview of the fit curve.
- **GUI: batch export button**: Added "Export Batch Workbook" button with XLSX path input in batch panel.

### Regression Tests (3 new)
- `fit_batch_item_file_uses_stem_no_extension`: verifies item.file = "Ag_EIS_OCV" not "Ag_EIS_OCV.txt"
- `fit_outcome_has_parity_placeholder_fields`: verifies confidence_intervals is empty, correlation_matrix_max is NaN
- `batch_workbook_headers_match_python`: verifies workbook creation succeeds with new headers

### Commands Run
```
cargo fmt                                      -> clean
cargo test                                     -> 75/75 passed
cargo check -p eismaster-tauri                -> clean
npx tsc --noEmit                               -> clean
npx vite build                                 -> 164KB JS + 4.9KB CSS
```

---

## Phase 8: Packaging & Size Evaluation

### What's Implemented
- **Icon generation**: Generated 32x32.png, 128x128.png, 128x128@2x.png from icon.ico using PIL.
- **Bundle config**: `tauri.conf.json` updated — targets set to `["nsis"]`, removed icon.icns, added `bundle.resources` for matlab_bridge and matlab-DRTtools-local.
- **Resource resolution**: `get_dev_resource_paths` updated to check Tauri resource directory first (packaged mode), fall back to dev paths.
- **NSIS installer**: `EISMaster_0.1.0_x64-setup.exe` built successfully (3.3 MB).

### Size Comparison

| Metric | Python (PySide6) | Rust/Tauri | Reduction |
|--------|-----------------|------------|-----------|
| Installer | ~300-400 MB | **3.3 MB** | **~99%** |
| Executable | N/A | 11 MB | — |

### Key Findings
- The Rust/Tauri rewrite achieves **99% size reduction** vs the Python/PySide6 package.
- Main Python bloat sources eliminated: PySide6/Qt DLLs (~200MB), Python runtime (~50MB), pandas/numpy/scipy (~80MB).
- WebView2 is system-provided (Windows 10+), adding 0 MB to the package.
- MATLAB DRT resources (1.4 MB) are bundled but MATLAB itself (~2 GB) remains a separate install.

### Commands Run
```
npx tauri build                                -> NSIS installer at 3.3 MB
cargo test                                     -> 75/75 passed
cargo check -p eismaster-tauri                -> clean
npx tsc --noEmit                               -> clean
npx vite build                                 -> 164KB JS + 4.9KB CSS
```
