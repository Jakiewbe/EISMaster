# Rewrite Phases

## Phase 0: Freeze Reference

Deliverables:

- Keep current Python source intact.
- Keep a known-good packaged build.
- Save representative input and output fixtures.

Validation:

- Existing Python app can still parse, fit, run MATLAB DRT, and export DRT workbook.

## Phase 1: Rust Core Data Model and Parser

Deliverables:

- Rust crate with `SpectrumMetadata` and `SpectrumData`.
- CHI TXT parser.
- CLI command: `eismaster-core parse <file> --json`.

Validation:

- Compare Rust JSON output with Python parser output for representative CHI files.

## Phase 2: Export Reproduction

Deliverables:

- Rust xlsx/csv export module.
- Reproduce raw plot, fit overlay, Rs/Rct, and DRT workbook layouts.

Validation:

- Same sheet names and headers.
- Numeric values equal within tolerance.
- DRT workbook has `drt_line_plot`, `drt_cloud_density`, `drt_quant_area`.

## Phase 3: MATLAB DRT Integration

Deliverables:

- Rust staging of MATLAB input TXT files.
- Rust MATLAB command builder.
- Packaged DRTtools resource handling.

Validation:

- Command matches Python semantics.
- MATLAB outputs are parsed into the same DRT workbook.

## Phase 4: Fitting and Quality

Deliverables:

- Circuit model representation.
- Least-squares fitting.
- Batch fitting runner.
- Quality checks.

Validation:

- Same model keys and parameter names.
- Numeric results close to Python reference on fixtures.

## Phase 5: Tauri UI

Deliverables:

- Desktop shell.
- Import, plot, fitting, batch, MATLAB DRT, export workflows.

Validation:

- User workflow parity with Python app.
- Packaged size measured against Python release.

## Phase 6: Replacement Candidate

Deliverables:

- Windows installer or portable zip.
- Regression report against Python reference.

Validation:

- All reference fixtures pass.
- Manual smoke test passes.

