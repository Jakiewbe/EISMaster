# Phase 6 Audit Report

## Commands Run

| Command | Result |
|---------|--------|
| `cargo fmt --check` | PASS (clean) |
| `cargo test` | PASS (72/72) |
| `cargo check -p eismaster-tauri` | PASS |
| `npx tsc --noEmit` | PASS (clean) |
| `npx vite build` | PASS (162KB JS + 4.9KB CSS) |
| `cargo run -p eismaster-cli -- list-models` | PASS (2 models listed) |
| `cargo run -p eismaster-cli -- fit <file> --model zview_segmented_rq_rwo` | PASS (chi2=1.29e-3) |
| `cargo run -p eismaster-cli -- fit-folder <dir> --model ... --output ... --export-xlsx ...` | PASS (2/2 ok, JSON+XLSX written) |

## Files Verified

| File | Claimed Implementation | Actually Present |
|------|----------------------|-----------------|
| `crates/eismaster-core/src/fitting.rs` | BatchItemResult, BatchSummary, fit_batch, write_batch_workbook, batch tests | YES |
| `crates/eismaster-cli/src/main.rs` | fit-folder command with --output/--export-xlsx flags | YES |
| `apps/eismaster-tauri/src-tauri/src/commands.rs` | fit_batch_folder, export_batch_workbook commands | YES |
| `apps/eismaster-tauri/src-tauri/src/lib.rs` | Commands registered in handler | YES |
| `apps/eismaster-tauri/src/types.ts` | BatchItemResult, BatchSummary interfaces | YES |
| `apps/eismaster-tauri/src/App.tsx` | NyquistPlot overlay, Batch Fitting panel, batch state/handler | YES |
| `NOTES.md` | Phase 6 section with implementation details | YES |

## Batch Fitting Behavior

- **One item per input file**: YES. `fit_batch` iterates sorted paths, pushes one BatchItemResult per file.
- **Per-file failure isolation**: YES. Parse failures are caught with `match`, recorded with `error: Some(...)`, and `continue` skips to next file. Counted in `n_failed`.
- **Input order preserved**: YES. Paths are sorted before iteration; items are pushed in sorted order.
- **Label extraction**: FIXED. Original `batch_label_from_stem` was a simplified reimplementation of `drt::label_from_stem`. Replaced with a delegation call. Test updated to match `drt::label_from_stem` behavior (e.g. "random_file" → "file", not "random_file").
- **Predicted arrays length**: CORRECT. `evaluate_model` uses all `spectrum.freq_hz` points; `pred_real.len() == spectrum.z_real_ohm.len()`.

## Workbook Export

- **raw_plot sheet**: EXISTS. 2 columns per spectrum (z_real_ohm, z_imag_ohm_pos), label + blank header row, column headers "z_real_ohm" / "z_imag_ohm_pos". Writes all experimental points.
- **rs_rct sheet**: EXISTS. Header row: label, file, Rs, Rct, Rsei. One row per spectrum.
- **fit_overlay sheet**: EXISTS. 4 columns per spectrum (z_real_exp, z_imag_exp_pos, z_real_fit, z_imag_fit_pos). Label + 3-blank header row, then column header row, then data.
- **Layout vs Python**: Minor difference — rs_rct "file" column uses full filename with extension (Rust: `file_name()`), Python uses stem without extension (`file_path.stem`). Cosmetic only.

## GUI

- **Nyquist overlay**: CORRECT. Fit curve drawn as red line (`#ef4444`, 1.5px) behind blue data points. Axis range includes both experimental and fit data via `allX`/`allY` arrays.
- **Batch panel placement**: CORRECT. Located between Circuit Fitting and Manual DRT Export sections. Does not overlap or break existing panels.
- **TypeScript types**: MATCH. `BatchSummary` and `BatchItemResult` interfaces match Rust serde output. `z_real_ohm`/`z_imag_ohm` correctly excluded (Rust `#[serde(skip)]`).

## Bugs Found and Fixed

1. **Duplicate label extraction logic** (FIXED)
   - `batch_label_from_stem` reimplemented a subset of `drt::label_from_stem`.
   - Would produce wrong results for stems with `-` separators, or preferred tokens like `rest`, `charge`, `discharge`, `before`, `after`, `init`, `mid`, `end`.
   - Fix: replaced body with `crate::drt::label_from_stem(stem)`.
   - Test updated: "random_file" → "file" (correct per `label_from_stem` which returns last informative token).

## Remaining Gaps vs Python Reference

| Feature | Python | Rust |
|---------|--------|------|
| Hysteresis check | `hysteresis_flag`, `hysteresis_ratio` | Not implemented |
| Progress callback | `on_progress` callback during batch | Not implemented |
| rs_rct file column | Uses filename stem (no ext) | Uses filename with ext |
| Confidence intervals | bootstrap | Not implemented |
| Batch export from GUI | button to export xlsx | `export_batch_workbook` command exists but no GUI button wired up |
| DRT + fit_report sheets | In export bundle | Not in batch workbook |
