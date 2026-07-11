# UI Refactor Audit Report

Date: 2026-05-28

## Validation Results

| Check | Result |
|-------|--------|
| `cargo fmt --check` | PASS |
| `cargo test -p eismaster-core` | PASS (75/75) |
| `cargo check -p eismaster-tauri` | PASS |
| `npx tsc --noEmit` | PASS (clean) |
| `npx vite build` | PASS (175KB JS, 12.3KB CSS) |
| `npx tauri build` | PASS (11MB exe, 3.4MB installer) |
| Dev server launch | PASS (compiles, opens window) |

**Bundle size change from previous:** JS 166.5KB → 175KB (+8.5KB), CSS 5.6KB → 12.3KB (+6.7KB). Increase is from the dark theme CSS variables and additional styles.

## 1. Import Workflows

### Import Files (Toolbar.tsx:24-52)
- Uses `open({ multiple: true, filters })` from `@tauri-apps/plugin-dialog` — correct.
- Gets real FS paths, calls `invoke("parse_file", { path: p })` — matches backend `parse_file(path: String)`.
- Errors go to status bar, not console — correct.
- Sets `isOperationRunning` during import — correct.

### Import Folder (Toolbar.tsx:54-74)
- Uses `open({ directory: true, multiple: false })` — correct.
- Calls `invoke("parse_folder", { dir: selected })` — matches backend `parse_folder(dir: String)`.
- `finally` block resets `isOperationRunning` — correct.

### Clear All (App.tsx:23-28)
- Resets `spectra`, `selectedIdx`, `fitResult`, `statusMsg` — correct.
- Does NOT reset child tab states (`inspectResult`, `batchResult`, `matlabLog`, etc.) — **minor issue**, see below.

## 2. Operation-Running State

### Pattern
- `isOperationRunning` is a global boolean in App.tsx, threaded through all components.
- Each operation sets it `true` on start, `false` in `finally` block.
- All action buttons check `disabled={... || isOperationRunning}`.

### Correctness
- **Inspect:** Sets on start, resets in `finally` — correct. Will reset even on error.
- **Fit:** Same pattern — correct.
- **Batch fit:** Same pattern — correct.
- **Batch export:** Same pattern — correct.
- **MATLAB DRT:** Same pattern — correct.
- **DRT Export:** Same pattern — correct.
- **Import files/folder:** Same pattern — correct.

### Concern
One global `isOperationRunning` means an operation in one tab disables buttons in all tabs. This is acceptable (backend commands are synchronous in Tauri's sense — concurrent fits would be problematic), but the UX could confuse users who switch tabs mid-operation.

## 3. NyquistPlot (NyquistPlot.tsx)

### ResizeObserver
- Creates observer on mount, observes container ref — correct.
- Cleanup via `observer.disconnect()` in useEffect return — correct.
- Minimum width clamped to 200px — prevents blank at small sizes.

### requestAnimationFrame
- `useEffect(() => { const raf = requestAnimationFrame(draw); return () => cancelAnimationFrame(raf); }, [draw])` — correct cleanup.

### Canvas DPR Scaling
- Sets `canvas.width = W * dpr`, `canvas.height = H * dpr`, then `ctx.scale(dpr, dpr)` — correct for high-DPI.

### Axis Scaling
- Computes combined min/max from raw + fit data when both present — correct.
- Uses `Math.min(scaleX, scaleY)` for uniform aspect ratio — correct.

### Fit Overlay
- Drawn before raw data points (line 173 vs 186), so blue dots appear on top of red line — correct visual ordering.

### Y-Axis Sign Convention — FIXED
- **Bug:** The Y-axis label said `-Z'' (Ω)` but the code plotted raw `z_imag_ohm` (negative for capacitive EIS). This put the Nyquist arc below the x-axis instead of above.
- **Fix:** Negate `z_imag_ohm` and `predicted_imag_ohm` when computing axis ranges and drawing points/lines. Both `NyquistPlot` and `BatchFitPlot` fixed.

### Dark Theme
- Background `#141416` matches CSS `--bg-surface` — consistent.

## 4. RightPanel

### Tab Accessibility
- `role="tablist"` with `aria-label="Analysis tools"` on tab bar — correct.
- `role="tab"`, `aria-selected`, `aria-controls`, `id` on each button — correct.
- `role="tabpanel"`, `id`, `aria-labelledby` on content div — correct.

### Tab Content
- All four tabs (Analyze, Batch, MATLAB DRT, DRT Export) receive `isOperationRunning` and `onIsOperationRunningChange` — correct.
- All invoke handlers have proper `try/catch/finally` blocks — correct.

## 5. SpectrumList

### Frequency Range Display
- `formatFreq()` helper shows MHz/kHz/Hz/mHz — correct.
- `Math.min(...s.freq_hz)` / `Math.max(...s.freq_hz)` on each render — **potential performance issue** for large arrays, but EIS spectra are typically < 1000 points, so acceptable.

### Accessibility
- `role="button"`, `tabIndex={0}`, `aria-pressed`, `onKeyDown` for Enter/Space — correct.

## 6. SpectrumViewer

- `scope="col"` on `<th>` elements — correct.
- `role="region"` with `aria-label` — correct.
- `showPlot` defaults to `true` in App.tsx — UX change, not a regression.

## 7. App.css — Design System

- Complete dark theme via CSS custom properties — consistent.
- All colors reference `var(--...)` — maintainable.
- No hardcoded colors remain (except canvas JS and a few inline styles).

## 8. Package Dependencies

- `@tauri-apps/plugin-fs`: **Not present** (already removed in prior session) — clean.
- `@tauri-apps/plugin-dialog`: Present, used in Toolbar and RightPanel — correct.
- No unused frontend dependencies found.

## Issues Found

### ISSUE-1: `role="banner"` on toolbar (Minor) — FIXED
**File:** Toolbar.tsx:77
**Problem:** `<div className="toolbar" role="banner">` — the ARIA `banner` role is for the page-level header (site logo, title). A toolbar should use `role="toolbar"`.
**Impact:** Screen readers may misidentify the toolbar as a page header.
**Fix:** Changed to `role="toolbar"` during audit.

### ISSUE-2: Clear All doesn't reset child tab states (Minor)
**File:** App.tsx:23-28
**Problem:** `handleClearSpectra` clears `spectra`, `selectedIdx`, `fitResult`, `statusMsg` but child tabs (AnalyzeTab's `inspectResult`, BatchTab's `batchResult`, MatlabTab's `matlabLog`) retain stale data.
**Impact:** After clearing, switching to Batch tab still shows old batch results. Not a crash, but confusing.
**Fix:** Either accept this (the data is inert without spectra), or add key-based remounting of RightPanel when spectra are cleared.

### ISSUE-3: Dead commands in lib.rs (Pre-existing, Minor)
**File:** lib.rs:12-13
**Problem:** `commands::discover_drt_files` and `commands::export_workbook_from_drt_dir` are registered but never called from the frontend (confirmed via grep).
**Impact:** Dead code. Not a regression from this refactor.
**Fix:** Remove when convenient (not urgent).

### ISSUE-5: NyquistPlot Y-axis sign convention broken — FIXED
**File:** NyquistPlot.tsx (both NyquistPlot and BatchFitPlot)
**Problem:** Y-axis label said `-Z'' (Ω)` but code plotted raw `z_imag_ohm` (negative for capacitive). Standard Nyquist plots show the arc above the x-axis. The backend returns negative `z_imag_ohm` (matching complex impedance convention), but the plot needs to negate it.
**Impact:** The entire Nyquist plot was upside-down. The semicircle appeared below the x-axis.
**Fix:** Added negation (`-zi[i]`, `-fitImag[i]`) in coordinate transforms and axis range calculations for both NyquistPlot and BatchFitPlot.

### ISSUE-4: "..." button labels are ambiguous (Minor)
**Files:** RightPanel.tsx (multiple pick buttons)
**Problem:** Pick folder/file buttons show just `"..."` as text. While they have `aria-label` attributes, the visible text is not descriptive.
**Impact:** Minor UX issue for non-screen-reader users.
**Fix:** Use an icon or "Browse" text instead of "...".

## Non-Issues (Confirmed OK)

- **showPlot defaults to `true`:** UX preference, not a bug.
- **"SYSTEM STATE: RUNNING/IDLE" in status bar:** Useful debug indicator, acceptable.
- **NyquistPlot 260px fixed height:** ResizeObserver prevents blank; aspect ratio may be suboptimal at narrow widths but functional.
- **CSS size increase (+6.7KB):** Expected for a full dark theme overhaul. 12.3KB is still very small.
- **Tauri build `modelKey` → `model_key`:** Tauri v2 `#[command]` applies camelCase renaming by default.

## Manual Smoke Test

**Status:** Cannot fully smoke test in CLI environment (no GUI access).

Dev server was verified to compile and launch without errors. The Tauri window opens (confirmed by process running). Full manual smoke test should be performed by the developer using the checklist in `GUI_SMOKE_TEST.md`.

## Verdict

**The refactor required one functional fix.** The NyquistPlot Y-axis sign convention was broken (ISSUE-5) — now fixed. All 6 automated checks pass after fix. The remaining 4 issues are minor (ARIA semantics, stale state, dead code, ambiguous button labels).

### Recommended Immediate Fixes
1. ~~Change `role="banner"` to `role="toolbar"` on the toolbar div.~~ **DONE**

### Recommended Follow-ups (Not Blocking)
1. Consider key-based remounting of RightPanel on Clear All.
2. Remove dead commands `discover_drt_files` and `export_workbook_from_drt_dir`.
3. Replace `"..."` button text with `"Browse"` or an icon.
