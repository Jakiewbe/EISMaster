# GUI Smoke Test Checklist

Run after every Tauri build (dev or release). Mark each step PASS/FAIL.

## Prerequisites

- Build: `cd rewrite/apps/eismaster-tauri && npm run tauri dev` (or launch the installed .exe)
- Have 2–3 `.mpt` / `.z` / `.DTA` EIS files ready in a known folder

---

## 1. Launch & Layout

- [ ] App opens without crash or blank screen
- [ ] Three panels visible: left (Spectra list), center (viewer), right (tabs)
- [ ] Status bar shows "Ready" at bottom
- [ ] Window is resizable; panels resize correctly (no overflow)

## 2. Import Files (Native Dialog)

- [ ] Click **Import Files** → native OS file picker opens (not browser-style)
- [ ] Select multiple files → files appear in left panel with filenames and point counts
- [ ] First imported file is auto-selected in the list
- [ ] Center panel shows filename, technique, and data table

## 3. Import Folder (Native Dialog)

- [ ] Click **Import Folder** → native OS folder picker opens
- [ ] Select a folder containing EIS files → all recognized files are imported
- [ ] Status bar updates with import count

## 4. Spectrum List Selection

- [ ] Click different items in the left panel → selection highlight updates
- [ ] Center panel switches to show the selected spectrum's data
- [ ] Empty state: when no spectra loaded, shows "Select a spectrum to preview"

## 5. Nyquist Plot

- [ ] Toggle **Show Plot** checkbox in toolbar
- [ ] Nyquist plot renders in center panel (canvas, not blank)
- [ ] Toggle off → plot hides, data table remains
- [ ] Switch spectra with plot on → plot redraws for new data

## 6. Analyze Tab — Inspect

- [ ] Right panel defaults to "Analyze" tab
- [ ] Click **Inspect** with a spectrum selected
- [ ] Inspect results appear: quality verdict, issues list, template matches
- [ ] Inspect with no spectrum selected → button is disabled or shows message

## 7. Analyze Tab — Fit

- [ ] Select equivalent circuit (e.g., R-RQ-RQ)
- [ ] Set initial parameter values
- [ ] Click **Fit** → button shows "Fitting...", then results appear
- [ ] Fitted parameters displayed with values
- [ ] Nyquist plot updates with fit overlay (red dashed line)
- [ ] Click **Clear** → fit results and overlay removed

## 8. Batch Tab

- [ ] Switch to "Batch" tab
- [ ] Click folder picker → native dialog opens
- [ ] Select folder → path appears in input field
- [ ] Select circuit and click **Fit All**
- [ ] Progress updates in status bar
- [ ] Summary table appears with per-file results (chi², params)
- [ ] Export button saves workbook (.xlsx) via native save dialog

## 9. MATLAB DRT Tab

- [ ] Switch to "MATLAB DRT" tab
- [ ] Settings form visible: lambda range, solver, plot type, etc.
- [ ] Resource path auto-detected or shows "Not found"
- [ ] Run DRT → status shows progress (or error if MATLAB not available)

## 10. DRT Export Tab

- [ ] Switch to "DRT Export" tab
- [ ] Settings form visible: lambda range, output folder
- [ ] Output folder picker → native dialog
- [ ] Export triggers correctly

## 11. Tab Accessibility

- [ ] Tab bar buttons have visible focus indicator on keyboard Tab
- [ ] Active tab is visually distinct (blue underline)
- [ ] Screen reader (if available) announces tab name and selected state

## 12. Status Bar

- [ ] Status bar updates on every action (import, fit, export, errors)
- [ ] Error messages show in red or distinct style
- [ ] Status bar does not overflow or truncate on long messages

## 13. Edge Cases

- [ ] Import a non-EIS file → gracefully ignored or error shown
- [ ] Import empty folder → "No spectra found" message
- [ ] Fit with terrible initial values → still converges or shows clear error
- [ ] Rapidly click between spectra → no crash or stale data

## 14. Window Close

- [ ] Close button (X) exits cleanly
- [ ] No orphan processes left after exit

## 15. Packaged Installer (Release Only)

- [ ] Run NSIS installer → installs without error
- [ ] Launch installed app → same behavior as dev mode
- [ ] Uninstall via Windows Settings → cleanly removed

---

## Result Summary

| Step | Result | Notes |
|------|--------|-------|
| 1    |        |       |
| 2    |        |       |
| 3    |        |       |
| 4    |        |       |
| 5    |        |       |
| 6    |        |       |
| 7    |        |       |
| 8    |        |       |
| 9    |        |       |
| 10   |        |       |
| 11   |        |       |
| 12   |        |       |
| 13   |        |       |
| 14   |        |       |
| 15   |        |       |

**Overall:** PASS / FAIL — _date: YYYY-MM-DD_
