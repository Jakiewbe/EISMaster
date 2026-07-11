# Phase 8: Packaging & Size Evaluation Report

Last updated: 2026-05-28

## Commands Run

| Command | Result |
|---------|--------|
| `cargo fmt --check` | PASS |
| `cargo test` | PASS (75/75) |
| `cargo check -p eismaster-tauri` | PASS |
| `npx tsc --noEmit` | PASS (clean) |
| `npx vite build` | PASS (164KB JS + 4.9KB CSS) |
| `npx tauri build` | PASS (NSIS installer produced) |

**Note:** Build artifacts have been cleaned (`target/` removed). Sizes below are from the last successful build. Rebuild with `npx tauri build` to regenerate.

## Build Artifacts

| Artifact | Path | Size (last build) |
|----------|------|------|
| Executable | `target/release/eismaster-tauri.exe` | **~11 MB** |
| NSIS Installer | `target/release/bundle/nsis/EISMaster_0.1.0_x64-setup.exe` | **~3.4 MB** |

### Bundled Resources

| Resource | Source | Bundled Size |
|----------|--------|-------------|
| matlab_bridge | `matlab_bridge/eismaster_batch_drt.m` | ~12 KB |
| matlab-DRTtools-local | Full DRTtools repo (m, fig, docs, src) | ~1.4 MB |

## Size Comparison vs Python

| Metric | Python (PySide6) | Rust/Tauri | Reduction |
|--------|-----------------|------------|-----------|
| Installer | ~300-400 MB | **3.3 MB** | **~99%** |
| Installed | ~400-500 MB | ~15 MB (est.) | **~96%** |
| Executable only | N/A (bundled Python) | 11 MB | — |

### Python Breakdown (approximate, from EISMaster.spec)
- PySide6 + Qt DLLs: ~200 MB (even with exclusions)
- Python runtime + stdlib: ~50 MB
- pandas + numpy + scipy: ~80 MB
- pyqtgraph + qfluentwidgets: ~30 MB
- pyinstaller bootloader + misc: ~20 MB
- App code + resources: ~5 MB

### Rust/Tauri Breakdown
- WebView2 (system-installed): 0 MB (Windows 10+ includes it)
- React + Vite frontend: 164 KB
- Rust core + Tauri runtime: ~11 MB (compressed to ~3 MB in installer)
- MATLAB resources: ~1.4 MB

## Configuration Changes

### tauri.conf.json
- `bundle.targets`: Changed from `"all"` to `["nsis"]` (Windows NSIS installer only)
- `bundle.icon`: Removed `icon.icns` (macOS only), generated missing PNG sizes from `icon.ico`
- `bundle.resources`: Added matlab_bridge and matlab-DRTtools-local for packaged mode
- `get_dev_resource_paths`: Updated to check Tauri resource dir first (packaged mode), fall back to dev paths

### Plugin Changes (post-Phase 8)
- Added `tauri-plugin-dialog` (Rust + JS) for native file/folder pickers
- Added `dialog:default` and `dialog:allow-open` to capabilities
- Removed unused `@tauri-apps/plugin-fs` (confirmed zero references)
- Tab accessibility: `role="tablist"`, `role="tab"`, `aria-selected`, `aria-controls` on RightPanel tabs

### Icons Generated
- `icons/32x32.png` (132 bytes)
- `icons/128x128.png` (1.7 KB)
- `icons/128x128@2x.png` (4.2 KB)

## Smoke Test Status

### Automated
- Binary exists and is ~11 MB: VERIFIED
- NSIS installer produced (~3.4 MB): VERIFIED
- Build completed without errors: VERIFIED

### Manual (requires GUI)
See [GUI_SMOKE_TEST.md](GUI_SMOKE_TEST.md) for the full 15-step checklist with result tracking.

Quick summary of what to verify:
1. Launch & layout (3 panels, status bar)
2. Import files via native dialog
3. Import folder via native dialog
4. Spectrum list selection
5. Nyquist plot toggle & redraw
6. Analyze tab — Inspect
7. Analyze tab — Fit (with overlay)
8. Batch tab — fit all + export
9. MATLAB DRT tab — settings & run
10. DRT Export tab — export
11. Tab accessibility (keyboard focus, ARIA)
12. Status bar updates
13. Edge cases (bad files, bad params)
14. Clean exit
15. Packaged installer install/uninstall

## MATLAB DRT Resource Status

- Resources are bundled in the NSIS installer: **YES** (verified by reading `installer.nsi` — `matlab_bridge\eismaster_batch_drt.m` at line 683, full `matlab-DRTtools-local\` tree at lines 645–682)
- `get_dev_resource_paths` now checks `app.path().resource_dir()` first: YES
- In packaged mode, resources resolve from `<install_dir>/resources/`: YES
- MATLAB execution still requires a separate MATLAB installation: YES (not bundled, ~2 GB)

## Known Packaging Limitations

1. **No macOS bundle**: `icon.icns` removed. macOS build would need icns regeneration.
2. **No MSIX/WiX installer**: Only NSIS. Can add later with `bundle.targets: "all"`.
3. **WebView2 dependency**: Requires WebView2 Runtime. Windows 10 20H2+ includes it; older systems may need manual install.
4. **MATLAB not bundled**: The MATLAB runtime (~2 GB) is a separate install. The app only calls `matlab.exe` via process spawn.
5. **No code signing**: The installer is unsigned. Production release would need a code signing certificate.
6. **No auto-update**: Tauri supports it but not configured yet.
7. **Resource path test**: Resources confirmed present in `installer.nsi`. Full runtime verification pending manual smoke test (step 15).

## Next Recommended Optimization Steps

1. **Strip debug symbols**: The 11MB executable could shrink further with `strip = true` in Cargo.toml profile.
2. **UPX compression**: Already enabled in EXE spec but could try more aggressive settings.
3. **Feature-gate MATLAB**: Make MATLAB DRT an optional feature to reduce binary size when not needed.
4. **WebView2 offline installer**: Bundle WebView2 bootstrapper for systems without it.
5. **Code signing**: Required for Windows SmartScreen trust.
6. **CI/CD packaging**: Automate builds via GitHub Actions.
7. **Icon improvements**: Generate higher-quality icons from a proper SVG source.
