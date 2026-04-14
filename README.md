# EISMaster

Electrochemical Impedance Spectroscopy desktop analysis tool — spectrum parsing, quality assessment, equivalent-circuit fitting, batch processing, and DRT integration.

## Features

- **Spectrum Parser** — Import CHI660F `.bin`, CH Instruments `.txt`, and generic `.csv` files. Auto-detect format and validate data integrity.
- **Quality Assessment** — KK compliance, noise floor estimation, and point-level anomaly detection.
- **Equivalent-Circuit Fitting** — Two built-in models:
  - **Single-arc** `R(Q(RWo))` — for simple electrode/electrolyte interfaces
  - **Double-arc** `R(QR)(Q(RWo))` — for SEI + charge-transfer dual-semicircle spectra
- **Async Fitting Engine** — Background QThread worker prevents UI freezing during computation. Two-stage solver reduces `least_squares` calls by ~63%.
- **Batch Processing** — Auto-detect arc segments, fit an entire folder of operando spectra, and track parameter trends across cycles.
- **DRT Integration** — Stage inputs for MATLAB DRTtools, parse `f_alpha` / `gamma` results, and overlay DRT peaks on impedance plots.
- **Interactive Plots** — Nyquist, Bode magnitude, Bode phase, and residual charts powered by `pyqtgraph`. Region selection via range sliders for arc segmentation.

## Installation

```bash
# Clone the repository
git clone https://github.com/Jakiewbe/EISMaster.git
cd EISMaster

# Create a virtual environment (Python >= 3.11 required)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install with advanced features (pyimpspec for initial guesses)
pip install -e .[advanced]

# Or minimal install (no pyimpspec, uses heuristic guesses)
pip install -e .
```

## Usage

```bash
python -m eismaster
```

### Workflow

1. **Import** — Drag or browse to load `.bin` / `.txt` / `.csv` EIS files. Multiple files appear in the queue sidebar.
2. **Inspect** — Switch to the spectrum tab to view Nyquist/Bode plots, quality report, and raw data table.
3. **Fit** — Select a circuit template (single-arc or double-arc), optionally mask outlier points on the chart, then click "开始拟合" (Start Fitting). Results appear in real-time with parameter values, R², and residual plots.
4. **Batch Fit** — Load a folder of operando spectra, click "批量拟合" to auto-fit all files. Results are exported as a summary table.
5. **DRT Analysis** — Select spectra, configure MATLAB DRTtools settings, and run distributed relaxation time analysis.
6. **Export** — Export fitted parameters, raw data tables, batch summaries, and publication-ready plot overlays as TXT or XLSX.

## Supported Inputs

| Format | Extension | Source |
|--------|-----------|--------|
| CHI660F binary | `.bin` | `A.C. Impedance` experiments |
| CH Instruments text | `.txt` | Exported from CHI software |
| Generic delimited | `.csv` | Any `frequency, Zreal, Zimag` format |

## Equivalent Circuit Models

### Single-arc: `R(Q(RWo))` — Randles circuit

```
      +--- CPE ------+
Rs ---+              +---
      +--- Rct - Zw -+
```

**Parameters**: `Rs`, `CPE_T`, `CPE_P`, `Rct`, `Wo_R`, `Wo_T`, `Wo_P`

### Double-arc: `R(QR)(Q(RWo))` — Dual-arc lithium-ion circuit

```
      +--- R_sei ----+   +--- R_ct ---- Zw ----+
Rs ---+              +---+                      +---
      +--- CPE_f ----+   +--- CPE_dl -----------+
    |  High-frequency  |   |  Mid/Low-frequency   |
    |  Surface film    |   |  Charge transfer     |
```

**Parameters**: `Rs`, `Q1`, `n1`, `Rsei`, `Q2`, `n2`, `Rct`, `Wo_R`, `Wo_T`, `Wo_P`

## Project Structure

```
src/eismaster/
  app.py                  # Application entry point
  models.py               # Data structures (SpectrumData, FitOutcome, etc.)
  matlab_drt.py           # MATLAB DRTtools integration
  exporters.py            # TXT/XLSX export functions
  io/
    chi.py                # CHI660F .bin parser
  analysis/
    fitting.py            # Core fitting engine (least_squares + pyimpspec)
    batch.py              # Batch auto-fit with arc segmentation
    circuits.py           # Circuit template definitions
    diagnostics.py        # Fit quality diagnostics
    native_drt.py         # Native DRT computation
    preprocessing.py      # Data preprocessing & cleaning
    quality.py            # KK compliance & quality assessment
    segmentation.py       # Automatic arc region detection
  ui/
    main_window.py        # FluentWindow GUI (PySide6 + qfluentwidgets)
    range_slider.py       # Custom range slider for arc selection
    theme.py              # Dark theme color palette
    circuit_builder/      # Interactive circuit diagram builder
tests/                    # Unit tests for all analysis modules
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Qt GUI framework |
| PySide6-Fluent-Widgets | Fluent Design UI components |
| pyqtgraph | Interactive Nyquist/Bode plotting |
| numpy / scipy | Numerical computing & `least_squares` optimizer |
| pandas | Data handling & export |
| pyimpspec *(optional)* | Initial parameter estimation via impedance spectroscopy |

## Screenshots

> TODO: Add screenshots of the main window, fitting overlay, and batch results.

## License

MIT
