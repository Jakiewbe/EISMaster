# Claude Code Task Breakdown

Use these as small implementation prompts. Each task should produce tests before moving on.

## Task 1: Create Rust Workspace

Create a Rust workspace with:

- `crates/eismaster-core`
- `crates/eismaster-cli`
- `apps/eismaster-tauri` placeholder

No GUI implementation yet.

## Task 2: Implement CHI Parser

Implement CHI TXT parsing in Rust.

Acceptance:

- Load representative `.txt` files.
- Output JSON with frequency, real impedance, imaginary impedance.
- Match Python parser numeric arrays.

## Task 3: Implement DRT Parser

Implement parser for MATLAB DRT output files.

Acceptance:

- Supports `tau/gamma`, `freq/gamma`, and peak-fit exports.
- Can produce sorted `logtau`, `tau_s`, `gamma_tau`, and `area_dln_tau`.

## Task 4: Implement DRT Workbook Export

Implement `drt_matrix.xlsx`.

Acceptance:

- Same three sheets as Python reference.
- Supports `logtau/gamma` and `tau/s/gamma`.
- Supports custom logtau integration breaks.

## Task 5: Implement MATLAB Command Builder

Implement MATLAB `-batch` command generation and staging directories.

Acceptance:

- Command string matches Python semantics.
- Works with paths containing spaces and Chinese characters.

## Task 6: Implement Quality Checks

Port current EIS quality checks.

Acceptance:

- Same issue labels and severity for fixtures.

## Task 7: Implement Circuit Model Evaluation

Implement impedance calculation for all current circuit templates.

Acceptance:

- Same predicted impedance for known parameters.

## Task 8: Implement Fitting

Implement least-squares fitting.

Acceptance:

- Same parameter names.
- Results close to Python reference.

## Task 9: Build Tauri UI Skeleton

Implement import and plotting first.

Acceptance:

- Can load spectra and display Nyquist/Bode plots.

