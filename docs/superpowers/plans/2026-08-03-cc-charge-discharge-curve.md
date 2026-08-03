# CC Charge-Discharge Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent CC curve page that reads CHI Chronopotentiometry `.bin` files, excludes EIS/static data, merges charge/discharge segments into continuous time-voltage curves, and exports the curve plus segment summary.

**Architecture:** Keep the CC path separate from the existing EIS path. `io/cc.py` decodes and validates one file or a folder; `cc_curves.py` groups, orders, and joins segments; `cc_exporters.py` builds TXT/XLSX outputs; `ui/cc_page.py` owns the independent UI. `MainWindow` only registers the page and preserves existing EIS state and workflows.

**Tech Stack:** Python 3.11, NumPy, pandas, openpyxl, PySide6, PySide6-Fluent-Widgets, pyqtgraph, pytest.

## Global Constraints

- Only `.bin` files whose header identifies `CP`/`Chronopotentiometry` are CC inputs.
- EIS files identified by `IMP` and all non-CC files are skipped.
- Static/EIS data is never inserted into a CC curve.
- Current direction and voltage endpoint direction must agree; contradictory data raises a clear error.
- Existing EIS import, fitting, DRT, and export behavior must remain unchanged.
- Do not modify or stage unrelated existing worktree changes.

---

### Task 1: Add the CC binary parser

**Files:**
- Create: `src/eismaster/io/cc.py`
- Modify: `src/eismaster/io/__init__.py`
- Create: `tests/cc_fixtures.py`
- Create: `tests/test_cc_parser.py`

**Interfaces:**
- Produces `CCSegment`, `parse_cc_bin(path)`, `load_cc_segments_from_folder(folder)`.
- `CCSegment` fields are `source_path`, `sample_id`, `target_voltage_v`, `mode`, `sample_interval_s`, `voltage_v`, and `current_sign_raw`.

- [ ] **Step 1: Write failing parser tests**

Create `tests/cc_fixtures.py` with deterministic binary builders used by the parser and UI tests:

```python
import struct
from pathlib import Path

def write_cc_bin(path: Path, voltages: list[float], current_raw: int) -> Path:
    raw = bytearray(0x696 + len(voltages) * 4)
    raw[8:10] = b"CP"
    raw[14:34] = b"Chronopotentiometry"
    struct.pack_into("<H", raw, 0x262, len(voltages))
    struct.pack_into("<f", raw, 0x492, 0.01)
    struct.pack_into("<h", raw, 0x4C4, current_raw)
    for index, voltage in enumerate(voltages):
        struct.pack_into("<f", raw, 0x696 + index * 4, voltage)
    path.write_bytes(raw)
    return path

def write_eis_marker(path: Path) -> Path:
    raw = bytearray(0x696)
    raw[8:11] = b"IMP"
    path.write_bytes(raw)
    return path
```

```python
def test_parse_cc_bin_decodes_header_interval_direction_and_voltage(tmp_path):
    path = write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 3.2, 4.0], current_raw=-18128)
    segment = parse_cc_bin(path)
    assert segment.sample_id == "S01"
    assert segment.target_voltage_v == pytest.approx(4.0)
    assert segment.mode == "charge"
    assert segment.sample_interval_s == pytest.approx(0.01)
    assert segment.voltage_v.tolist() == pytest.approx([3.0, 3.2, 4.0])

def test_folder_loader_skips_eis_and_unknown_bin(tmp_path):
    write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 4.0], current_raw=-18128)
    write_eis_marker(tmp_path / "L_S01_EIS_4p00V.bin")
    (tmp_path / "notes.bin").write_bytes(b"unknown")
    segments = load_cc_segments_from_folder(tmp_path)
    assert [item.source_path.name for item in segments] == ["L_S01_CC_4p00V.bin"]
```

The fixture helper must write `CP` at byte offset `0x08`, `1` point count at `0x262`, `0.01` at `0x492`, signed current at `0x4C4`, and little-endian float points beginning at `len(raw) - count * 4`; use the actual filename pattern for sample and target parsing.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_cc_parser.py -q`

Expected: collection/import failure because `eismaster.io.cc` and its interfaces do not exist.

- [ ] **Step 3: Implement the minimal parser**

Implement `parse_cc_bin(path)` with these validations:

```python
if raw[8:10] != b"CP" or b"Chronopotentiometry" not in raw[:96]:
    raise ValueError(f"Not a CHI Chronopotentiometry file: {path.name}")
count = struct.unpack_from("<H", raw, 0x262)[0]
interval_s = struct.unpack_from("<f", raw, 0x492)[0]
current_sign_raw = struct.unpack_from("<h", raw, 0x4C4)[0]
data_start = len(raw) - count * 4
voltage = np.frombuffer(raw, dtype="<f4", count=count, offset=data_start).copy()
```

Reject zero/invalid counts, non-positive intervals, non-finite voltages, zero current direction, malformed `*_CC_*V.bin` names, and a voltage endpoint that does not rise for charge or fall for discharge. Map negative current to `charge` and positive current to `discharge`, matching the supplied L files.

`load_cc_segments_from_folder` must inspect the header before parsing, skip EIS/unknown files, collect CC parse errors, and raise one error containing all failed CC filenames if any valid CC file cannot be parsed.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python -m pytest tests/test_cc_parser.py -q`

Expected: all parser tests pass.

- [ ] **Step 5: Export the public IO names**

Update `src/eismaster/io/__init__.py` so the CC page imports only the public CC interfaces; leave `chi.py` behavior unchanged.

- [ ] **Step 6: Commit the parser unit**

```powershell
git add src/eismaster/io/cc.py src/eismaster/io/__init__.py tests/test_cc_parser.py
git commit -m "feat: parse CHI chronopotentiometry files"
```

### Task 2: Group, order, and join CC segments

**Files:**
- Create: `src/eismaster/cc_curves.py`
- Create: `tests/test_cc_curves.py`

**Interfaces:**
- Consumes `CCSegment` from `eismaster.io.cc`.
- Produces `CCCurve` and `merge_cc_segments(segments) -> list[CCCurve]`.
- Each `CCCurve` represents one `(sample_id, mode)` pair and contains `sample_id`, `mode`, `segments`, `total_time_s`, `segment_time_s`, `voltage_v`, `segment_index`, and `source_file` arrays/lists.

- [ ] **Step 1: Write failing ordering and time tests**

```python
import numpy as np
from pathlib import Path
from eismaster.io.cc import CCSegment

def segment(sample_id, mode, target_voltage, voltages):
    return CCSegment(
        source_path=Path(f"{sample_id}_CC_{str(target_voltage).replace('.', 'p')}V.bin"),
        sample_id=sample_id,
        target_voltage_v=target_voltage,
        mode=mode,
        sample_interval_s=0.01,
        voltage_v=np.asarray(voltages, dtype=float),
        current_sign_raw=-1 if mode == "charge" else 1,
    )

def test_charge_segments_are_sorted_by_target_and_time_is_continuous():
    segments = [segment("S01", "charge", 4.2, [4.0, 4.2]), segment("S01", "charge", 4.0, [3.0, 4.0])]
    curve = merge_cc_segments(segments)[0]
    assert curve.source_file == ["S01_CC_4p00V.bin", "S01_CC_4p20V.bin"]
    assert curve.total_time_s.tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert curve.segment_index.tolist() == [1, 1, 2, 2]

def test_discharge_segments_are_sorted_by_target_descending_and_separated_from_charge():
    curves = merge_cc_segments([
        segment("S01", "charge", 4.0, [3.0, 4.0]),
        segment("S03", "discharge", 4.2, [4.6, 4.2]),
        segment("S03", "discharge", 2.0, [2.2, 2.0]),
    ])
    assert {(curve.sample_id, curve.mode) for curve in curves} == {("S01", "charge"), ("S03", "discharge")}
    discharge = next(curve for curve in curves if curve.mode == "discharge")
    assert discharge.source_file[0].endswith("4p20V.bin")
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_cc_curves.py -q`

Expected: import failure because `eismaster.cc_curves` does not exist.

- [ ] **Step 3: Implement minimal grouping and joining**

Group by `(sample_id, mode)`. Sort charge targets ascending and discharge targets descending. For each segment, use `np.arange(n, dtype=float) * sample_interval_s`; append it at the current offset and advance the offset by `n * sample_interval_s`. Do not insert EIS/static points and do not silently deduplicate boundary points.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python -m pytest tests/test_cc_curves.py -q`

Expected: all curve tests pass.

- [ ] **Step 5: Commit the curve unit**

```powershell
git add src/eismaster/cc_curves.py tests/test_cc_curves.py
git commit -m "feat: merge CC charge and discharge segments"
```

### Task 3: Add CC curve and segment exports

**Files:**
- Create: `src/eismaster/cc_exporters.py`
- Create: `tests/test_cc_exporters.py`

**Interfaces:**
- Consumes `list[CCCurve]`.
- Produces `build_cc_curve_table(curves)`, `build_cc_segment_summary(curves)`, and `export_cc_curves(target, curves, fmt) -> dict[str, Path]`.

- [ ] **Step 1: Write failing export tests**

```python
from pathlib import Path
from eismaster.cc_curves import merge_cc_segments
from tests.test_cc_curves import segment

def test_curve_export_contains_traceability_columns(tmp_path):
    curves = merge_cc_segments([segment("S01", "charge", 4.0, [3.0, 4.0])])
    table = build_cc_curve_table(curves)
    assert list(table.columns) == ["sample_id", "mode", "total_time_s", "segment_time_s", "voltage_v", "segment_index", "source_file"]
    assert table["source_file"].notna().all()

def test_xlsx_export_has_curve_and_segment_sheets(tmp_path):
    curves = merge_cc_segments([segment("S01", "charge", 4.0, [3.0, 4.0])])
    paths = export_cc_curves(tmp_path / "cc.xlsx", curves, fmt="xlsx")
    workbook = openpyxl.load_workbook(paths["curve"], read_only=True)
    assert workbook.sheetnames == ["cc_curve", "cc_segments"]
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_cc_exporters.py -q`

Expected: import failure because `eismaster.cc_exporters` does not exist.

- [ ] **Step 3: Implement TXT/XLSX output**

Use the exact curve columns above. The summary must contain `sample_id`, `mode`, `segment_index`, `source_file`, `target_voltage_v`, `start_voltage_v`, `end_voltage_v`, `point_count`, `duration_s`, and `order`. TXT writes two comma-separated files with `_curve` and `_segments` suffixes; XLSX writes one workbook with `cc_curve` and `cc_segments` sheets.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python -m pytest tests/test_cc_exporters.py -q`

Expected: all export tests pass.

- [ ] **Step 5: Commit the export unit**

```powershell
git add src/eismaster/cc_exporters.py tests/test_cc_exporters.py
git commit -m "feat: export merged CC curves"
```

### Task 4: Add the independent CC page and register it

**Files:**
- Create: `src/eismaster/ui/cc_page.py`
- Modify: `src/eismaster/ui/main_window.py`
- Create: `tests/test_cc_ui.py`

**Interfaces:**
- `CCCurvePage(QWidget)` exposes `load_folder(folder)`, `export_current(target, fmt)`, `curves`, and `status_label`.
- `MainWindow` registers the page as `cc_widget` with object name `ccInterface` and navigation label `CC曲线`.

- [ ] **Step 1: Write failing UI smoke tests**

```python
def test_main_window_registers_cc_curve_page():
    window = MainWindow()
    try:
        assert window.cc_widget.objectName() == "ccInterface"
        assert window.cc_widget.windowTitle() == "CC曲线"
    finally:
        window.close()

def test_cc_page_loads_folder_and_shows_curve(tmp_path):
    write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 4.0], current_raw=-18128)
    page = CCCurvePage()
    page.load_folder(tmp_path)
    assert len(page.curves) == 1
    assert page.status_label.text().startswith("已加载")
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_cc_ui.py -q`

Expected: import or attribute failure because the page and registration do not exist.

- [ ] **Step 3: Implement the independent page**

Build a folder selector, load button, export button, status label, pyqtgraph time-voltage plot, curve table, and segment summary table. `load_folder` calls `load_cc_segments_from_folder` then `merge_cc_segments`; errors are displayed with the failed filename. The page must never call `load_spectrum` or mutate `AppState.spectra`.

Register the page in `_build_ui` after the existing EIS pages and leave the existing import buttons and navigation handlers unchanged except for recognizing the new page.

- [ ] **Step 4: Run the focused UI tests and confirm GREEN**

Run: `python -m pytest tests/test_cc_ui.py -q`

Expected: all CC UI smoke tests pass in offscreen mode.

- [ ] **Step 5: Commit the UI unit**

```powershell
git add src/eismaster/ui/cc_page.py src/eismaster/ui/main_window.py tests/test_cc_ui.py
git commit -m "feat: add standalone CC curve page"
```

### Task 5: Run real-file integration checks and regression verification

**Files:**
- Modify: `tests/test_cc_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Add a real L-folder integration test**

Use `Path(__file__).parents[1] / "L"` when present. Assert that only ten `*_CC_*.bin` files are loaded, no `_EIS_` file appears in any curve row, S01 is charge, S03 is discharge, and every curve has strictly increasing `total_time_s`. Skip only when the user-provided L folder is absent so the normal test suite remains portable.

- [ ] **Step 2: Document the new page**

Add a short README section stating that the CC曲线 page scans CP/CC files, skips EIS/static data, merges by sample and direction, and exports curve plus segment summary.

- [ ] **Step 3: Run focused and full verification**

Run:

```powershell
python -m pytest tests/test_cc_parser.py tests/test_cc_curves.py tests/test_cc_exporters.py tests/test_cc_ui.py tests/test_cc_integration.py -q
python -m pytest -q
ruff check src tests
```

Expected: CC tests pass, full existing suite passes, and Ruff reports no errors.

- [ ] **Step 4: Perform the real UI smoke check**

Launch EISMaster, open `CC曲线`, load the root `L` folder, confirm the plot contains CC data only, confirm charge/discharge labels, export XLSX, and reopen the workbook to verify both sheets.

- [ ] **Step 5: Commit verification/documentation**

```powershell
git add tests/test_cc_integration.py README.md
git commit -m "test: verify CC curve extraction with L samples"
```
