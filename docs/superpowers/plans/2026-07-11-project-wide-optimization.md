# EISMaster Project-Wide Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有用户工作流和科学计算含义的前提下，把 EISMaster 整理成可复现、可测试、可维护、可稳定打包的 Python/PySide6 桌面应用，并用固定对照数据决定 Rust/Tauri 重写是否具备替换资格。

**Architecture:** Python/PySide6 继续作为当前正式实现和行为基准。优化按可独立验收的纵向切片推进：先恢复测试基线，再稳定导出契约，然后分离 UI 状态与后台任务，随后收敛拟合/批量逻辑，最后验证 MATLAB、打包和 Rust 对照结果。Rust/Tauri 在完成固定样例对照前不替换 Python 主线。

**Tech Stack:** Python 3.11、NumPy、pandas、SciPy、PySide6、PySide6-Fluent-Widgets、pyqtgraph、openpyxl、unittest/pytest、Ruff、PyInstaller；Rust/Cargo、React/TypeScript、Tauri 仅用于对照迁移。

## Global Constraints

- 不创建新分支或 worktree，除非用户明确同意。
- 保留现有未提交修改：`EISMaster.spec`、`src/eismaster/exporters.py`、`src/eismaster/ui/main_window.py`、`tests/test_exports.py`。
- Python 正式运行版本固定为 3.11；不得依赖机器上的默认 `python` 指向正确解释器。
- 不通过静默降级、吞异常或后处理掩盖失败；解析、拟合、MATLAB 和导出失败必须暴露明确原因。
- 不改变阻抗符号约定：模型数据使用 `z_imag_ohm`，Nyquist 展示和 `imag_pos` 导出使用 `-z_imag_ohm`。
- 不改变现有工作簿表名和主要字段，除非相应回归测试先明确新契约。
- 每项任务结束后执行 Review 查 Bug，再做第一性原理检查：是否仍有更简单且同样可靠的实现。
- 每项任务通过独立测试后才可进入下一项；提交动作需用户授权。

## Target File Structure

```text
src/eismaster/
├── analysis/
│   ├── fitting.py              # 对外 fit_spectrum 编排入口
│   ├── fitting_math.py         # 无状态数值辅助函数
│   ├── batch.py                # 批量流程编排
│   ├── preprocessing.py
│   ├── quality.py
│   └── segmentation.py
├── exporting/
│   ├── __init__.py             # 保持公开导出 API
│   ├── bundle.py               # 文件/工作簿写入
│   ├── matrices.py             # raw、fit、Rs/Rct 表格
│   └── drt.py                  # DRT 解析和三类矩阵
├── ui/
│   ├── state.py                # AppState 及状态失效规则
│   ├── workers.py              # Qt 后台任务对象
│   ├── presenters.py           # 纯格式化和表格模型
│   └── main_window.py          # 窗口组装和事件转发
└── matlab_drt.py               # MATLAB 配置、校验和进程调用

tests/
├── __init__.py                 # 确保测试辅助模块稳定导入
├── fixture_factory.py          # 确定性 TXT/BIN/双弧样例生成
├── test_state.py
├── test_presenters.py
├── test_matlab_drt.py
├── test_packaging.py
└── 原有测试文件
```

---

### Task 1: 冻结当前工作区并建立执行基线

**Files:**
- Inspect: `EISMaster.spec`
- Inspect: `src/eismaster/exporters.py`
- Inspect: `src/eismaster/ui/main_window.py`
- Inspect: `tests/test_exports.py`
- Create during execution only: `docs/optimization-baseline.md`

**Interfaces:**
- Consumes: 当前 `main` 工作区和未提交修改。
- Produces: 一份明确记录版本、脏文件、测试命令和已知阻断项的基线说明。

- [ ] **Step 1: 记录真实 Git 状态**

Run:

```powershell
git status --short --branch
git diff --numstat
git log -5 --oneline --decorate
```

Expected: 分支为 `main`；四个已修改文件保持原状；未跟踪的 `rewrite/`、`rust-rewrite-plan/`、`dist_rebuild4/` 不被误删或加入正式主线。

- [ ] **Step 2: 写入基线说明**

Create `docs/optimization-baseline.md` with this structure and the real command output values:

```markdown
# EISMaster Optimization Baseline

- Baseline commit: `d72afc93265e13c25f4ec4da8346a7f3d1ce160a`
- Python contract: `>=3.11`
- Formal implementation: `src/eismaster/`
- Comparison implementation: `rewrite/`
- Protected modified files:
  - `EISMaster.spec`
  - `src/eismaster/exporters.py`
  - `src/eismaster/ui/main_window.py`
  - `tests/test_exports.py`
- Known baseline blockers:
  - repository-root sample files are missing
  - default `python` is not Python 3.11
  - UI dependency is not installed in the Python 3.11 interpreter
```

- [ ] **Step 3: Review and first-principles gate**

Run:

```powershell
git diff -- docs/optimization-baseline.md
git status --short
```

Expected: 除基线文档外没有新增修改；受保护文件的 diff 与任务开始前一致。

---

### Task 2: 固定 Python 3.11 开发环境和测试入口

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `pyproject.toml`

**Interfaces:**
- Consumes: Python 3.11 和现有项目依赖。
- Produces: `pip install -e ".[dev]"` 和 `py -3.11 -m pytest` 两个唯一开发入口。

- [ ] **Step 1: 添加失败的配置检查**

Run:

```powershell
py -3.11 -c "import tomllib, pathlib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); assert 'dev' in p['project']['optional-dependencies']"
```

Expected: FAIL with `AssertionError`。

- [ ] **Step 2: 增加开发依赖和 pytest 配置**

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
advanced = [
  "pyimpspec>=5.1",
]
dev = [
  "pytest>=8.2,<9",
  "ruff>=0.5,<1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Keep the existing `advanced` dependency exactly once.

- [ ] **Step 3: 验证配置可读**

Run:

```powershell
py -3.11 -c "import tomllib, pathlib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); assert p['project']['requires-python'] == '>=3.11'; assert 'pytest>=8.2,<9' in p['project']['optional-dependencies']['dev']"
```

Expected: PASS with exit code 0。

- [ ] **Step 4: 统一 README 命令**

Replace generic test instructions with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,advanced]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

- [ ] **Step 5: Review and first-principles gate**

Run:

```powershell
git diff --check -- pyproject.toml README.md
py -3.11 -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))"
```

Expected: 2 commands pass, 0 failures。

---

### Task 3: 消除测试对仓库根目录私人样例的依赖

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/fixture_factory.py`
- Modify: `tests/test_parsers.py`
- Modify: `tests/test_exports.py`
- Modify: `tests/test_matlab_drt.py`
- Modify: `tests/test_batch_runtime.py`
- Modify: `tests/test_ui_smoke.py`

**Interfaces:**
- Consumes: `SpectrumData`、CHI TXT/BIN 解析格式。
- Produces: `write_single_arc_txt(path) -> Path`、`write_single_arc_bin(path) -> Path`、`make_double_arc_spectrum(path) -> SpectrumData`。

- [ ] **Step 1: 写入失败的工厂接口测试**

Add to `tests/test_parsers.py`:

```python
from tests.fixture_factory import write_single_arc_bin, write_single_arc_txt

    def test_generated_txt_and_bin_match(self) -> None:
        txt = write_single_arc_txt(self.tmp_root / "single.txt")
        binary = write_single_arc_bin(self.tmp_root / "single.bin")
        txt_spectrum = load_spectrum(txt)
        bin_spectrum = load_spectrum(binary)
        np.testing.assert_allclose(bin_spectrum.freq_hz, txt_spectrum.freq_hz, rtol=1e-4)
        np.testing.assert_allclose(bin_spectrum.z_real_ohm, txt_spectrum.z_real_ohm, rtol=1e-5)
        np.testing.assert_allclose(bin_spectrum.z_imag_ohm, txt_spectrum.z_imag_ohm, rtol=1e-5)
```

- [ ] **Step 2: 运行并确认缺少工厂**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_parsers.py::ParserTests::test_generated_txt_and_bin_match -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tests.fixture_factory'`。

- [ ] **Step 3: 实现确定性样例工厂**

Create an empty `tests/__init__.py`, then create `tests/fixture_factory.py`:

```python
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from eismaster.models import SpectrumData, SpectrumMetadata


_SINGLE_FREQ = np.geomspace(100000.0, 0.01, 85)
_SINGLE_OMEGA = 2.0 * np.pi * _SINGLE_FREQ
_SINGLE_Z = 5.0 + 42.0 / (1.0 + (1j * _SINGLE_OMEGA * 2e-3) ** 0.88)
SINGLE_ROWS = tuple(
    (float(freq), float(value.real), float(value.imag))
    for freq, value in zip(_SINGLE_FREQ, _SINGLE_Z)
)


def write_single_arc_txt(path: Path) -> Path:
    rows = ["Freq/Hz, Z'/ohm, Z''/ohm"]
    rows.extend(f"{freq:.12g},{real:.12g},{imag:.12g}" for freq, real, imag in SINGLE_ROWS)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_single_arc_bin(path: Path) -> Path:
    header = bytearray(0x280)
    header[:26] = b"IMP A.C. Impedance CHI660F"
    struct.pack_into("<H", header, 0x25E, len(SINGLE_ROWS))
    records = b"".join(struct.pack("<4f", freq, freq, real, imag) for freq, real, imag in SINGLE_ROWS)
    path.write_bytes(bytes(header) + records)
    return path


def make_double_arc_spectrum(path: Path) -> SpectrumData:
    freq = np.geomspace(100000.0, 0.1, 61)
    omega = 2.0 * np.pi * freq
    z = 4.0 + 18.0 / (1.0 + (1j * omega * 2e-4) ** 0.9)
    z += 55.0 / (1.0 + (1j * omega * 2e-2) ** 0.82)
    return SpectrumData(
        metadata=SpectrumMetadata(file_path=path, source_format="synthetic"),
        freq_hz=freq,
        z_real_ohm=z.real,
        z_imag_ohm=z.imag,
        z_mod_ohm=np.abs(z),
        phase_deg=np.degrees(np.angle(z)),
    )
```

- [ ] **Step 4: 把所有根目录样例引用替换为测试工厂**

Use each test's existing `setUp()` temporary directory. Replace `ROOT / "Ag_EIS_OCV.txt"` and `next(ROOT.glob("*放电完后.txt"))` with generated paths or in-memory spectra. Do not weaken numeric assertions; update them to the exact `SINGLE_ROWS` values.

- [ ] **Step 5: 运行样例相关测试**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_parsers.py tests/test_exports.py tests/test_matlab_drt.py tests/test_batch_runtime.py -q
```

Expected: 0 missing-file errors；若拟合数值断言失败，只调整生成数据或算法，不跳过测试。

- [ ] **Step 6: Review and first-principles gate**

Run:

```powershell
rg -n "Ag_EIS_OCV|放电完后|ROOT /.*\.txt|ROOT /.*\.bin" tests -g '*.py'
git diff --check -- tests
```

Expected: 第一个命令不再发现外部样例依赖；第二个命令通过。

---

### Task 4: 固化导出与 DRT 工作簿契约

**Files:**
- Modify: `src/eismaster/exporters.py`
- Modify: `tests/test_exports.py`
- Create: `tests/test_export_contract.py`

**Interfaces:**
- Consumes: `SpectrumData`、`FitOutcome`、MATLAB DRT 文本目录。
- Produces: `raw_plot`、`rs_rct`、`fit_overlay`、`drt_line_plot`、`drt_cloud_density`、`drt_quant_area`、`fit_report`。

- [ ] **Step 1: 为工作簿写入完整契约测试**

Create `tests/test_export_contract.py` exactly as follows:

```python
from pathlib import Path

import numpy as np
import openpyxl

from eismaster.exporters import write_drt_only_export
from eismaster.models import SpectrumData, SpectrumMetadata


def _spectrum(path: Path) -> SpectrumData:
    freq = np.array([1000.0, 100.0, 10.0])
    real = np.array([5.0, 6.0, 8.0])
    imag = np.array([-0.2, -0.7, -1.1])
    return SpectrumData(
        metadata=SpectrumMetadata(file_path=path),
        freq_hz=freq,
        z_real_ohm=real,
        z_imag_ohm=imag,
        z_mod_ohm=np.hypot(real, imag),
        phase_deg=np.degrees(np.arctan2(imag, real)),
    )


def test_drt_workbook_contract(tmp_path: Path) -> None:
    spectrum = _spectrum(tmp_path / "Ag_S01_EIS_T5M.txt")
    drt_result = tmp_path / "Ag_S01_EIS_T5M_DRT.txt"
    drt_result.write_text("tau\tgamma(tau)\n1\t1\n10\t2\n100\t3\n", encoding="utf-8")
    target = tmp_path / "drt.xlsx"
    paths = write_drt_only_export(target, [spectrum], tmp_path, fmt="xlsx")
    workbook = openpyxl.load_workbook(paths["drt"], data_only=True)
    assert workbook.sheetnames == ["drt_line_plot", "drt_cloud_density", "drt_quant_area"]
    assert workbook["drt_line_plot"]["A2"].value in {"logtau", "tau/s"}
    assert workbook["drt_cloud_density"]["A2"].value == "logtau"
    assert workbook["drt_quant_area"]["A1"].value == "sample"
    workbook.close()
```

- [ ] **Step 2: 先运行当前导出测试**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_exports.py tests/test_export_contract.py -q
```

Expected: 新契约测试在未满足的具体字段处失败，而不是因环境或样例缺失失败。

- [ ] **Step 3: 收敛 DRT 导出实现**

Keep exactly one internal representation:

```python
DrtMatrices = dict[str, pd.DataFrame]

DRT_SHEET_ORDER = (
    "drt_line_plot",
    "drt_cloud_density",
    "drt_quant_area",
)
```

Iterate using `DRT_SHEET_ORDER` in both workbook and delimited export paths so sheet/file order cannot drift. Reject unsupported `line_x_axis` values with `ValueError` instead of silently treating them as `logtau`.

- [ ] **Step 4: 运行导出回归**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_exports.py tests/test_export_contract.py -q
```

Expected: 全部通过，0 failures。

- [ ] **Step 5: Review and first-principles gate**

Check that parsing, matrix construction, and file writing each have one responsibility. If `exporters.py` still exceeds 500 lines after current DRT work stabilizes, execute the next extraction as a behavior-preserving move:

- `src/eismaster/exporting/drt.py`: DRT parsing and matrix construction.
- `src/eismaster/exporting/matrices.py`: raw/fit/Rs-Rct tables.
- `src/eismaster/exporting/bundle.py`: xlsx/csv/txt writing.
- `src/eismaster/exporters.py`: compatibility imports only.

Run the same export test command after each file move.

---

### Task 5: 从主窗口提取状态和后台任务

**Files:**
- Create: `src/eismaster/ui/state.py`
- Create: `src/eismaster/ui/workers.py`
- Modify: `src/eismaster/ui/main_window.py`
- Create: `tests/test_state.py`
- Create: `tests/test_workers.py`

**Interfaces:**
- Consumes: `SpectrumData`、`QualityReport`、`FitOutcome`、`BatchSummary`。
- Produces: `AppState.clear_all()`、`AppState.invalidate_batch_outputs()`、`SingleFitWorker`、`BatchFitWorker`、`MatlabDrtWorker`。

- [ ] **Step 1: 写入状态失效测试**

Create `tests/test_state.py`:

```python
import numpy as np

from eismaster.models import BatchSummary, SpectrumData, SpectrumMetadata
from eismaster.ui.state import AppState


def test_clear_all_removes_derived_state(tmp_path) -> None:
    spectrum = SpectrumData(
        metadata=SpectrumMetadata(file_path=tmp_path / "sample.txt"),
        freq_hz=np.array([1.0]),
        z_real_ohm=np.array([1.0]),
        z_imag_ohm=np.array([-1.0]),
        z_mod_ohm=np.array([2**0.5]),
        phase_deg=np.array([-45.0]),
    )
    state = AppState(spectra=[spectrum], current_index=0)
    state.point_masks[spectrum.display_name] = np.array([True])
    state.batch_summary = BatchSummary(model_key="demo", items=[])
    state.clear_all()
    assert state.spectra == []
    assert state.current_index == -1
    assert state.point_masks == {}
    assert state.batch_summary is None
```

- [ ] **Step 2: 运行并确认模块缺失**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_state.py -q
```

Expected: FAIL with `ModuleNotFoundError: eismaster.ui.state`。

- [ ] **Step 3: 实现状态对象**

Create `src/eismaster/ui/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eismaster.analysis.segmentation import SegmentDetection
from eismaster.models import BatchSummary, FitOutcome, QualityReport, SpectrumData


@dataclass
class AppState:
    spectra: list[SpectrumData] = field(default_factory=list)
    qualities: dict[str, QualityReport] = field(default_factory=dict)
    fits: dict[tuple[str, str], FitOutcome] = field(default_factory=dict)
    segment_hints: dict[str, SegmentDetection] = field(default_factory=dict)
    point_masks: dict[str, np.ndarray] = field(default_factory=dict)
    batch_summary: BatchSummary | None = None
    current_index: int = -1
    drt_busy: bool = False
    batch_busy: bool = False
    fit_busy: bool = False

    def invalidate_batch_outputs(self) -> None:
        self.batch_summary = None

    def clear_all(self) -> None:
        self.spectra.clear()
        self.qualities.clear()
        self.fits.clear()
        self.segment_hints.clear()
        self.point_masks.clear()
        self.batch_summary = None
        self.current_index = -1
```

- [ ] **Step 4: 移动三个 Worker 类**

Move `SingleFitWorker`, `BatchFitWorker`, and `MatlabDrtWorker` unchanged into `src/eismaster/ui/workers.py`. Keep their signal signatures unchanged so `MainWindow` only changes imports. Create `tests/test_workers.py`:

```python
from pathlib import Path
from unittest.mock import patch

from eismaster.models import BatchSummary, FitOutcome
from eismaster.ui.workers import BatchFitWorker, SingleFitWorker
from tests.fixture_factory import make_double_arc_spectrum


def test_single_fit_worker_emits_result(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    fit = FitOutcome(model_key="model", model_label="Model", status="ok", message="done")
    emitted: list[tuple[object, object, str]] = []
    worker = SingleFitWorker(spectrum, "model", None)
    worker.finished.connect(lambda result, error, name: emitted.append((result, error, name)))
    with patch("eismaster.ui.workers.fit_spectrum", return_value=fit):
        worker.run()
    assert emitted == [(fit, None, spectrum.display_name)]


def test_single_fit_worker_exposes_original_exception(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    failure = RuntimeError("fit failed")
    emitted: list[tuple[object, object, str]] = []
    worker = SingleFitWorker(spectrum, "model", None)
    worker.finished.connect(lambda result, error, name: emitted.append((result, error, name)))
    with patch("eismaster.ui.workers.fit_spectrum", side_effect=failure):
        worker.run()
    assert emitted == [(None, failure, spectrum.display_name)]


def test_batch_worker_forwards_progress_and_result(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    summary = BatchSummary(model_key="auto", items=[])
    progress: list[tuple[int, int, str]] = []
    finished: list[tuple[object, object]] = []
    worker = BatchFitWorker([spectrum])
    worker.progress.connect(lambda index, total, name: progress.append((index, total, name)))
    worker.finished.connect(lambda result, error: finished.append((result, error)))

    def run_batch(spectra, *, progress_callback):
        item = type("Item", (), {"spectrum": spectrum})()
        progress_callback(1, 1, item)
        return summary

    with patch("eismaster.ui.workers.analyze_batch_auto", side_effect=run_batch):
        worker.run()
    assert progress == [(1, 1, spectrum.display_name)]
    assert finished == [(summary, None)]
```

- [ ] **Step 5: 更新主窗口调用**

Replace local class definitions with:

```python
from eismaster.ui.state import AppState
from eismaster.ui.workers import BatchFitWorker, MatlabDrtWorker, SingleFitWorker
```

Make `_invalidate_batch_outputs()` call `self.state.invalidate_batch_outputs()` and `_clear_all_spectra()` call `self.state.clear_all()` before clearing widgets.

- [ ] **Step 6: 验证状态和 UI 冒烟测试**

Run:

```powershell
$env:PYTHONPATH='src;.'
$env:QT_QPA_PLATFORM='offscreen'
py -3.11 -m pytest tests/test_state.py tests/test_workers.py tests/test_ui_smoke.py -q
```

Expected: 全部通过，0 failures。

- [ ] **Step 7: Review and first-principles gate**

Run:

```powershell
rg -n "^class (AppState|SingleFitWorker|BatchFitWorker|MatlabDrtWorker)" src/eismaster/ui
```

Expected: 每个类只定义一次；`main_window.py` 不再拥有后台执行逻辑。

---

### Task 6: 把 UI 文本和表格格式化改成纯函数

**Files:**
- Create: `src/eismaster/ui/presenters.py`
- Modify: `src/eismaster/ui/main_window.py`
- Create: `tests/test_presenters.py`

**Interfaces:**
- Consumes: `FitOutcome | None`、`BatchSummary | None`、busy 状态。
- Produces: `fit_status_text(...) -> str`、`format_stat(value) -> str`、`fit_error_summary(fit) -> str`。

- [ ] **Step 1: 写入纯函数测试**

Create `tests/test_presenters.py`:

```python
from eismaster.models import FitOutcome
from eismaster.ui.presenters import fit_error_summary, fit_status_text, format_stat


def test_fit_status_text_exposes_running_and_failed_states() -> None:
    assert fit_status_text(None, None, fit_busy=True, batch_busy=False, drt_busy=False) == "正在拟合..."
    fit = FitOutcome(model_key="m", model_label="M", status="failed", message="bad fit")
    assert fit_status_text(fit, None, fit_busy=False, batch_busy=False, drt_busy=False) == "拟合失败"


def test_format_stat_is_stable() -> None:
    assert format_stat(None) == ""
    assert format_stat(float("nan")) == "nan"
    assert format_stat(1.234) == "1.23"


def test_error_summary_uses_primary_parameter_errors() -> None:
    fit = FitOutcome(
        model_key="zview_segmented_rq_rwo",
        model_label="M",
        status="warn",
        message="warning",
        statistics={"Rs_stderr_pct": 5.0, "Rct_stderr_pct": 25.0},
    )
    assert fit_error_summary(fit) == "超误差:Rct 25.0%"
```

- [ ] **Step 2: 运行并确认模块缺失**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_presenters.py -q
```

Expected: FAIL with `ModuleNotFoundError: eismaster.ui.presenters`。

- [ ] **Step 3: 实现并替换窗口内对应方法**

Create `src/eismaster/ui/presenters.py` with these pure implementations; it must not import Qt:

```python
from __future__ import annotations

import numpy as np

from eismaster.models import BatchSummary, FitOutcome


def format_stat(value: object) -> str:
    if value is None:
        return ""
    number = float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.2f}"


def _primary_error_pairs(fit: FitOutcome) -> list[tuple[str, str]]:
    if fit.model_key == "zview_double_rq_qrwo":
        return [("Rs_stderr_pct", "Rs"), ("Rsei_stderr_pct", "Rsei"), ("Rct_stderr_pct", "Rct")]
    return [("Rs_stderr_pct", "Rs"), ("Rct_stderr_pct", "Rct")]


def fit_error_summary(fit: FitOutcome | None) -> str:
    if fit is None:
        return ""
    values: list[tuple[str, float]] = []
    for key, label in _primary_error_pairs(fit):
        raw = fit.statistics.get(key)
        if raw is None:
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append((label, value))
    if not values:
        return "未计算误差"
    high = [f"{label} {value:.1f}%" for label, value in values if value > 20.0]
    if high:
        return "超误差:" + "，".join(high)
    return "，".join(f"{label} {value:.1f}%" for label, value in values)


def fit_status_text(
    fit: FitOutcome | None,
    batch_summary: BatchSummary | None,
    *,
    fit_busy: bool,
    batch_busy: bool,
    drt_busy: bool,
) -> str:
    if drt_busy:
        return "DRT 运行中"
    if batch_busy:
        return "批量拟合进行中"
    if fit_busy:
        return "正在拟合..."
    if fit is not None:
        labels = {"ok": "拟合正常", "warn": "拟合警告", "failed": "拟合失败", "unavailable": "拟合不可用"}
        return labels.get(fit.status, fit.status)
    if batch_summary is not None and batch_summary.items:
        total = len(batch_summary.items)
        done = sum(item.fit is not None for item in batch_summary.items)
        failed = any(item.fit is not None and item.fit.status == "failed" for item in batch_summary.items)
        warned = any(item.fit is not None and item.fit.status == "warn" for item in batch_summary.items)
        suffix = "，含失败项目" if failed else "，含警告项目" if warned else ""
        return f"批量完成 {done}/{total}{suffix}"
    return "等待中"
```

Replace `_format_stat`、`_fit_error_summary` and状态文本分支 with calls to these functions. Preserve existing Chinese labels exactly.

- [ ] **Step 4: 验证 UI 行为**

Run:

```powershell
$env:PYTHONPATH='src;.'
$env:QT_QPA_PLATFORM='offscreen'
py -3.11 -m pytest tests/test_presenters.py tests/test_ui_smoke.py -q
```

Expected: 全部通过，0 failures。

- [ ] **Step 5: Review and first-principles gate**

The presenter module must remain independent of PySide6 and qfluentwidgets. Run:

```powershell
rg -n "PySide6|qfluentwidgets" src/eismaster/ui/presenters.py
```

Expected: no matches。

---

### Task 7: 逐步收敛拟合和批量逻辑

**Files:**
- Create: `src/eismaster/analysis/fitting_math.py`
- Modify: `src/eismaster/analysis/fitting.py`
- Modify: `src/eismaster/analysis/batch.py`
- Modify: `tests/test_fitting_algorithms.py`
- Modify: `tests/test_batch_auto.py`

**Interfaces:**
- Consumes: complex impedance arrays、`FitConfig`、`FitOutcome`。
- Produces: 纯数值辅助函数和不变的 `fit_spectrum(...) -> FitOutcome`、`analyze_batch_auto(...) -> BatchSummary`。

- [ ] **Step 1: 锁定公开签名和状态集合**

Add assertions:

```python
import inspect

from eismaster.analysis.batch import analyze_batch_auto
from eismaster.analysis.fitting import fit_spectrum


def test_public_fit_signatures_remain_stable() -> None:
    assert "arc_ranges" in inspect.signature(fit_spectrum).parameters
    assert "progress_callback" in inspect.signature(analyze_batch_auto).parameters
```

Keep fit statuses limited to `ok`, `warn`, `failed`, and the explicit batch-only `unavailable` status.

- [ ] **Step 2: 提取无状态数值函数**

Move these definitions without behavior changes into `fitting_math.py`:

```text
adaptive_weight_floor
estimate_cpe_n
local_noise_estimate
zview_warburg_open
```

Expose names without a leading underscore inside `fitting_math.py`; import them back under their existing private aliases in `fitting.py` to preserve current tests and callers.

- [ ] **Step 3: 运行拟合算法测试**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_fitting_algorithms.py tests/test_parsers.py -q
```

Expected: 全部通过，0 failures；数值容差不放宽。

- [ ] **Step 4: 删除批量模块中的重复决策来源**

Keep these constants as the only model-selection source:

```python
SINGLE_MODEL_KEY = "zview_segmented_rq_rwo"
DOUBLE_MODEL_KEY = "zview_double_rq_qrwo"
DOUBLE_SWITCH_COUNT = 2
SINGLE_SWITCH_COUNT = 2
```

Make `_worker_auto`、sequential path and parallel path all call the same `_choose_auto_item` function. Do not duplicate candidate scoring or hysteresis in worker-specific branches.

- [ ] **Step 5: 运行批量回归**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_batch_auto.py tests/test_batch_runtime.py -q
```

Expected: 全部通过，0 failures。

- [ ] **Step 6: Review and first-principles gate**

Run:

```powershell
py -3.11 -m ruff check src/eismaster/analysis/fitting.py src/eismaster/analysis/fitting_math.py src/eismaster/analysis/batch.py
```

Expected: 0 errors。若函数仍需在两个模块间来回传递大量内部状态，则停止继续拆分，保留当前边界。

---

### Task 8: 让 MATLAB DRT 和打包失败尽早暴露

**Files:**
- Modify: `src/eismaster/matlab_drt.py`
- Modify: `tests/test_matlab_drt.py`
- Modify: `EISMaster.spec`
- Create: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `MatlabDrtConfig`、输入目录、输出目录、PyInstaller 资源根目录。
- Produces: `validate_matlab_drt_config(config) -> None`；失败时抛出明确 `FileNotFoundError` 或 `RuntimeError`。

- [ ] **Step 1: 写入严格失败测试**

Add to `tests/test_matlab_drt.py`:

```python
import subprocess
from unittest.mock import patch

from eismaster.matlab_drt import validate_matlab_drt_config


def test_missing_matlab_executable_is_rejected(self) -> None:
    config = MatlabDrtConfig(matlab_exe=str(self.tmp_root / "missing.exe"), drttools_dir=str(self.tmp_root))
    with self.assertRaisesRegex(FileNotFoundError, "MATLAB executable"):
        validate_matlab_drt_config(config)


def test_nonzero_matlab_exit_is_rejected(self) -> None:
    matlab = self.tmp_root / "matlab.exe"
    matlab.write_bytes(b"")
    drttools = self.tmp_root / "drttools"
    drttools.mkdir()
    runner = self.tmp_root / "matlab_bridge" / "eismaster_batch_drt.m"
    runner.parent.mkdir()
    runner.write_text("function eismaster_batch_drt(varargin)\nend\n", encoding="utf-8")
    staging_dir = self.tmp_root / "staging"
    staging_dir.mkdir()
    output_dir = self.tmp_root / "results"
    valid_config = MatlabDrtConfig(matlab_exe=str(matlab), drttools_dir=str(drttools))
    completed = subprocess.CompletedProcess(["matlab"], returncode=1, stdout="", stderr="DRT failed")
    with patch("eismaster.matlab_drt._matlab_runner_path", return_value=runner), patch(
        "eismaster.matlab_drt.subprocess.run", return_value=completed
    ):
        with self.assertRaisesRegex(RuntimeError, "DRT failed"):
            run_matlab_drt(valid_config, staging_dir, output_dir)
```

- [ ] **Step 2: 实现配置与结果校验**

Implement:

```python
def validate_matlab_drt_config(config: MatlabDrtConfig) -> None:
    matlab = Path(config.matlab_exe)
    if not matlab.is_file():
        raise FileNotFoundError(f"MATLAB executable not found: {matlab}")
    drttools = Path(config.drttools_dir)
    if not drttools.is_dir():
        raise FileNotFoundError(f"DRTtools directory not found: {drttools}")
    runner = _matlab_runner_path()
    if not runner.is_file():
        raise FileNotFoundError(f"MATLAB runner not found: {runner}")
```

Call it before staging the process. After `subprocess.run`, raise `RuntimeError` on nonzero return code and include trimmed stderr.

- [ ] **Step 3: 添加打包资源测试**

Create `tests/test_packaging.py`:

```python
from pathlib import Path


def test_pyinstaller_spec_bundles_matlab_resources() -> None:
    text = Path("EISMaster.spec").read_text(encoding="utf-8")
    assert '"matlab_bridge"' in text
    assert '"matlab-DRTtools-local"' in text
    assert "datas.append" in text
```

- [ ] **Step 4: 验证模块和资源声明**

Run:

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m pytest tests/test_matlab_drt.py tests/test_packaging.py -q
```

Expected: 全部通过，0 failures。

- [ ] **Step 5: 构建并做真实启动检查**

Use the documented Python 3.11 environment and run:

```powershell
pyinstaller --noconfirm EISMaster.spec
Test-Path 'dist\EISMaster\EISMaster.exe'
Test-Path 'dist\EISMaster\_internal\matlab_bridge\eismaster_batch_drt.m'
Test-Path 'dist\EISMaster\_internal\matlab-DRTtools-local'
```

Expected: build exit code 0 and all three `Test-Path` results are `True`。如果本地没有合法 DRTtools 源目录，构建必须失败并明确指出资源缺失，不能产出不完整发布包。

- [ ] **Step 6: Review and first-principles gate**

Launch the built executable and complete import、single fit、batch fit、DRT configuration、export smoke checks. Report only passed/failed counts and key failures.

---

### Task 9: 清理仓库边界并建立 Rust/Tauri 替换门槛

**Files:**
- Modify: `.gitignore`
- Inspect: `rewrite/`
- Inspect: `rust-rewrite-plan/FEATURE_MATRIX.md`
- Create: `docs/rust-parity-verdict.md`

**Interfaces:**
- Consumes: Python 固定样例输出和 Rust CLI/Tauri 输出。
- Produces: 明确的 `continue`、`pause` 或 `replace-candidate` 结论；没有对照证据时只能是 `pause`。

- [ ] **Step 1: 排除生成目录**

Add to `.gitignore`:

```gitignore
rewrite/target/
rewrite/apps/eismaster-tauri/node_modules/
rewrite/apps/eismaster-tauri/dist/
dist_rebuild*/
```

Do not ignore Rust source, Cargo manifests, frontend source, audit reports, or fixture source files.

- [ ] **Step 2: 运行 Python 固定基准**

Run the parser、fitting、export tests from Tasks 3、4、7 and save only numeric summaries and SHA256 hashes of generated contract files in `docs/rust-parity-verdict.md`.

- [ ] **Step 3: 运行 Rust 对照**

Run:

```powershell
Push-Location rewrite
cargo fmt --check
cargo test --workspace
cargo run -p eismaster-cli -- --help
Pop-Location
```

Expected: all commands exit 0 before any feature is marked equivalent。

- [ ] **Step 4: 按功能矩阵给出结论**

The verdict document must contain exact rows for parser、quality、single fit、double fit、batch、MATLAB staging、DRT export、GUI smoke、installer. Each row records Python evidence、Rust evidence、numeric tolerance、result. A missing evidence cell is `not verified`, never `pass`.

- [ ] **Step 5: Review and first-principles gate**

The default decision is:

- `continue`: Rust passes automated parity but still lacks complete GUI/release parity.
- `pause`: any core parser/fitting/export contract differs or cannot be reproduced.
- `replace-candidate`: every matrix row passes and packaged GUI smoke test passes.

No production switch occurs inside this task.

---

### Task 10: 建立持续验证和最终验收

**Files:**
- Create: `.github/workflows/python-ci.yml`
- Modify: `README.md`
- Modify: `UPLOAD_NOTES.md`

**Interfaces:**
- Consumes: Tasks 2–9 的稳定测试和构建命令。
- Produces: 每次提交可重复执行的静态检查、单元测试和发布前检查。

- [ ] **Step 1: 添加 Windows CI**

Create `.github/workflows/python-ci.yml`:

```yaml
name: python-ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m ruff check src tests
      - run: python -m pytest
        env:
          QT_QPA_PLATFORM: offscreen
```

- [ ] **Step 2: 运行完整本地验证**

Run:

```powershell
$env:PYTHONPATH='src;.'
$env:QT_QPA_PLATFORM='offscreen'
py -3.11 -m ruff check src tests
py -3.11 -m pytest
git diff --check
```

Expected: 3 commands pass, 0 failures。

- [ ] **Step 3: 更新使用和发布文档**

README must state Python 3.11、installation、test、source launch、release launch、MATLAB dependency、DRTtools resource requirement. `UPLOAD_NOTES.md` must list exact release artifact path and the smoke checks required before upload.

- [ ] **Step 4: 执行最终 GUI 验收**

Run the source app and packaged app separately. For each, verify:

1. import TXT and BIN;
2. render Nyquist and Bode;
3. single fit;
4. double fit;
5. batch fit;
6. export workbook;
7. configure MATLAB DRT;
8. close without orphan process.

Report: passed count、failed count、关键失败；不逐条展开已通过步骤。

- [ ] **Step 5: Final Review and first-principles gate**

Confirm all of the following:

- no repository-root private fixture dependency;
- no duplicated AppState or Worker definitions;
- no silent MATLAB nonzero exit;
- no export sheet-name drift;
- no generated Rust/Node/build directories in source scope;
- no claim of Rust parity without current evidence;
- current user-owned modifications were preserved or intentionally incorporated.

Only after this checklist and the full verification command pass may the optimization be called complete.

## Execution Order and Parallelism

```text
Task 1 → Task 2 → Task 3 → Task 4
                         ├→ Task 5 → Task 6
                         ├→ Task 7
                         └→ Task 8
Tasks 6, 7, 8 complete → Task 9 → Task 10
```

- Tasks 5、7、8 may be handled by separate subagents after Task 4 passes because their write sets are disjoint.
- Task 6 depends on Task 5.
- Task 9 depends on stable Python contracts from Tasks 3、4、7、8.
- Task 10 is the only final completion gate.

## Commit Policy

After each task passes its own tests and review, show the exact diff and ask for user authorization before committing. Recommended commit messages:

```text
test: make EIS fixtures reproducible
refactor: stabilize export contracts
refactor: extract UI state and workers
refactor: isolate UI presenters
refactor: isolate fitting math
fix: validate MATLAB DRT execution
chore: define Rust parity gate
ci: add Windows Python validation
```
