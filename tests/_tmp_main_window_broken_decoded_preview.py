from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from eismaster.analysis.batch import analyze_batch_auto
from eismaster.analysis.circuits import TEMPLATES
from eismaster.analysis.fitting import fit_spectrum
from eismaster.analysis.quality import assess_spectrum_quality
from eismaster.analysis.segmentation import SegmentDetection, detect_segments
from eismaster.exporters import export_batch_summary, export_fit_results, export_spectrum_bundle
from eismaster.io import load_spectra_from_folder, load_spectrum
from eismaster.matlab_drt import MatlabDrtConfig, MatlabDrtResult, run_matlab_drt, stage_matlab_drt_inputs
from eismaster.models import BatchSummary, FitOutcome, QualityReport, SpectrumData
from PySide6.QtCore import QObject, QSignalBlocker, QThread, Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None
from qfluentwidgets import (
    MSFluentWindow,
    NavigationItemPosition,
    setTheme,
    Theme,
    FluentIcon as FIF,
    PushButton,
    PrimaryPushButton,
    TransparentPushButton,
    ComboBox,
    LineEdit,
    TableWidget,
    ProgressBar,
    TableItemDelegate,
    ScrollArea,
    SimpleCardWidget,
    TitleLabel,
    BodyLabel,
    CaptionLabel,
    StrongBodyLabel,
    IndeterminateProgressBar,
    InfoBar,
    InfoBarIcon,
    InfoBarPosition,
)
INSPECT_TAB = 0
FIT_TAB = 1
BATCH_TAB = 2
if pg is not None:
    class InteractiveViewBox(pg.ViewBox):
        def __init__(self) -> None:
            super().__init__(enableMenu=False)
            self.setMouseMode(self.RectMode)
            self.enableAutoRange(x=False, y=False)
        def wheelEvent(self, event, axis=None) -> None:  # pragma: no cover
            event.ignore()
        def mouseDoubleClickEvent(self, event) -> None:  # pragma: no cover
            if event.button() == Qt.LeftButton:
                self.autoRange()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)
class MatlabDrtWorker(QObject):
    finished = Signal(object, object)
    def __init__(self, config: MatlabDrtConfig, spectra: list[SpectrumData], export_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.spectra = spectra
        self.export_dir = export_dir
    def run(self) -> None:
        try:
            staging_dir = stage_matlab_drt_inputs(self.spectra, self.export_dir)
            result = run_matlab_drt(self.config, staging_dir, self.export_dir / "results")
            self.finished.emit(result, None)
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, exc)
class BatchFitWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, object)
    def __init__(self, spectra: list[SpectrumData]) -> None:
        super().__init__()
        self.spectra = spectra
    def run(self) -> None:
        try:
            summary = analyze_batch_auto(self.spectra, progress_callback=self._emit_progress)
            self.finished.emit(summary, None)
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, exc)
    def _emit_progress(self, index: int, total: int, item) -> None:
        self.progress.emit(index, total, item.spectrum.display_name)
@dataclass
class AppState:
    spectra: list[SpectrumData] = field(default_factory=list)
    qualities: dict[str, QualityReport] = field(default_factory=dict)
    fits: dict[tuple[str, str], FitOutcome] = field(default_factory=dict)
    segment_hints: dict[str, SegmentDetection] = field(default_factory=dict)
    batch_summary: BatchSummary | None = None
    current_index: int = -1
    drt_busy: bool = False
    batch_busy: bool = False
class MainWindow(MSFluentWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self._matlab_thread: QThread | None = None
        self._matlab_worker: MatlabDrtWorker | None = None
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchFitWorker | None = None
        self._last_matlab_result: MatlabDrtResult | None = None
        self._last_drt_spectra: list[SpectrumData] = []
        self._queue_tables: dict[str, TableWidget] = {}
        self._queue_meta_boxes: dict[str, QTextEdit] = {}
        self._batch_plot_cache: dict[str, dict[str, object]] = {}
        self._segment_handles: dict[str, object] = {}
  