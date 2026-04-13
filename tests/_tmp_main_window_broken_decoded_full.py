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
        self._segment_overlay_items: list[object] = []
        self._segment_handle_sync = False
        self._segment_drag_active = False
        self._segment_control_sync = False
        self._fit_plot_click_connected = False
        # 1. UI Initialization Settings
        setTheme(Theme.DARK)
        self.setMicaEffectEnabled(True)
        self.setWindowTitle("EISMaster æºè½é»æå·¥ä½å°")
        self.setMinimumSize(1200, 780)
        self.resize(1500, 960)
        # 2. Main Page Setup
        self._build_ui()
        self._configure_plots()
        self._clear_views()
        # Ensure focus on the first page
        self.navigationInterface.setCurrentItem(self.inspect_widget.objectName())
    def _build_ui(self) -> None:
        # Create interface pages
        self.inspect_widget, _ = self._build_inspect_tab()
        self.fit_widget, _ = self._build_fit_tab()
        self.batch_widget, _ = self._build_batch_tab()
        # Set object names for navigation tracking
        self.inspect_widget.setObjectName("inspectInterface")
        self.fit_widget.setObjectName("fitInterface")
        self.batch_widget.setObjectName("batchInterface")
        # Register interfaces to MSFluentWindow sidebar
        self.addSubInterface(self.inspect_widget, FIF.TAG, "æ°æ®æ£æ¥")
        self.addSubInterface(self.fit_widget, FIF.SETTING, "é»ææå")
        self.addSubInterface(self.batch_widget, FIF.UPDATE, "æ¹å¤çä¸­å¿")
        # Connect navigation signal to refresh content
        self.stackedWidget.currentChanged.connect(self._on_tab_changed)
    def _build_header(self) -> QWidget:
        container = SimpleCardWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(20, 15, 20, 15)
        left = QVBoxLayout()
        self.brand_label = TitleLabel("EISMaster Pro")
        self.subtitle_label = CaptionLabel("é«æ§è½çµåå­¦é»æåæå·¥ä½å° Â· Native DRT Engine")
        left.addWidget(self.brand_label)
        left.addWidget(self.subtitle_label)
        info_row = QHBoxLayout()
        self.instrument_chip = StrongBodyLabel("ä»ªå¨: ç­å¾å¯¼å¥")
        self.selection_chip = BodyLabel("æ ·å: å¾éæ©")
        info_row.addWidget(self.instrument_chip)
        info_row.addSpacing(20)
        info_row.addWidget(self.selection_chip)
        info_row.addStretch(1)
        left.addLayout(info_row)
        layout.addLayout(left, 1)
        metrics = QHBoxLayout()
        self.loaded_metric = self._metric("å·²å è½½ä»½æ°", "0")
        self.fit_metric = self._metric("ææ°æåç¶æ", "åå¤å°±ç»ª")
        metrics.addWidget(self.loaded_metric["container"])
        metrics.addWidget(self.fit_metric["container"])
        layout.addLayout(metrics, 0)
        return container
    def _build_sidebar(self, page_key: str) -> QWidget:
        container = SimpleCardWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        title = StrongBodyLabel("æ ·åéå")
        layout.addWidget(title)
        import_files_btn = None
        import_folder_btn = None
        remove_btn = None
        clear_all_btn = None
        if page_key == "inspect":
            btn_layout = QHBoxLayout()
            import_files_btn = PrimaryPushButton(FIF.FOLDER, "å¯¼å¥")
            import_files_btn.clicked.connect(self._import_files)
            import_folder_btn = PushButton(FIF.FOLDER_ADD, "ç®å½")
            import_folder_btn.clicked.connect(self._import_folder)
            btn_layout.addWidget(import_files_btn)
            btn_layout.addWidget(import_folder_btn)
            layout.addLayout(btn_layout)
        table = TableWidget()
        table.setColumnCount(1)
        table.verticalHeader().hide()
        table.horizontalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setShowGrid(False)
        table.setBorderRadius(8)
        table.itemSelectionChanged.connect(lambda key=page_key: self._on_queue_row_changed(key))
        layout.addWidget(table, 1)
        if page_key == "inspect":
            ctrl_layout = QHBoxLayout()
            self.move_up_btn = PushButton(FIF.UP, "ä¸ç§»")
            self.move_up_btn.clicked.connect(self._move_current_spectrum_up)
            self.move_down_btn = PushButton(FIF.DOWN, "ä¸ç§»")
            self.move_down_btn.clicked.connect(self._move_current_spectrum_down)
            remove_btn = PushButton(FIF.DELETE, "ç§»é¤")
            remove_btn.clicked.connect(self._remove_current_spectrum)
            ctrl_layout.addWidget(self.move_up_btn)
            ctrl_layout.addWidget(self.move_down_btn)
            clear_all_btn = PushButton(FIF.CLEAR_SELECTION, "æ¸ç©ºå¨é¨")
            clear_all_btn.clicked.connect(self._clear_all_spectra)
            ctrl_layout.addStretch(1)
            ctrl_layout.addWidget(clear_all_btn)
            ctrl_layout.addWidget(remove_btn)
            layout.addLayout(ctrl_layout)
        meta_box = self._info_box()
        meta_box.setMaximumHeight(150)
        layout.addWidget(meta_box)
        self._queue_tables[page_key] = table
        self._queue_meta_boxes[page_key] = meta_box
        if page_key == "inspect":
            self.file_list = table
            self.remove_btn = remove_btn
            self.clear_all_btn = clear_all_btn
            self.sidebar_meta = meta_box
            self.import_files_btn = import_files_btn
            self.import_folder_btn = import_folder_btn
        return container
    def _build_inspect_tab(self) -> tuple[QWidget, QVBoxLayout]:
        tab, layout = self._tab_shell()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._build_header())
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._build_sidebar("inspect"), 1)
        split.addWidget(left_container)
        right_panel = self._panel("å®éªæ°æ®åæ", "å½åè°±å¾ç Nyquist ä¸ Bode å¾å±ç¤ºï¼ä»¥åè´¨éè¯ä¼°ç»è®ºã")
        right_layout = right_panel.layout()
        grid = QGridLayout()
        grid.setSpacing(10)
        self.nyquist_plot = self._plot("Nyquist Plot", "Z' [ohm]", "-Z'' [ohm]")
        self.bode_mag_plot = self._plot("Bode Magnitude", "freq [Hz]", "|Z| [ohm]", log_x=True, log_y=True)
        self.bode_phase_plot = self._plot("Bode Phase", "freq [Hz]", "phase [deg]", log_x=True)
        grid.addWidget(self.nyquist_plot, 0, 0, 2, 1)
        grid.addWidget(self.bode_mag_plot, 0, 1)
        grid.addWidget(self.bode_phase_plot, 1, 1)
        right_layout.addLayout(grid, 5)
        self.quality_text = self._info_box()
        self.quality_text.setPlaceholderText("ç­å¾æ°æ®å è½½ä»¥è¿è¡è´¨éè¯ä¼°...")
        right_layout.addWidget(self.quality_text, 1)
        split.addWidget(right_panel)
        split.setSizes([350, 1000])
        layout.addWidget(split, 1)
        table_panel = self._panel("æ°æ®éè§", "å½åè°±å¾çé¢çãé»æå®èé¨ãæ¨¡å¼ä¸ç¸ä½ã")
        self.data_table = self._table()
        table_panel.layout().addWidget(self.data_table)
        layout.addWidget(table_panel, 1)
        return tab, layout
    def _build_fit_tab(self) -> tuple[QWidget, QVBoxLayout]:
        tab, layout = self._tab_shell()
        layout.setContentsMargins(20, 20, 20, 20)
        top = self._panel("ç­æçµè·¯æå", "éå¯¹åæ¡è°±å¾çæå¨æåä¸ DRT å¼å¯¼ã")
        top_layout = top.layout()
        row = QHBoxLayout()
        row.addWidget(StrongBodyLabel("æåæ¨¡æ¿"))
        self.template_combo = ComboBox()
        for key, template in TEMPLATES.items():
            self.template_combo.addItem(template.label, userData=key)
        self.template_combo.currentIndexChanged.connect(self._refresh_fit_for_current_selection)
        row.addWidget(self.template_combo, 1)
        self.fit_current_btn = PrimaryPushButton(FIF.PLAY, "å¼å§æå")
        self.fit_current_btn.clicked.connect(self._fit_current)
        self.export_current_btn = PushButton(FIF.SHARE, "å¯¼åºç»æ")
        self.export_current_btn.clicked.connect(self._export_current_bundle)
        self.run_drt_current_btn = PushButton(FIF.BASKETBALL, "MATLAB DRT")
        self.run_drt_current_btn.clicked.connect(self._run_matlab_drt_current)
        row.addWidget(self.fit_current_btn)
        row.addWidget(self.export_current_btn)
        row.addWidget(self.run_drt_current_btn)
        top_layout.addLayout(row)
        layout.addWidget(top)
        split = QSplitter(Qt.Horizontal)
        segment_panel = self._panel("ååºåæ (ZView)", "åçç¹è®¾ç½®ä»¥ä¼åç­æçµè·¯æååå¼ã")
        segment_layout = QVBoxLayout()
        self.segment_mode_combo = ComboBox()
        self.segment_mode_combo.addItem("èªå¨è¯å«", userData="auto")
        self.segment_mode_combo.addItem("ååå", userData="single")
        self.segment_mode_combo.addItem("ååå", userData="double")
        self.segment_mode_combo.currentIndexChanged.connect(self._refresh_fit_for_current_selection)
        self.segment_split1_spin = LineEdit()
        self.segment_split1_spin.setPlaceholderText("ç´¢å¼ 1")
        self.segment_split1_spin.setFixedWidth(80)
        self.segment_split2_spin = LineEdit()
        self.segment_split2_spin.setPlaceholderText("ç´¢å¼ 2")
        self.segment_split2_spin.setFixedWidth(80)
        self.segment_detect_btn = PushButton(FIF.SEARCH, "æºè½è¯å«")
        self.segment_detect_btn.clicked.connect(self._detect_segments_for_current)
        segment_layout.addWidget(BodyLabel("æ¨¡å¼:"))
        segment_top = QHBoxLayout()
        segment_layout.addLayout(segment_top)
        segment_top.addWidget(BodyLabel("æ¨¡å¼:"))
        segment_top.addWidget(self.segment_mode_combo)
        segment_top.addStretch(1)
        segment_layout.addWidget(BodyLabel("åçç¹:"))
        segment_grid = QGridLayout()
        segment_grid.setHorizontalSpacing(10)
        segment_grid.setVerticalSpacing(8)
        self.segment_split1_freq_label = BodyLabel("split1 Frequency [Hz]")
        self.segment_split1_freq_edit = LineEdit()
        self.segment_split1_freq_edit.setPlaceholderText("split1 Frequency [Hz]")
        self.segment_split1_freq_edit.setFixedWidth(150)
        self.segment_split1_freq_edit.editingFinished.connect(lambda: self._on_segment_frequency_edited("split1"))
        self.segment_split1_index_label = CaptionLabel("Index")
        self.segment_split1_spin.setReadOnly(True)
        self.segment_split1_spin.setFixedWidth(72)
        self.segment_split2_freq_label = BodyLabel("split2 Frequency [Hz]")
        self.segment_split2_freq_edit = LineEdit()
        self.segment_split2_freq_edit.setPlaceholderText("split2 Frequency [Hz]")
        self.segment_split2_freq_edit.setFixedWidth(150)
        self.segment_split2_freq_edit.editingFinished.connect(lambda: self._on_segment_frequency_edited("split2"))
        self.segment_split2_index_label = CaptionLabel("Index")
        self.segment_split2_spin.setReadOnly(True)
        self.segment_split2_spin.setFixedWidth(72)
        segment_grid.addWidget(self.segment_split1_freq_label, 0, 0)
        segment_grid.addWidget(self.segment_split1_freq_edit, 0, 1)
        segment_grid.addWidget(self.segment_split1_index_label, 0, 2)
        segment_grid.addWidget(self.segment_split1_spin, 0, 3)
        segment_grid.addWidget(self.segment_split2_freq_label, 1, 0)
        segment_grid.addWidget(self.segment_split2_freq_edit, 1, 1)
        segment_grid.addWidget(self.segment_split2_index_label, 1, 2)
        segment_grid.addWidget(self.segment_split2_spin, 1, 3)
        segment_grid.setColumnStretch(4, 1)
        segment_layout.addLayout(segment_grid)
        self.segment_split2_widgets = [
            self.segment_split2_freq_label,
            self.segment_split2_freq_edit,
            self.segment_split2_index_label,
            self.segment_split2_spin,
        ]
        self.segment_mode_status = CaptionLabel("Segment mode: pending")
        self.segment_peak_info_label = CaptionLabel("Peaks: -")
        self.segment_arc1_label = CaptionLabel("Arc1: -")
        self.segment_arc2_label = CaptionLabel("Arc2: -")
        self.segment_tail_label = CaptionLabel("Tail: -")
        segment_layout.addWidget(self.segment_mode_status)
        segment_layout.addWidget(self.segment_peak_info_label)
        segment_layout.addWidget(self.segment_arc1_label)
        segment_layout.addWidget(self.segment_arc2_label)
        segment_layout.addWidget(self.segment_tail_label)
        segment_panel.layout().addLayout(segment_layout)
        layout.addWidget(segment_panel)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_sidebar("fit"))
        self.param_table = self._table()
        result_panel = self._panel("æåç®æ¥", "å½åæ ·åçç©çåæ°æ»ç»ä¸è¯¯å·®è¯ä¼°ã")
        self.fit_text = self._info_box()
        result_panel.layout().addWidget(self.fit_text, 1)
        split.addWidget(result_panel)
        plot_panel = self._panel("ç»æé¢è§", "Nyquist å¯¹ç§å¾ã")
        plot_layout = QVBoxLayout()
        self.fit_nyquist_plot = self._plot("Nyquist Fit", "Z' [ohm]", "-Z'' [ohm]")
        self.fit_residual_plot = self._plot("Residuals [%]", "freq [Hz]", "Error", log_x=True)
        plot_layout.addWidget(self.fit_nyquist_plot, 3)
        plot_layout.addWidget(self.fit_residual_plot, 1)
        plot_panel.layout().addLayout(plot_layout)
        split.addWidget(plot_panel)
        split.setSizes([320, 320, 900])
        layout.addWidget(split, 1)
        batch_actions = QHBoxLayout()
        self.fit_batch_btn = PrimaryPushButton(FIF.PLAY, "åæ­¥è³æ¹å¤ç")
        self.fit_batch_btn.clicked.connect(self._fit_batch)
        self.export_current_btn = PushButton(FIF.SHARE, "å¯¼åºå½åç»æ")
        self.export_current_btn.clicked.connect(self._export_current_bundle)
        batch_actions.addWidget(self.fit_batch_btn)
        batch_actions.addWidget(self.export_current_btn)
        batch_actions.addStretch(1)
        batch_footer = self._panel("å¿«éæä½", "å°å½åæåéç½®åºç¨å°å¨å±éåæå¯¼åºæ¥åã")
        batch_footer.layout().addLayout(batch_actions)
        layout.addWidget(batch_footer)
        return tab, layout
    def _build_batch_tab(self) -> tuple[QWidget, QVBoxLayout]:
        tab, layout = self._tab_shell()
        layout.setContentsMargins(20, 20, 20, 20)
        batch_top_panel = self._panel("Batch Actions", "Run batch jobs, view progress, and export results.")
        batch_top_layout = batch_top_panel.layout()
        action_layout = QHBoxLayout()
        self.export_batch_btn = PushButton(FIF.SHARE, "å¯¼åºæ¹éç»æ")
        self.export_batch_btn.clicked.connect(self._export_batch_bundle)
        self.run_drt_batch_btn = PushButton(FIF.BASKETBALL, "æ¹é MATLAB DRT")
        self.run_drt_batch_btn.clicked.connect(self._run_matlab_drt_batch)
        action_layout.addWidget(self.fit_batch_btn)
        action_layout.addWidget(self.export_batch_btn)
        action_layout.addWidget(self.run_drt_batch_btn)
        action_layout.addStretch(1)
        batch_top_layout.addLayout(action_layout)
        self.batch_progress_label = BodyLabel("ç­å¾å¼å§æ¹éæå")
        self.batch_progress = ProgressBar()
        self.batch_progress.setValue(0)
        batch_top_layout.addWidget(self.batch_progress_label)
        batch_top_layout.addWidget(self.batch_progress)
        layout.addWidget(batch_top_panel)
        summary_panel = self._panel("æ¹å¤çæè¦", "æ¾ç¤ºå½åæ¹æ¬¡çæ»ä½ç¶æã")
        self.batch_text = self._info_box()
        self.batch_text.setMaximumHeight(64)
        batch_top_layout.addWidget(self.batch_text)
        batch_top_panel.setMaximumHeight(190)
        top = self._panel("æ¹å¤çæè¦ & è¶å¿", "æ»è§å½åæ¹æ¬¡çæåæ¶ææ§ä¸åæ°æ¼åã")
        summary_split = QSplitter(Qt.Horizontal)
        summary_split.setChildrenCollapsible(False)
        summary_split.addWidget(self._build_sidebar("batch"))
        trend_container = QWidget()
        trend_layout = QVBoxLayout(trend_container)
        trend_cfg = QHBoxLayout()
        self.trend_rs_check = QCheckBox("Rs")
        self.trend_rs_check.setChecked(True)
        self.trend_rsei_check = QCheckBox("Rsei")
        self.trend_rsei_check.setChecked(True)
        self.trend_rct_check = QCheckBox("Rct")
        self.trend_rct_check.setChecked(True)
        self.trend_rs_check.toggled.connect(self._refresh_batch_plot)
        self.trend_rsei_check.toggled.connect(self._refresh_batch_plot)
        self.trend_rct_check.toggled.connect(self._refresh_batch_plot)
        trend_cfg.addWidget(self.trend_rs_check)
        trend_cfg.addWidget(self.trend_rsei_check)
        trend_cfg.addWidget(self.trend_rct_check)
        trend_layout.addLayout(trend_cfg)
        self.batch_plot = self._plot("Batch Trends", "Index", "Resistance [ohm]")
        trend_layout.addWidget(self.batch_plot)
        summary_split.addWidget(trend_container)
        summary_split.setSizes([320, 900])
        top.layout().addWidget(summary_split)
        layout.addWidget(top)
        table_panel = self._panel("æ°æ®ç»æç©éµ", "è¯¦ç»çæååæ°ä¸è¯¯å·®ç»è®¡è¡¨ã")
        self.batch_table = self._table()
        self.batch_table.itemSelectionChanged.connect(self._on_batch_table_selection_changed)
        table_panel.layout().addWidget(self.batch_table)
        layout.addWidget(table_panel, 1)
        matlab_panel = self._panel("MATLAB DRT èå¨éç½®", "é«æ§è½ Tikhonov åè¾¨çå¢å¼ºå·¥ä½å°ã")
        grid = QGridLayout()
        defaults = MatlabDrtConfig()
        self.matlab_exe_edit = LineEdit()
        self.matlab_exe_edit.setText(defaults.matlab_exe)
        self.drttools_dir_edit = LineEdit()
        self.drttools_dir_edit.setText(defaults.drttools_dir)
        self.matlab_lambda_edit = LineEdit()
        self.matlab_lambda_edit.setText(f"{defaults.lambda_value:.6g}")
        self.matlab_coeff_edit = LineEdit()
        self.matlab_coeff_edit.setText(f"{defaults.coeff_value:.6g}")
        self.matlab_method_combo = ComboBox()
        for label, key in [("Simple", "simple"), ("Bayesian", "credit"), ("BHT", "BHT"), ("Peak", "peak")]:
            self.matlab_method_combo.addItem(label, userData=key)
        self.matlab_drt_type_combo = ComboBox()
        for label, value in [("tau / gamma", 1), ("freq / gamma", 2), ("tau / g", 3), ("freq / g", 4)]:
            self.matlab_drt_type_combo.addItem(label, userData=value)
        self.matlab_inductance_combo = ComboBox()
        for label, value in [("Ignore", 1), ("Fit", 2), ("Discard", 3)]:
            self.matlab_inductance_combo.addItem(label, userData=value)
        matlab_exe_browse = PushButton(FIF.FOLDER, "æµè§")
        matlab_exe_browse.clicked.connect(self._browse_matlab_exe)
        drttools_browse = PushButton(FIF.FOLDER, "æµè§")
        drttools_browse.clicked.connect(self._browse_drttools_dir)
        grid.addWidget(BodyLabel("MATLAB è·¯å¾"), 0, 0)
        grid.addWidget(self.matlab_exe_edit, 0, 1)
        grid.addWidget(matlab_exe_browse, 0, 2)
        grid.addWidget(BodyLabel("DRTtools è·¯å¾"), 1, 0)
        grid.addWidget(self.drttools_dir_edit, 1, 1)
        grid.addWidget(drttools_browse, 1, 2)
        grid.addWidget(BodyLabel("æ¹æ³"), 2, 0)
        grid.addWidget(self.matlab_method_combo, 2, 1)
        grid.addWidget(BodyLabel("DRT type"), 3, 0)
        grid.addWidget(self.matlab_drt_type_combo, 3, 1)
        grid.addWidget(BodyLabel("Lambda"), 4, 0)
        grid.addWidget(self.matlab_lambda_edit, 4, 1)
        grid.addWidget(BodyLabel("Coeff"), 5, 0)
        grid.addWidget(self.matlab_coeff_edit, 5, 1)
        grid.addWidget(BodyLabel("Inductance"), 6, 0)
        grid.addWidget(self.matlab_inductance_combo, 6, 1)
        matlab_panel.layout().addLayout(grid)
        self.matlab_status = self._info_box()
        self.matlab_status.setMaximumHeight(100)
        matlab_panel.layout().addWidget(self.matlab_status)
        self.matlab_log = self._info_box()
        self.matlab_log.setMinimumHeight(200)
        matlab_panel.layout().addWidget(self.matlab_log)
        layout.addWidget(matlab_panel)
        return tab, layout
    def _tab_shell(self) -> tuple[QWidget, QVBoxLayout]:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.viewport().setObjectName("scrollAreaWidgetContents")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        scroll.setWidget(container)
        return scroll, layout
    def _panel(self, title: str, subtitle: str) -> SimpleCardWidget:
        card = SimpleCardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 15)
        layout.setSpacing(10)
        header = QVBoxLayout()
        t = StrongBodyLabel(title)
        s = CaptionLabel(subtitle)
        s.setWordWrap(True)
        header.addWidget(t)
        header.addWidget(s)
        layout.addLayout(header)
        return card
    def _configure_plots(self) -> None:
        if pg is None:
            return
        pg.setConfigOptions(antialias=False, foreground="#A0A0B0")
        for plot in self.findChildren(pg.PlotWidget):
            item = plot.getPlotItem()
            plot.setBackground("#121214")
            plot.setMenuEnabled(False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            item.setDownsampling(auto=True, mode="peak")
            item.setClipToView(True)
            item.getViewBox().setMouseEnabled(x=True, y=True)
            for axis in ("bottom", "left"):
                item.getAxis(axis).setPen(pg.mkPen("#cfd4dc"))
                item.getAxis(axis).setTextPen(pg.mkPen("#6e6e73"))
        if hasattr(self, "bode_phase_plot") and hasattr(self, "bode_mag_plot"):
            self.bode_phase_plot.setXLink(self.bode_mag_plot)
    def _info_box(self) -> QTextEdit:
        box = QTextEdit()
        box.setObjectName("infoPane")
        box.setReadOnly(True)
        return box
    def _chip(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("chipLabel")
        return label
    def _metric(self, label: str, value: str) -> dict[str, QWidget]:
        container = SimpleCardWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        l = CaptionLabel(label)
        v = StrongBodyLabel(value)
        v.setWordWrap(True)
        layout.addWidget(l)
        layout.addWidget(v)
        return {"container": container, "value": v}
    def _plot(self, title: str, x_label: str, y_label: str, *, log_x: bool = False, log_y: bool = False):
        plot = pg.PlotWidget(title=title, viewBox=InteractiveViewBox())
        plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.setLogMode(x=log_x, y=log_y)
        return plot
    def _table(self) -> TableWidget:
        table = TableWidget()
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return table
    def _import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "å¯¼å¥ EIS æä»¶", str(Path.cwd()), "EIS Files (*.bin *.txt *.csv)")
        if not paths:
            return
        spectra: list[SpectrumData] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    spectra.append(load_spectrum(path))
                except Exception as exc:
                    self._warn(f"å è½½å¤±è´¥ï¼
{path}
{exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._merge_state(spectra)
    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "å¯¼å¥æä»¶å¤¹", str(Path.cwd()))
        if not folder:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            spectra = load_spectra_from_folder(folder)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._warn(str(exc))
            return
        QApplication.restoreOverrideCursor()
        self._merge_state(spectra)
    def _replace_state(self, spectra: list[SpectrumData]) -> None:
        self.state = AppState(spectra=spectra)
        self._rebuild_file_list(0 if spectra else -1)
        self._refresh_import_text()
        self._refresh_batch_view()
        if not spectra:
            self._clear_views()
    def _merge_state(self, spectra: list[SpectrumData]) -> None:
        if not spectra:
            return
        existing_paths = {self._spectrum_path_key(spectrum) for spectrum in self.state.spectra}
        appended: list[SpectrumData] = []
        for spectrum in spectra:
            path_key = self._spectrum_path_key(spectrum)
            if path_key in existing_paths:
                continue
            existing_paths.add(path_key)
            appended.append(spectrum)
        if not appended:
            self._warn("å¯¼å¥çæä»¶å·²å¨å½åéåä¸­ã")
            return
        self.state.spectra.extend(appended)
        self._invalidate_batch_outputs()
        new_index = len(self.state.spectra) - len(appended)
        self._rebuild_file_list(new_index)
        self._refresh_import_text()
        self._refresh_current_tab()
        self._update_global_status()
    def _rebuild_file_list(self, current_row: int) -> None:
        for table in self._queue_tables.values():
            blocker = QSignalBlocker(table)
            table.clearContents()
            table.setRowCount(len(self.state.spectra))
            for i, spectrum in enumerate(self.state.spectra):
                item = QTableWidgetItem(f"{spectrum.display_name}
{spectrum.metadata.source_format.upper()} Â· {spectrum.n_points} pt")
                table.setItem(i, 0, item)
            del blocker
        if self.state.spectra:
            row = min(max(current_row, 0), len(self.state.spectra) - 1)
            self._set_current_index(row)
        else:
            self.state.current_index = -1
            self._sync_queue_selection(-1)
    def _refresh_import_text(self) -> None:
        count = len(self.state.spectra)
        self.loaded_metric["value"].setText(str(count))
        if count == 0:
            self.brand_label.setText("EISMaster Pro")
            self.selection_chip.setText("æ ·å: æªéæ©")
            self.instrument_chip.setText("ä»ªå¨: ç­å¾å¯¼å¥")
        else:
            current = self._current_spectrum(silent=True) or self.state.spectra[0]
            self.brand_label.setText(f"EISMaster Pro - {current.display_name}")
    def _sync_queue_selection(self, row: int) -> None:
        for table in self._queue_tables.values():
            blocker = QSignalBlocker(table)
            if 0 <= row < table.rowCount():
                table.setCurrentCell(row, 0)
                table.selectRow(row)
                item = table.item(row, 0)
                if item is not None:
                    table.scrollToItem(item)
            else:
                table.clearSelection()
            del blocker
    def _set_current_index(self, row: int, *, refresh_page: bool = True) -> None:
        if row < 0 or row >= len(self.state.spectra):
            return
        self.state.current_index = row
        self._sync_queue_selection(row)
        spectrum = self.state.spectra[row]
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        self._prime_segment_controls(spectrum)
        self._refresh_sidebar(spectrum, quality)
        self._update_global_status()
        if refresh_page:
            self._refresh_current_tab()
    def _spectrum_path_key(self, spectrum: SpectrumData) -> str:
        return str(spectrum.metadata.file_path.resolve()).lower()
    def _invalidate_batch_outputs(self) -> None:
        self.state.batch_summary = None
        self._last_matlab_result = None
        self._last_drt_spectra = []
        if hasattr(self, "batch_progress"):
            self.batch_progress.setValue(0)
        if hasattr(self, "batch_progress_label"):
            self.batch_progress_label.setText("ç­å¾å¼å§æ¹éæå")
    def _remove_current_spectrum(self) -> None:
        row = self.state.current_index
        if row < 0 or row >= len(self.state.spectra):
            return
        spectrum = self.state.spectra.pop(row)
        self.state.qualities.pop(spectrum.display_name, None)
        self.state.segment_hints.pop(spectrum.display_name, None)
        for key in [key for key in self.state.fits if key[0] == spectrum.display_name]:
            self.state.fits.pop(key, None)
        self._invalidate_batch_outputs()
        self._rebuild_file_list(min(row, len(self.state.spectra) - 1))
        self._refresh_import_text()
        if not self.state.spectra:
            self._clear_views()
            return
        self._set_current_index(self.state.current_index)
    def _clear_all_spectra(self) -> None:
        if not self.state.spectra:
            return
        self._replace_state([])
    def _move_current_spectrum_up(self) -> None:
        row = self.state.current_index
        if row <= 0 or row >= len(self.state.spectra):
            return
        self.state.spectra[row - 1], self.state.spectra[row] = self.state.spectra[row], self.state.spectra[row - 1]
        self._invalidate_batch_outputs()
        self._rebuild_file_list(row - 1)
        self._refresh_import_text()
    def _move_current_spectrum_down(self) -> None:
        row = self.state.current_index
        if row < 0 or row >= len(self.state.spectra) - 1:
            return
        self.state.spectra[row + 1], self.state.spectra[row] = self.state.spectra[row], self.state.spectra[row + 1]
        self._invalidate_batch_outputs()
        self._rebuild_file_list(row + 1)
        self._refresh_import_text()
    def _clear_views(self) -> None:
        for meta_box in self._queue_meta_boxes.values():
            meta_box.setPlainText("å¯¼å¥åï¼è¿éä¼æ¾ç¤ºå½åè°±å¾çå³é®ä¿¡æ¯ã")
        self.quality_text.setPlainText("å°æªéæ©è°±å¾ã")
        self.fit_text.setPlainText("å°æªéæ©è°±å¾ã")
        self.batch_text.setPlainText("å°æªæ§è¡æ¹éæåã")
        self.matlab_status.setPlainText("å°æªè¿è¡ MATLAB DRTã")
        self.matlab_log.setPlainText("")
        if hasattr(self, "segment_split1_freq_edit"):
            self.segment_split1_freq_edit.clear()
            self.segment_split2_freq_edit.clear()
            self.segment_split1_spin.clear()
            self.segment_split2_spin.clear()
            self._update_split_info_labels(None)
        if hasattr(self, "batch_progress") and self.batch_progress:
            self.batch_progress.setValue(0)
        self.data_table.clearContents()
        self.data_table.setRowCount(0)
        self.batch_table.clearContents()
        self.batch_table.setRowCount(0)
        if pg is not None:
            plots = [
                getattr(self, "nyquist_plot", None), 
                getattr(self, "bode_mag_plot", None), 
                getattr(self, "bode_phase_plot", None),
                getattr(self, "fit_nyquist_plot", None),
                getattr(self, "fit_residual_plot", None),
                getattr(self, "batch_plot", None)
            ]
            for plot in plots:
                if plot:
                    self._reset_plot(plot)
            self._clear_segment_overlay()
        self._update_global_status()
    def _update_global_status(self) -> None:
        if not self.state.spectra:
            self.instrument_chip.setText("ä»ªå¨: ç­å¾å¯¼å¥")
            self.selection_chip.setText("æ ·å: æªéæ©")
            return
        current = self._current_spectrum(silent=True)
        self.loaded_metric["value"].setText(str(len(self.state.spectra)))
        if current:
            self.instrument_chip.setText(f"ä»ªå¨: {current.metadata.instrument_model or '-'}")
            self.selection_chip.setText(f"å½å: {current.display_name}")
            self.brand_label.setText(f"EISMaster Pro - {current.display_name}")
        else:
            self.selection_chip.setText("æ ·å: æªéæ©")
        self.fit_metric["value"].setText(self._status_text_for_current(current))
    def _status_text_for_current(self, current: SpectrumData | None) -> str:
        if self.state.drt_busy:
            return "DRT è¿è¡ä¸­"
        if self.state.batch_busy:
            return "æ¹éæåä¸­"
        if current is not None:
            fit = self._latest_fit_for_spectrum(current.display_name)
            if fit is not None:
                labels = {
                    "ok": "åè°±æåå®æ",
                    "warn": "åè°±æåè­¦å",
                    "failed": "åè°±æåå¤±è´¥",
                    "unavailable": "æåä¸å¯ç¨",
                }
                return labels.get(fit.status, fit.status)
        summary = self.state.batch_summary
        if summary is not None and summary.items:
            total = len(summary.items)
            done = sum(1 for item in summary.items if item.fit is not None)
            failed = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "failed")
            warned = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "warn")
            if failed:
                return f"æ¹éå®æ {done}/{total} å«å¤±è´¥"
            if warned:
                return f"æ¹éå®æ {done}/{total} å«è­¦å"
            return f"æ¹éå®æ {done}/{total}"
        return "åå¤å°±ç»ª"
    def _latest_fit_for_spectrum(self, display_name: str) -> FitOutcome | None:
        preferred_key = str(self.template_combo.currentData()) if hasattr(self, "template_combo") else None
        if preferred_key is not None:
            fit = self.state.fits.get((display_name, preferred_key))
            if fit is not None:
                return fit
        for key, fit in reversed(list(self.state.fits.items())):
            if key[0] == display_name:
                return fit
        return None
    def _on_queue_row_changed(self, page_key: str) -> None:
        table = self._queue_tables.get(page_key)
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self.state.spectra):
            return
        self._set_current_index(row)
    def _on_tab_changed(self, _: int) -> None:
        self._refresh_current_tab()
    def _current_interface_id(self) -> str:
        widget = self.stackedWidget.currentWidget()
        return widget.objectName() if widget is not None else ""
    def _refresh_current_tab(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        current_id = self._current_interface_id()
        if current_id == "inspectInterface":
            self._refresh_inspect_view(spectrum, quality)
        elif current_id == "fitInterface":
            self._refresh_fit_view(spectrum)
        elif current_id == "batchInterface":
            self._refresh_batch_view()
    def _refresh_sidebar(self, spectrum: SpectrumData, quality: QualityReport) -> None:
        lines = [
            f"æä»¶: {spectrum.metadata.file_path.name}",
            f"æ ¼å¼: {spectrum.metadata.source_format.upper()} Â· {spectrum.n_points} ç¹",
            f"éé: {spectrum.acquired_label}",
            f"æ°æ®æ£æ¥: {quality.status}",
        ]
        text = "
".join(lines)
        for meta_box in self._queue_meta_boxes.values():
            meta_box.setPlainText(text)
    def _refresh_inspect_view(self, spectrum: SpectrumData, quality: QualityReport) -> None:
        lines = [f"æ£æ¥ç»è®º: {quality.status.upper()}", f"é¢ç¹æ°: {spectrum.n_points}"]
        lines.append("-" * 20)
        lines.extend(quality.summary_lines())
        self.quality_text.setPlainText("
".join(lines))
        headers = ["freq_hz", "z_real_ohm", "z_imag_ohm", "minus_z_imag_ohm", "z_mod_ohm", "phase_deg"]
        rows = [
            [f"{a:.8g}", f"{b:.8g}", f"{c:.8g}", f"{d:.8g}", f"{e:.8g}", f"{f:.8g}"]
            for a, b, c, d, e, f in zip(
                spectrum.freq_hz,
                spectrum.z_real_ohm,
                spectrum.z_imag_ohm,
                spectrum.minus_z_imag_ohm,
                spectrum.z_mod_ohm,
                spectrum.phase_deg,
            )
        ]
        self._fill_table(self.data_table, headers, rows)
        if pg is None:
            return
        pen = pg.mkPen("#0071e3", width=1.6)
        brush = pg.mkBrush(QColor("#5ac8fa"))
        for plot in (self.nyquist_plot, self.bode_mag_plot, self.bode_phase_plot):
            self._reset_plot(plot)
        self.nyquist_plot.plot(spectrum.z_real_ohm, spectrum.minus_z_imag_ohm, pen=pen, symbol="o", symbolSize=5, symbolBrush=brush)
        self.bode_mag_plot.plot(spectrum.freq_hz, spectrum.z_mod_ohm, pen=pen, symbol="o", symbolSize=4, symbolBrush=brush)
        self.bode_phase_plot.plot(spectrum.freq_hz, spectrum.phase_deg, pen=pen, symbol="o", symbolSize=4, symbolBrush=brush)
        self._set_nyquist_axes(self.nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
    def _toggle_quality_detail(self) -> None:
        pass
    def _run_kk_for_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            quality = assess_spectrum_quality(spectrum, run_kk=True)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._warn(f"KK è¿è¡å¤±è´¥ï¼{exc}")
            return
        QApplication.restoreOverrideCursor()
        self.state.qualities[spectrum.display_name] = quality
        self._refresh_sidebar(spectrum, quality)
        self._refresh_inspect_view(spectrum, quality)
        self._update_global_status()
    def _is_zview_template_selected(self) -> bool:
        return str(self.template_combo.currentData()).startswith("zview_")
    def _segment_min_points(self) -> int:
        return 6
    def _format_segment_frequency(self, value: float | None) -> str:
        if value is None or not np.isfinite(value) or value <= 0:
            return ""
        return f"{value:.6g}"
    def _segment_freq_for_index(self, spectrum: SpectrumData, index: int | None) -> float | None:
        if index is None or index < 0 or index >= spectrum.n_points:
            return None
        return float(spectrum.freq_hz[int(index)])
    def _segment_index_for_frequency(self, spectrum: SpectrumData, freq_hz: float | None) -> int | None:
        if freq_hz is None or not np.isfinite(freq_hz) or freq_hz <= 0:
            return None
        freqs = spectrum.freq_hz.astype(float, copy=False)
        safe_freqs = np.clip(freqs, np.finfo(float).tiny, None)
        target = float(np.clip(freq_hz, safe_freqs.min(), safe_freqs.max()))
        nearest = int(np.argmin(np.abs(np.log10(safe_freqs) - np.log10(target))))
        return nearest
    def _segment_index_bounds(
        self,
        spectrum: SpectrumData,
        resolved_mode: str,
        base_detection: SegmentDetection,
        split1: int | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int] | None]:
        min_points = self._segment_min_points()
        n_points = spectrum.n_points
        if resolved_mode == "single":
            lo = min_points - 1
            hi = max(lo, n_points - min_points)
            return (lo, hi), None
        peak2 = base_detection.peak_indices[1] if len(base_detection.peak_indices) > 1 else None
        split2_hi = max(min_points - 1, n_points - min_points)
        split1_lo = min_points - 1
        split1_hi = max(split1_lo, split2_hi - (min_points - 1))
        if peak2 is not None:
            split1_hi = max(split1_lo, min(split1_hi, peak2 - 1))
        split1_bounds = (split1_lo, split1_hi)
        effective_split1 = split1 if split1 is not None else base_detection.split_indices[0]
        effective_split1 = int(np.clip(effective_split1, split1_bounds[0], split1_bounds[1]))
        split2_lo = effective_split1 + (min_points - 1)
        if peak2 is not None:
            split2_lo = max(split2_lo, peak2)
        split2_lo = min(split2_lo, split2_hi)
        split2_bounds = (split2_lo, split2_hi)
        return split1_bounds, split2_bounds
    def _sanitize_segment_indices(
        self,
        spectrum: SpectrumData,
        requested_mode: str,
        split1: int | None,
        split2: int | None,
        base_detection: SegmentDetection | None = None,
    ) -> tuple[int | None, int | None, SegmentDetection]:
        detection = base_detection or detect_segments(spectrum, mode=requested_mode)
        resolved_mode = detection.resolved_mode if requested_mode == "auto" else requested_mode
        if resolved_mode == "single":
            bounds, _ = self._segment_index_bounds(spectrum, "single", detection)
            candidate = detection.split_indices[0] if split1 is None else split1
            split1_value = int(np.clip(candidate, bounds[0], bounds[1]))
            return split1_value, None, detection
        split1_bounds, split2_bounds = self._segment_index_bounds(spectrum, "double", detection, split1)
        default_split1 = detection.split_indices[0] if detection.split_indices else split1_bounds[0]
        split1_value = int(np.clip(default_split1 if split1 is None else split1, split1_bounds[0], split1_bounds[1]))
        split1_bounds, split2_bounds = self._segment_index_bounds(spectrum, "double", detection, split1_value)
        default_split2 = detection.split_indices[1] if len(detection.split_indices) > 1 else split2_bounds[0]
        split2_value = int(np.clip(default_split2 if split2 is None else split2, split2_bounds[0], split2_bounds[1]))
        return split1_value, split2_value, detection
    def _set_split2_controls_visible(self, visible: bool) -> None:
        for widget in getattr(self, "segment_split2_widgets", []):
            widget.setVisible(visible)
        if hasattr(self, "segment_arc2_label"):
            self.segment_arc2_label.setVisible(visible)
    def _sync_segment_controls_from_detection(self, spectrum: SpectrumData, detection: SegmentDetection) -> None:
        split_indices = list(detection.split_indices)
        split1 = split_indices[0] if split_indices else None
        split2 = split_indices[1] if len(split_indices) > 1 else None
        show_split2 = detection.resolved_mode == "double"
        self._set_split2_controls_visible(show_split2)
        self._segment_control_sync = True
        try:
            with QSignalBlocker(self.segment_split1_freq_edit):
                self.segment_split1_freq_edit.setText(self._format_segment_frequency(self._segment_freq_for_index(spectrum, split1)))
            with QSignalBlocker(self.segment_split1_spin):
                self.segment_split1_spin.setText("" if split1 is None else str(split1))
            with QSignalBlocker(self.segment_split2_freq_edit):
                self.segment_split2_freq_edit.setText(self._format_segment_frequency(self._segment_freq_for_index(spectrum, split2)))
            with QSignalBlocker(self.segment_split2_spin):
                self.segment_split2_spin.setText("" if split2 is None else str(split2))
        finally:
            self._segment_control_sync = False
        self.segment_split1_freq_label.setText("split Frequency [Hz]" if not show_split2 else "split1 Frequency [Hz]")
        if hasattr(self, "segment_split2_freq_label"):
            self.segment_split2_freq_label.setText("split2 Frequency [Hz]")
    def _refresh_segment_controls_state(self, spectrum: SpectrumData | None = None) -> None:
        enabled = self._is_zview_template_selected()
        for widget in (
            self.segment_mode_combo,
            self.segment_split1_freq_edit,
            self.segment_split2_freq_edit,
            self.segment_split1_spin,
            self.segment_split2_spin,
            self.segment_detect_btn,
        ):
            widget.setEnabled(enabled)
        if enabled and spectrum is not None:
            detection = detect_segments(spectrum, mode=str(self.segment_mode_combo.currentData() or "auto"))
            show_split2 = detection.resolved_mode == "double"
            self._set_split2_controls_visible(show_split2)
            freq_min = float(np.min(spectrum.freq_hz))
            freq_max = float(np.max(spectrum.freq_hz))
            self.segment_split1_freq_edit.setPlaceholderText(f"{freq_min:.3g} - {freq_max:.3g}")
            self.segment_split2_freq_edit.setPlaceholderText(f"{freq_min:.3g} - {freq_max:.3g}")
            self.segment_split2_freq_edit.setEnabled(enabled and show_split2)
        else:
            self._set_split2_controls_visible(False)
    def _prime_segment_controls(self, spectrum: SpectrumData) -> None:
        self._refresh_segment_controls_state(spectrum)
        if not self._is_zview_template_selected():
            return
        detection = detect_segments(spectrum, mode=str(self.segment_mode_combo.currentData() or "auto"))
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._update_split_info_labels(spectrum, detection)
    def _detect_segments_for_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        self._prime_segment_controls(spectrum)
        detection = self._current_segment_hint(spectrum)
        if detection is not None:
            self.state.segment_hints[spectrum.display_name] = detection
        self._refresh_segment_preview_only(spectrum)
    def _current_segment_hint(self, spectrum: SpectrumData) -> SegmentDetection | None:
        if not self._is_zview_template_selected():
            return None
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        stored_hint = self.state.segment_hints.get(spectrum.display_name)
        if stored_hint is not None and stored_hint.requested_mode == requested_mode:
            return stored_hint
        base_detection = detect_segments(spectrum, mode=requested_mode)
        try:
            split1_freq = float(self.segment_split1_freq_edit.text()) if self.segment_split1_freq_edit.text() else None
        except ValueError:
            split1_freq = None
        try:
            split2_freq = float(self.segment_split2_freq_edit.text()) if self.segment_split2_freq_edit.text() else None
        except ValueError:
            split2_freq = None
        split1 = self._segment_index_for_frequency(spectrum, split1_freq)
        split2 = self._segment_index_for_frequency(spectrum, split2_freq)
        split1, split2, _ = self._sanitize_segment_indices(spectrum, requested_mode, split1, split2, base_detection)
        return detect_segments(
            spectrum,
            mode=requested_mode,
            manual_split1=split1,
            manual_split2=split2,
        )
    def _on_segment_frequency_edited(self, split_role: str) -> None:
        if self._segment_control_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or not self._is_zview_template_selected():
            return
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        base_detection = detect_segments(spectrum, mode=requested_mode)
        edit1 = self.segment_split1_freq_edit.text().strip()
        edit2 = self.segment_split2_freq_edit.text().strip()
        try:
            split1 = self._segment_index_for_frequency(spectrum, float(edit1)) if edit1 else None
        except ValueError:
            split1 = None
        try:
            split2 = self._segment_index_for_frequency(spectrum, float(edit2)) if edit2 else None
        except ValueError:
            split2 = None
        split1, split2, _ = self._sanitize_segment_indices(spectrum, requested_mode, split1, split2, base_detection)
        detection = detect_segments(spectrum, mode=requested_mode, manual_split1=split1, manual_split2=split2)
        self.state.segment_hints[spectrum.display_name] = detection
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._refresh_segment_preview_only(spectrum, detection)
    def _refresh_fit_for_current_selection(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        self._refresh_segment_controls_state(spectrum)
        if spectrum is not None and self._is_zview_template_selected() and not self._segment_drag_active:
            self._prime_segment_controls(spectrum)
        else:
            self._update_split_info_labels(spectrum)
        if spectrum is not None and self._current_interface_id() == "fitInterface":
            self._refresh_fit_view(spectrum)
        self._update_global_status()
    def _update_split_info_labels(
        self,
        spectrum: SpectrumData | None = None,
        detection: SegmentDetection | None = None,
    ) -> None:
        if spectrum is None or not self._is_zview_template_selected():
            self.segment_mode_status.setText("Segment mode: unavailable")
            self.segment_peak_info_label.setText("Peaks: -")
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
            self._set_split2_controls_visible(False)
            return
        detection = detection or self._current_segment_hint(spectrum)
        if detection is None:
            self.segment_mode_status.setText("Segment mode: unavailable")
            self.segment_peak_info_label.setText("Peaks: -")
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
            self._set_split2_controls_visible(False)
            return
        self._set_split2_controls_visible(detection.resolved_mode == "double")
        peak_bits = [
            f"p{i + 1}=idx {peak} @ {self._format_segment_frequency(self._segment_freq_for_index(spectrum, peak))} Hz"
            for i, peak in enumerate(detection.peak_indices)
        ]
        self.segment_mode_status.setText(
            f"Segment mode: {detection.requested_mode} -> {detection.resolved_mode}"
        )
        self.segment_peak_info_label.setText("Peaks: " + (", ".join(peak_bits) if peak_bits else "-"))
        def region_text(name: str, start: int, stop: int) -> str:
            start_freq = self._format_segment_frequency(self._segment_freq_for_index(spectrum, start))
            stop_freq = self._format_segment_frequency(self._segment_freq_for_index(spectrum, stop))
            points = max(stop - start + 1, 0)
            return f"{name}: idx {start}-{stop} | {start_freq} -> {stop_freq} Hz | {points} pt"
        split_indices = list(detection.split_indices)
        if detection.resolved_mode == "double" and len(split_indices) >= 2:
            split1, split2 = split_indices[:2]
            self.segment_arc1_label.setText(region_text("Arc1", 0, split1))
            self.segment_arc2_label.setText(region_text("Arc2", split1, split2))
            self.segment_tail_label.setText(region_text("Tail", split2, spectrum.n_points - 1))
        elif split_indices:
            split1 = split_indices[0]
            self.segment_arc1_label.setText(region_text("Arc1", 0, split1))
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText(region_text("Tail", split1, spectrum.n_points - 1))
        else:
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
    def _clear_segment_overlay(self, *, keep_handles: bool = False) -> None:
        if pg is None or not hasattr(self, "fit_nyquist_plot"):
            self._segment_overlay_items = []
            if not keep_handles:
                self._segment_handles = {}
            return
        seen: set[int] = set()
        items = list(self._segment_overlay_items)
        if not keep_handles:
            items.extend(self._segment_handles.values())
        for item in items:
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            try:
                self.fit_nyquist_plot.removeItem(item)
            except Exception:
                pass
        self._segment_overlay_items = []
        if not keep_handles:
            self._segment_handles = {}
    def _render_segment_overlay(self, spectrum: SpectrumData, segment_hint: SegmentDetection | None) -> None:
        self._clear_segment_overlay(keep_handles=True)
        if pg is None or segment_hint is None:
            self._clear_segment_overlay()
            return
        split_indices = list(segment_hint.split_indices)
        sections: list[tuple[int, int, str]] = []
        handle_specs: list[tuple[str, int, str]] = []
        if segment_hint.resolved_mode == "double" and len(split_indices) >= 2:
            split1, split2 = split_indices[:2]
            sections = [
                (0, split1, "#ff9500"),
                (split1, split2, "#af52de"),
                (split2, spectrum.n_points - 1, "#ff3b30"),
            ]
            handle_specs = [("split1", split1, "#ffd60a"), ("split2", split2, "#ff453a")]
        elif split_indices:
            split1 = split_indices[0]
            sections = [(0, split1, "#ff9500"), (split1, spectrum.n_points - 1, "#ff3b30")]
            handle_specs = [("split1", split1, "#ffd60a")]
        for start, stop, color in sections:
            xs = spectrum.z_real_ohm[start : stop + 1]
            ys = spectrum.minus_z_imag_ohm[start : stop + 1]
            item = self.fit_nyquist_plot.plot(xs, ys, pen=pg.mkPen(color=color, width=2, style=Qt.DashLine))
            self._segment_overlay_items.append(item)
        active_roles = {role for role, _, _ in handle_specs}
        for role, handle in list(self._segment_handles.items()):
            if role in active_roles:
                continue
            try:
                self.fit_nyquist_plot.removeItem(handle)
            except Exception:
                pass
            self._segment_handles.pop(role, None)
        for role, index, color in handle_specs:
            x = float(spectrum.z_real_ohm[index])
            y = float(spectrum.minus_z_imag_ohm[index])
            handle = self._segment_handles.get(role)
            if handle is None:
                handle = pg.TargetItem(
                    pos=(x, y),
                    size=14,
                    symbol="o",
                    pen=pg.mkPen(color=color, width=1.8),
                    hoverPen=pg.mkPen(color="#ffffff", width=2.2),
                    brush=pg.mkBrush(QColor(color)),
                    hoverBrush=pg.mkBrush(QColor("#ffffff")),
                    movable=True,
                )
                handle.sigPositionChanged.connect(lambda _, split_role=role: self._on_segment_handle_moved(split_role))
                handle.sigPositionChangeFinished.connect(lambda _, split_role=role: self._on_segment_handle_released(split_role))
                self.fit_nyquist_plot.addItem(handle)
                self._segment_handles[role] = handle
            self._segment_handle_sync = True
            try:
                handle.setPos((x, y))
            finally:
                self._segment_handle_sync = False
    def _refresh_segment_preview_only(
        self,
        spectrum: SpectrumData,
        detection: SegmentDetection | None = None,
    ) -> None:
        detection = detection or self._current_segment_hint(spectrum)
        if detection is None:
            self._clear_segment_overlay()
            return
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._update_split_info_labels(spectrum, detection)
        if pg is not None:
            self._render_segment_overlay(spectrum, detection)
    def _ensure_fit_plot_click_handler(self) -> None:
        if pg is None or self._fit_plot_click_connected:
            return
        self.fit_nyquist_plot.scene().sigMouseClicked.connect(self._on_fit_plot_scene_clicked)
        self._fit_plot_click_connected = True
    def _move_segment_to_index(
        self,
        spectrum: SpectrumData,
        split_role: str,
        candidate_index: int,
        *,
        commit: bool,
    ) -> None:
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        base_detection = detect_segments(spectrum, mode=requested_mode)
        current_hint = self._current_segment_hint(spectrum) or base_detection
        current_split1 = current_hint.split_indices[0] if current_hint.split_indices else None
        current_split2 = current_hint.split_indices[1] if len(current_hint.split_indices) > 1 else None
        if split_role == "split1":
            current_split1 = candidate_index
        else:
            current_split2 = candidate_index
        split1, split2, _ = self._sanitize_segment_indices(
            spectrum,
            requested_mode,
            current_split1,
            current_split2,
            base_detection,
        )
        detection = detect_segments(spectrum, mode=requested_mode, manual_split1=split1, manual_split2=split2)
        self._segment_drag_active = not commit
        self._refresh_segment_preview_only(spectrum, detection)
        self._segment_drag_active = False
        if commit:
            self._fit_current()
    def _nearest_segment_index_from_handle(self, spectrum: SpectrumData, split_role: str) -> int | None:
        handle = self._segment_handles.get(split_role)
        if handle is None:
            return None
        point = handle.pos()
        x_pos = float(point.x())
        y_pos = float(point.y())
        dists = (spectrum.z_real_ohm - x_pos) ** 2 + (spectrum.minus_z_imag_ohm - y_pos) ** 2
        return int(np.argmin(dists))
    def _on_segment_handle_moved(self, split_role: str) -> None:
        if self._segment_handle_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        nearest_index = self._nearest_segment_index_from_handle(spectrum, split_role)
        if nearest_index is None:
            return
        self._segment_handle_sync = True
        try:
            self._move_segment_to_index(spectrum, split_role, nearest_index, commit=False)
        finally:
            self._segment_handle_sync = False
    def _on_segment_handle_released(self, split_role: str) -> None:
        if self._segment_handle_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        nearest_index = self._nearest_segment_index_from_handle(spectrum, split_role)
        if nearest_index is None:
            return
        self._move_segment_to_index(spectrum, split_role, nearest_index, commit=True)
    def _refresh_fit_view(self, spectrum: SpectrumData) -> None:
        key = str(self.template_combo.currentData())
        fit = self.state.fits.get((spectrum.display_name, key))
        if fit is None:
            for stored_key, stored_fit in self.state.fits.items():
                if stored_key[0] == spectrum.display_name:
                    fit = stored_fit
                    break
        segment_hint = self._current_segment_hint(spectrum)
        if segment_hint is not None:
            self._sync_segment_controls_from_detection(spectrum, segment_hint)
            self._update_split_info_labels(spectrum, segment_hint)
        lines: list[str] = []
        detail_lines: list[str] = []
        if segment_hint is not None:
            detail_lines.extend(
                [
                    f"ååºæ¨¡å¼: {segment_hint.requested_mode} -> {segment_hint.resolved_mode}",
                    f"å³°ä½ç´¢å¼: {', '.join(str(i) for i in segment_hint.peak_indices) or '-'}",
                    f"åçç´¢å¼: {', '.join(str(i) for i in segment_hint.split_indices) or '-'}",
                    "",
                ]
            )
        if fit is None:
            lines.extend(["å½åæ¨¡æ¿å°æªæ§è¡æåã", "", "ä¸»åæ°", "  ç­å¾è¿è¡æå"])
            self.fit_text.setPlainText("
".join(lines))
            if pg is not None:
                self._reset_plot(self.fit_nyquist_plot, True)
                self.fit_nyquist_plot.plot(
                    spectrum.z_real_ohm,
                    spectrum.minus_z_imag_ohm,
                    pen=pg.mkPen("#0071e3", width=1.6),
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(QColor("#5ac8fa")),
                )
                self._ensure_fit_plot_click_handler()
                if segment_hint is not None:
                    self._refresh_segment_preview_only(spectrum, segment_hint)
                else:
                    self._clear_segment_overlay()
                self._set_nyquist_axes(self.fit_nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
            return
        lines.extend([f"æ¨¡å: {fit.model_label}", f"ç¶æ: {fit.status.upper()}", ""])
        if fit.preprocess_actions:
            lines.append("æ°æ®é¢å¤ç:")
            for action in fit.preprocess_actions:
                lines.append(f"  â {action}")
            lines.append("")
        if fit.fallback_from:
            lines.append(f"â  éçº§éç¥: åæ¨¡åå¨å±æåæªæ¶æï¼å½åç»ææ¥èªåå R(QR) éçº§æå")
            lines.append("")
        lines.append("ä¸»åæ°")
        for name in ("Rs", "Rsei", "Rct"):
            if name in fit.parameters:
                lines.append(f"  {name} = {fit.parameters[name]:.8g}")
        error_lines = self._fit_error_lines(fit)
        if error_lines:
            lines.extend(["", "ä¸»è¯¯å·®ç"])
            lines.extend(error_lines)
        diag_start = fit.message.find("[è¯æ­:")
        if diag_start >= 0:
            diag_text = fit.message[diag_start:]
            plain_message = fit.message[:diag_start].strip().rstrip(";").strip()
            lines.extend(["", "å¤å®è¯´æ", f"  {plain_message.split(';', 1)[0].strip()}"])
            lines.extend(["", "è¯æ­åæ", f"  {diag_text}"])
        else:
            lines.extend(["", "å¤å®è¯´æ", f"  {fit.message.split(';', 1)[0].strip()}"])
        detail_lines.extend(
            [
                "",
                f"å®æ´è¯´æ: {fit.message}",
                "",
                "æ¬¡çº§åæ°",
            ]
        )
        for name in ("Rsei_global", "Rct_global", "Wo_R", "Wo_T", "Wo_P", "CPE_T", "CPE_P", "Q1", "n1", "Q2", "n2"):
            if name in fit.parameters:
                detail_lines.append(f"  {name} = {fit.parameters[name]:.8g}")
        detail_lines.extend(
            [
                "",
                "ç»è®¡",
                f"  rss = {fit.statistics.get('rss', float('nan')):.8g}",
                f"  aic = {fit.statistics.get('aic', float('nan')):.8g}",
                f"  bic = {fit.statistics.get('bic', float('nan')):.8g}",
                f"  chi2_reduced = {fit.statistics.get('chi2_reduced', float('nan')):.8g}",
            ]
        )
        self.fit_text.setPlainText("
".join(lines + ["", "é«çº§åæ°"] + detail_lines))
        if pg is None:
            self._update_global_status()
            return
        self._reset_plot(self.fit_nyquist_plot, True)
        self._ensure_legend(self.fit_nyquist_plot)
        self.fit_nyquist_plot.plot(
            spectrum.z_real_ohm,
            spectrum.minus_z_imag_ohm,
            pen=pg.mkPen("#0071e3", width=1.6),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(QColor("#5ac8fa")),
            name="å®æµ",
        )
        self._ensure_fit_plot_click_handler()
        if segment_hint is not None:
            self._refresh_segment_preview_only(spectrum, segment_hint)
        else:
            self._clear_segment_overlay()
        if fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
            self.fit_nyquist_plot.plot(
                fit.predicted_real_ohm,
                -fit.predicted_imag_ohm,
                pen=pg.mkPen("#34c759", width=2.1),
                name="æå",
            )
            self._set_nyquist_axes(
                self.fit_nyquist_plot,
                np.concatenate([spectrum.z_real_ohm, fit.predicted_real_ohm]),
                np.concatenate([spectrum.minus_z_imag_ohm, -fit.predicted_imag_ohm]),
            )
        else:
            self._set_nyquist_axes(self.fit_nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
        self._update_global_status()
    def _on_fit_plot_scene_clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton:
            return
        if not self._is_zview_template_selected():
            return
        segment_hint = self._current_segment_hint(self._current_spectrum(silent=True)) if self._current_spectrum(silent=True) is not None else None
        pos = self.fit_nyquist_plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x_clicked = pos.x()
        y_clicked = pos.y()
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        dists = (spectrum.z_real_ohm - x_clicked)**2 + (spectrum.minus_z_imag_ohm - y_clicked)**2
        nearest_idx = int(np.argmin(dists))
        if segment_hint is None or segment_hint.resolved_mode == "single" or len(segment_hint.split_indices) < 2:
            self._move_segment_to_index(spectrum, "split1", nearest_idx, commit=True)
            return
        idx1, idx2 = segment_hint.split_indices[:2]
        split_role = "split1" if abs(nearest_idx - idx1) <= abs(nearest_idx - idx2) else "split2"
        self._move_segment_to_index(spectrum, split_role, nearest_idx, commit=True)
    def _toggle_fit_detail(self) -> None:
        visible = not self.fit_detail_text.isVisible()
        self.fit_detail_text.setVisible(visible)
        self.fit_detail_btn.setText("æ¶èµ·è¯¦æ" if visible else "å±å¼è¯¦æ")
    def _fit_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        template_key = str(self.template_combo.currentData())
        segment_hint = self._current_segment_hint(spectrum)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fit = fit_spectrum(spectrum, template_key, segment_hint=segment_hint)
        finally:
            QApplication.restoreOverrideCursor()
        self.state.fits[(spectrum.display_name, template_key)] = fit
        self._sync_fit_to_batch(spectrum, fit)
        self._refresh_fit_view(spectrum)
    def _fit_batch(self) -> None:
        if not self.state.spectra:
            self._warn("å°æªå¯¼å¥ä»»ä½è°±å¾ã")
            return
        if self.state.batch_busy:
            self._warn("æ¹éæåæ­£å¨è¿è¡ï¼è¯·ç­å¾å½åä»»å¡ç»æã")
            return
        self.state.batch_summary = None
        self._refresh_batch_view()
        self.batch_progress.setValue(0)
        self.batch_progress_label.setText(f"åå¤å¤ç 0 / {len(self.state.spectra)}")
        self._set_batch_busy(True)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchFitWorker(list(self.state.spectra))
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()
    def _refresh_batch_view(self) -> None:
        summary = self.state.batch_summary
        if summary is None:
            self.batch_text.setPlainText("å°æªæ§è¡æ¹éæåã" if not self.state.batch_busy else "æ¹éæåæ­£å¨åå°è¿è¡ã")
            self.batch_table.clearContents()
            self.batch_table.setRowCount(0)
            self.batch_table.setColumnCount(0)
            self._batch_plot_cache = {}
            if pg is not None:
                self._reset_plot(self.batch_plot, True)
            return
        warn_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "warn")
        high_error_count = sum(1 for item in summary.items if item.fit is not None and self._fit_has_high_error(item.fit))
        failed_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "failed")
        ok_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "ok")
        fallback_count = sum(1 for item in summary.items if item.fit is not None and item.fit.fallback_from is not None)
        lines = [
            "æ¨¡åç­ç¥: èªå¨è¯å«ååå / ååå",
            f"è°±å¾æ°é: {len(summary.items)}",
            f"â æå: {ok_count}    â  è­¦å: {warn_count}    â å¤±è´¥: {failed_count}",
            f"ð éçº§æå: {fallback_count}    ð é«è¯¯å·®: {high_error_count}",
        ]
        self.batch_text.setPlainText("
".join(lines))
        names: list[str] = []
        for item in summary.items:
            if item.fit is None:
                continue
            for name in item.fit.parameters:
                if name not in names:
                    names.append(name)
        essential_names = [name for name in ("Rs", "Rsei", "Rct") if name in names]
        headers = [
            "åºå·", "æä»¶", "æ¨¡å", "å¤å®è¯´æ", "æåç¶æ", "éçº§", "é¢å¤ç", "è¯¯å·®æç¤º",
            "Rs_error_pct", "Rsei_error_pct", "Rct_error_pct", *essential_names,
        ]
        rows: list[list[str]] = []
        for index, item in enumerate(summary.items, start=1):
            fit = item.fit
            stats = fit.statistics if fit is not None else {}
            msg = ""
            if fit and fit.message:
                diag_pos = fit.message.find("[è¯æ­:")
                msg = fit.message[:diag_pos].strip().rstrip(";").strip() if diag_pos >= 0 else fit.message
                msg = msg.split(";", 1)[0]
            fallback_label = "R(QR)éçº§" if fit and fit.fallback_from else ""
            preprocess_label = "; ".join(fit.preprocess_actions) if fit and fit.preprocess_actions else ""
            row = [
                str(index), item.spectrum.display_name, fit.model_label if fit else "-",
                msg,
                fit.status if fit else "not_run", fallback_label, preprocess_label,
                self._fit_error_summary(fit) if fit else "",
                self._format_stat(stats.get("Rs_stderr_pct")),
                self._format_stat(stats.get("Rsei_stderr_pct")),
                self._format_stat(stats.get("Rct_stderr_pct")),
            ]
            row.extend("" if fit is None else f"{fit.parameters.get(name, float('nan')):.8g}" for name in essential_names)
            rows.append(row)
        self._fill_table(self.batch_table, headers, rows)
        for row_index, item in enumerate(summary.items):
            fit = item.fit
            if fit is None:
                continue
            if fit.status == "failed" or self._fit_has_high_error(fit):
                self._color_batch_row(row_index, QColor("#fff1f0"), QColor("#b42318"))
            elif fit.status == "warn":
                self._color_batch_row(row_index, QColor("#fff7e6"), QColor("#b54708"))
        self._rebuild_batch_plot_cache(summary)
        self._refresh_batch_plot()
        self._update_global_status()
    def _rebuild_batch_plot_cache(self, summary: BatchSummary) -> None:
        x = np.arange(1, len(summary.items) + 1, dtype=float)
        cache: dict[str, dict[str, object]] = {}
        for name, color, error_key in (
            ("Rs", "#0071e3", "Rs_stderr_pct"),
            ("Rsei", "#ff9500", "Rsei_stderr_pct"),
            ("Rct", "#34c759", "Rct_stderr_pct"),
        ):
            values = np.asarray([np.nan if item.fit is None else item.fit.parameters.get(name, np.nan) for item in summary.items], dtype=float)
            ok_x: list[float] = []
            ok_y: list[float] = []
            bad_x: list[float] = []
            bad_y: list[float] = []
            for idx, item in enumerate(summary.items, start=1):
                fit = item.fit
                if fit is None:
                    continue
                value = fit.parameters.get(name)
                if value is None or not np.isfinite(float(value)):
                    continue
                err = fit.statistics.get(error_key)
                if err is not None and (not np.isfinite(float(err)) or float(err) > 20.0):
                    bad_x.append(float(idx))
                    bad_y.append(float(value))
                else:
                    ok_x.append(float(idx))
                    ok_y.append(float(value))
            cache[name] = {
                "x": x,
                "values": values,
                "ok_x": ok_x,
                "ok_y": ok_y,
                "bad_x": bad_x,
                "bad_y": bad_y,
                "color": color,
            }
        self._batch_plot_cache = cache
    def _refresh_batch_plot(self) -> None:
        if pg is None:
            return
        if self.state.batch_summary is None:
            self._reset_plot(self.batch_plot, True)
            return
        self.batch_plot.setUpdatesEnabled(False)
        try:
            self._reset_plot(self.batch_plot, True)
            self._ensure_legend(self.batch_plot)
            enabled = {
                "Rs": self.trend_rs_check.isChecked(),
                "Rsei": self.trend_rsei_check.isChecked(),
                "Rct": self.trend_rct_check.isChecked(),
            }
            for name in ("Rs", "Rsei", "Rct"):
                if not enabled[name]:
                    continue
                series = self._batch_plot_cache.get(name)
                if not series:
                    continue
                values = np.asarray(series["values"], dtype=float)
                if not np.any(np.isfinite(values)):
                    continue
                color = str(series["color"])
                self.batch_plot.plot(np.asarray(series["x"], dtype=float), values, pen=pg.mkPen(color=color, width=2), name=name)
                ok_x = list(series["ok_x"])
                ok_y = list(series["ok_y"])
                bad_x = list(series["bad_x"])
                bad_y = list(series["bad_y"])
                if ok_x:
                    scatter_ok = pg.ScatterPlotItem(ok_x, ok_y, symbol="o", size=5, brush=pg.mkBrush(QColor(color)), pen=pg.mkPen(color))
                    scatter_ok.sigClicked.connect(self._on_trend_point_clicked)
                    self.batch_plot.addItem(scatter_ok)
                if bad_x:
                    scatter_bad = pg.ScatterPlotItem(bad_x, bad_y, symbol="o", size=7, brush=pg.mkBrush(255, 255, 255, 0), pen=pg.mkPen(color, width=2))
                    scatter_bad.sigClicked.connect(self._on_trend_point_clicked)
                    self.batch_plot.addItem(scatter_bad)
        finally:
            self.batch_plot.setUpdatesEnabled(True)
    def _on_trend_point_clicked(self, scatter_item, points, ev=None) -> None:
        if not points:
            return
        try:
            clicked_x = int(round(points[0].pos().x()))
        except Exception:
            return
        summary = self.state.batch_summary
        if summary is None:
            return
        row_index = clicked_x - 1
        if 0 <= row_index < len(summary.items):
            item = summary.items[row_index]
            self._set_current_index(row_index, refresh_page=False)
            self.batch_table.selectRow(row_index)
            self.batch_table.scrollToItem(self.batch_table.item(row_index, 0))
            self.batch_progress_label.setText(
                f"å·²éä¸­ç¬¬ {clicked_x} æ¡ï¼{item.spectrum.display_name}"
            )
    def _on_batch_table_selection_changed(self) -> None:
        if self.state.batch_summary is None:
            return
        row = self.batch_table.currentRow()
        if row < 0 or row >= len(self.state.spectra):
            return
        self._set_current_index(row, refresh_page=False)
    def _browse_matlab_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "éæ© matlab.exe", self.matlab_exe_edit.text() or str(Path.cwd()), "Executable (*.exe)")
        if path:
            self.matlab_exe_edit.setText(path)
    def _browse_drttools_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "éæ© DRTtools ç®å½", self.drttools_dir_edit.text() or str(Path.cwd()))
        if path:
            self.drttools_dir_edit.setText(path)
    def _current_matlab_drt_config(self) -> MatlabDrtConfig:
        return MatlabDrtConfig(
            matlab_exe=self.matlab_exe_edit.text().strip(),
            drttools_dir=self.drttools_dir_edit.text().strip(),
            method_tag=str(self.matlab_method_combo.currentData()),
            drt_type=int(self.matlab_drt_type_combo.currentData()),
            lambda_value=float(self.matlab_lambda_edit.text().strip()),
            coeff_value=float(self.matlab_coeff_edit.text().strip()),
            inductance_mode=int(self.matlab_inductance_combo.currentData()),
        )
    def _set_drt_busy(self, busy: bool) -> None:
        self.state.drt_busy = busy
        for button in (self.run_drt_current_btn, self.run_drt_batch_btn):
            button.setEnabled(not busy)
        if busy:
            self.fit_metric["value"].setText("DRT è¿è¡ä¸­")
        else:
            self._update_global_status()
    def _set_batch_busy(self, busy: bool) -> None:
        self.state.batch_busy = busy
        for button in (self.fit_batch_btn, self.export_batch_btn):
            button.setEnabled(not busy)
        if busy:
            self.fit_metric["value"].setText("æ¹éæåä¸­")
        else:
            self._update_global_status()
    def _on_batch_progress(self, index: int, total: int, display_name: str) -> None:
        percent = int(index * 100 / max(total, 1))
        self.batch_progress.setValue(percent)
        self.batch_progress_label.setText(f"æ­£å¨å¤ç {index} / {total}ï¼{display_name}")
    def _on_batch_finished(self, summary: BatchSummary | None, error: Exception | None) -> None:
        self._set_batch_busy(False)
        if error is not None:
            self.batch_progress.setValue(0)
            self.batch_progress_label.setText("æ¹éæåå¤±è´¥")
            self._warn(f"æ¹éæåå¤±è´¥ï¼{error}")
        elif summary is not None:
            self.state.batch_summary = summary
            for item in summary.items:
                self.state.qualities[item.spectrum.display_name] = item.quality
                if item.fit is not None:
                    self.state.fits[(item.spectrum.display_name, item.fit.model_key)] = item.fit
            self.batch_progress.setValue(100)
            self.batch_progress_label.setText(f"æ¹éæåå®æï¼{len(summary.items)} / {len(summary.items)}")
            self._refresh_batch_view()
        self._update_global_status()
        self._batch_worker = None
        self._batch_thread = None
    def _sync_fit_to_batch(self, spectrum: SpectrumData, fit: FitOutcome) -> None:
        summary = self.state.batch_summary
        if summary is None:
            return
        for item in summary.items:
            if item.spectrum.display_name == spectrum.display_name:
                item.fit = fit
                break
        if self._current_interface_id() == "batchInterface":
            self._refresh_batch_view()
        else:
            self._update_global_status()
    def _start_matlab_drt(self, spectra: list[SpectrumData], export_dir: Path) -> None:
        if self.state.drt_busy:
            self._warn("MATLAB DRT æ­£å¨è¿è¡ï¼è¯·ç­å¾å½åä»»å¡ç»æã")
            return
        try:
            config = self._current_matlab_drt_config()
        except ValueError as exc:
            self._warn(f"MATLAB DRT åæ°æ æï¼{exc}")
            return
        self._last_drt_spectra = list(spectra)
        self.matlab_status.setPlainText(f"ä»»å¡å·²æäº¤
è¾å¥è°±å¾æ°: {len(spectra)}
è¾åºç®å½: {export_dir / 'results'}")
        self.matlab_log.setPlainText("MATLAB DRT å·²å¯å¨ï¼æ­£å¨åå°è¿è¡ï¼è¯·ç¨åâ¦â¦")
        self._set_drt_busy(True)
        self._matlab_thread = QThread(self)
        self._matlab_worker = MatlabDrtWorker(config, spectra, export_dir)
        self._matlab_worker.moveToThread(self._matlab_thread)
        self._matlab_thread.started.connect(self._matlab_worker.run)
        self._matlab_worker.finished.connect(self._on_matlab_drt_finished)
        self._matlab_worker.finished.connect(self._matlab_thread.quit)
        self._matlab_worker.finished.connect(self._matlab_worker.deleteLater)
        self._matlab_thread.finished.connect(self._matlab_thread.deleteLater)
        self._matlab_thread.start()
    def _on_matlab_drt_finished(self, result: MatlabDrtResult | None, error: Exception | None) -> None:
        self._set_drt_busy(False)
        if error is not None:
            self.fit_metric["value"].setText("DRT å¤±è´¥")
            self.matlab_log.setPlainText(f"MATLAB DRT è¿è¡å¤±è´¥ï¼
{error}")
            self._warn(f"MATLAB DRT è¿è¡å¤±è´¥ï¼{error}")
        else:
            self._show_matlab_drt_result(result)
        self._matlab_worker = None
        self._matlab_thread = None
    def _run_matlab_drt_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "éæ© MATLAB DRT è¾åºç®å½", str(Path.cwd()))
        if folder:
            self._start_matlab_drt([spectrum], Path(folder) / f"{spectrum.metadata.file_path.stem}_matlab_drt")
    def _run_matlab_drt_batch(self) -> None:
        if not self.state.spectra:
            self._warn("å°æªå¯¼å¥ä»»ä½è°±å¾ã")
            return
        folder = QFileDialog.getExistingDirectory(self, "éæ© MATLAB DRT æ¹éè¾åºç®å½", str(Path.cwd()))
        if folder:
            self._start_matlab_drt(self.state.spectra, Path(folder) / "matlab_drt_batch")
    def _show_matlab_drt_result(self, result: MatlabDrtResult | None) -> None:
        if result is None:
            return
        self._last_matlab_result = result
        try:
            from eismaster.exporters import write_drt_only_export
            write_drt_only_export(result.output_dir / "drt_matrix.csv", self._last_drt_spectra, result.output_dir, fmt="csv")
        except Exception:
            pass
        lines = [
            f"å½ä»¤ï¼{' '.join(result.command)}",
            "",
        ]
        self.matlab_status.setPlainText(
            "
".join(
                [
                    f"è¿åç ï¼{result.returncode}",
                    f"è¾åºç®å½ï¼{result.output_dir}",
                    f"ä¸´æ¶è¾å¥ç®å½ï¼{result.staging_dir}",
                    f"ç»ææä»¶æ°ï¼{len(result.output_files)}",
                    "ç»ææä»¶ï¼",
                    *[f"  {path.name}" for path in result.output_files[:12]],
                ]
            )
        )
        if result.stdout.strip():
            lines.extend(["", "æ åè¾åºï¼", result.stdout.strip()])
        if result.stderr.strip():
            lines.extend(["", "éè¯¯è¾åºï¼", result.stderr.strip()])
        self.matlab_log.setPlainText("
".join(lines))
        if result.returncode == 0:
            self.fit_metric["value"].setText("DRT å·²å®æ")
            QMessageBox.information(self, "MATLAB DRT å®æ", f"å·²çæ {len(result.output_files)} ä¸ª DRT æä»¶ï¼å¹¶è¾åºç©éµæä»¶ï¼
{result.output_dir}")
        else:
            self.fit_metric["value"].setText("DRT å¤±è´¥")
            self._warn(f"MATLAB DRT è¿è¡å¤±è´¥ã
è¯·æ¥çä¸æ¹æ¥å¿ã
è¾åºç®å½ï¼{result.output_dir}")
    def _export_current_bundle(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        output_path, fmt = self._select_export_target("å¯¼åºå½åæ°æ®", spectrum.metadata.file_path.stem)
        if output_path is None:
            return
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        fit = self.state.fits.get((spectrum.display_name, str(self.template_combo.currentData())))
        drt_dir = self._last_matlab_result.output_dir if self._last_matlab_result is not None else None
        paths = export_spectrum_bundle(output_path, spectrum, fit=fit, quality=quality, fmt=fmt, drt_source_dir=drt_dir)
        QMessageBox.information(self, "å¯¼åºå®æ", f"æä»¶å·²å¯¼åºå°ï¼
{self._export_result_location(paths)}")
    def _export_batch_bundle(self) -> None:
        if self.state.batch_summary is None:
            self._warn("è¯·åå®ææ¹éæåã")
            return
        output_path, fmt = self._select_export_target("å¯¼åºæ¹å¤çæ°æ®", "batch_export")
        if output_path is None:
            return
        drt_dir = self._last_matlab_result.output_dir if self._last_matlab_result is not None else None
        paths = export_batch_summary(output_path, self.state.batch_summary, fmt=fmt, drt_source_dir=drt_dir)
        QMessageBox.information(self, "æ¹å¤çå¯¼åºå®æ", f"æä»¶å·²å¯¼åºå°ï¼
{self._export_result_location(paths)}")
    def _select_export_target(self, title: str, default_stem: str) -> tuple[Path | None, str]:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            str(Path.cwd() / default_stem),
            "Excel Workbook (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not path:
            return None, "txt"
        selected_filter = selected_filter.lower()
        if "xlsx" in selected_filter or path.lower().endswith(".xlsx"):
            return Path(path), "xlsx"
        if "csv" in selected_filter or path.lower().endswith(".csv"):
            return Path(path), "csv"
        return Path(path), "txt"
    def _export_result_location(self, paths: dict[str, Path]) -> Path:
        first = next(iter(paths.values()))
        return first if first.suffix.lower() == ".xlsx" else first.parent
    def _set_nyquist_axes(self, plot, x_values: np.ndarray, y_values: np.ndarray) -> None:
        if pg is not None:
            class InteractiveViewBox(pg.ViewBox):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            x = np.asarray(x_values, dtype=float)
            y = np.asarray(y_values, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                return
            x = x[mask]
            y = y[mask]
            x_min = min(float(np.min(x)), 0.0)
            y_min = min(float(np.min(y)), 0.0)
            x_max = float(np.max(x))
            y_max = float(np.max(y))
            span = max(x_max - x_min, y_max - y_min, 1.0)
            pad = span * 0.08
            plot.setAspectLocked(False)
            plot.setXRange(x_min - pad, x_min + span + pad, padding=0.0)
            plot.setYRange(y_min - pad, y_min + span + pad, padding=0.0)
    def _reset_current_nyquist(self, plot) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or plot is None:
            return
        fit = None
        if plot is getattr(self, "fit_plot", None):
            fit = self.state.fits.get((spectrum.display_name, str(self.template_combo.currentData())))
        if fit is not None and fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
            self._set_nyquist_axes(
                plot,
                np.concatenate([spectrum.z_real_ohm, fit.predicted_real_ohm]),
                np.concatenate([spectrum.minus_z_imag_ohm, -fit.predicted_imag_ohm]),
            )
        else:
            self._set_nyquist_axes(plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
    def _reset_bode(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or pg is None:
            return
        x_min = float(np.min(spectrum.freq_hz))
        x_max = float(np.max(spectrum.freq_hz))
        self.bode_mag_plot.setXRange(np.log10(x_min), np.log10(x_max), padding=0.0)
        self.bode_phase_plot.setXRange(np.log10(x_min), np.log10(x_max), padding=0.0)
    def _reset_plot(self, plot, keep_legend: bool = False) -> None:
        if pg is None:
            return
        plot.clear()
        if not keep_legend and plot.plotItem.legend:
            plot.plotItem.legend.setParentItem(None)
            plot.plotItem.legend = None
    def _ensure_legend(self, plot) -> None:
        if pg is not None and plot.plotItem.legend is None:
            plot.addLegend(offset=(12, 12))
    def _plot_segment_preview(self, spectrum: SpectrumData, segment_hint: SegmentDetection) -> None:
        self._render_segment_overlay(spectrum, segment_hint)
    def _fit_error_lines(self, fit: FitOutcome) -> list[str]:
        lines: list[str] = []
        for key, label in self._primary_error_pairs(fit):
            value = fit.statistics.get(key)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                lines.append(f"  {label}: æ æ³è¯ä¼°")
            elif value > 20.0:
                lines.append(f"  {label}: {value:.2f}%  å»ºè®®å¤æ ¸")
            elif value > 10.0:
                lines.append(f"  {label}: {value:.2f}%  è­¦å")
            else:
                lines.append(f"  {label}: {value:.2f}%")
        return lines
    def _fit_has_high_error(self, fit: FitOutcome, threshold: float = 20.0) -> bool:
        return any(value > threshold for _, value in self._primary_error_values(fit))
    def _fit_error_summary(self, fit: FitOutcome | None) -> str:
        if fit is None:
            return ""
        values = self._primary_error_values(fit)
        if not values:
            return "æªè¯ä¼°"
        high_parts = [f"{label} {value:.1f}%" for label, value in values if value > 20.0]
        if high_parts:
            return "é«è¯¯å·®: " + "ï¼".join(high_parts)
        return "ï¼".join(f"{label} {value:.1f}%" for label, value in values)
    def _primary_error_pairs(self, fit: FitOutcome) -> list[tuple[str, str]]:
        if fit.model_key == "zview_double_rq_qrwo":
            return [("Rs_stderr_pct", "Rs"), ("Rsei_stderr_pct", "Rsei"), ("Rct_stderr_pct", "Rct")]
        return [("Rs_stderr_pct", "Rs"), ("Rct_stderr_pct", "Rct")]
    def _primary_error_values(self, fit: FitOutcome) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        for key, label in self._primary_error_pairs(fit):
            value = fit.statistics.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                values.append((label, value))
        return values
    def _fill_table(self, table, headers: list[str], rows: list[list[str]]) -> None:
        table.setUpdatesEnabled(False)
        table.clearContents()
        table.setSortingEnabled(False)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(value))
        if table is self.data_table:
            table.resizeColumnsToContents()
        elif table is self.batch_table:
            self._apply_batch_table_widths()
        table.setUpdatesEnabled(True)
    def _apply_batch_table_widths(self) -> None:
        widths = {
            0: 68,
            1: 220,
            2: 110,
            3: 250,
            4: 90,
            5: 90,
            6: 120,
            7: 160,
            8: 90,
            9: 90,
            10: 90,
            11: 90,
            12: 90,
            13: 90,
        }
        header = self.batch_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        for column, width in widths.items():
            if column < self.batch_table.columnCount():
                self.batch_table.setColumnWidth(column, width)
    def _color_batch_row(self, row_index: int, background: QColor, foreground: QColor) -> None:
        for column in range(self.batch_table.columnCount()):
            item = self.batch_table.item(row_index, column)
            if item is not None:
                item.setBackground(background)
                item.setForeground(foreground)
    def _format_stat(self, value) -> str:
        if value is None:
            return ""
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.2f}"
    def _current_spectrum(self, silent: bool = False) -> SpectrumData | None:
        row = self.state.current_index
        if 0 <= row < len(self.state.spectra):
            return self.state.spectra[row]
        if not silent:
            self._warn("è¯·åéæ©ä¸æ¡è°±å¾ã")
        return None
    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "EISMaster", message)
    def _open_circuit_builder(self) -> None:
        try:
            from eismaster.ui.circuit_builder.graphics import CircuitBuilderWindow
            self.circuit_builder = CircuitBuilderWindow(self)
            self.circuit_builder.show()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Graph-based logic editor is under construction.")
        self.matlab_drt_type_combo = ComboBox()
        for label, value in [("tau / gamma", 1), ("freq / gamma", 2), ("tau / g", 3), ("freq / g", 4)]:
            self.matlab_drt_type_combo.addItem(label, userData=value)
        self.matlab_inductance_combo = ComboBox()
        for label, value in [("Ignore", 1), ("Fit", 2), ("Discard", 3)]:
            self.matlab_inductance_combo.addItem(label, userData=value)
        matlab_exe_browse = PushButton(FIF.FOLDER, "ÃÂÃÂ¦ÃÂÃÂµÃÂÃÂÃÂÃÂ¨ÃÂÃÂ§ÃÂÃÂ")
        matlab_exe_browse.clicked.connect(self._browse_matlab_exe)
        drttools_browse = PushButton(FIF.FOLDER, "ÃÂÃÂ¦ÃÂÃÂµÃÂÃÂÃÂÃÂ¨ÃÂÃÂ§ÃÂÃÂ")
        drttools_browse.clicked.connect(self._browse_drttools_dir)
        grid.addWidget(BodyLabel("MATLAB ÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ¯ÃÂÃÂ¥ÃÂÃÂ¾ÃÂÃÂ"), 0, 0)
        grid.addWidget(self.matlab_exe_edit, 0, 1)
        grid.addWidget(matlab_exe_browse, 0, 2)
        grid.addWidget(BodyLabel("DRTtools ÃÂÃÂ¨ÃÂÃÂ·ÃÂÃÂ¯ÃÂÃÂ¥ÃÂÃÂ¾ÃÂÃÂ"), 1, 0)
        grid.addWidget(self.drttools_dir_edit, 1, 1)
        grid.addWidget(drttools_browse, 1, 2)
        grid.addWidget(BodyLabel("ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¦ÃÂÃÂ³ÃÂÃÂ"), 2, 0)
        grid.addWidget(self.matlab_method_combo, 2, 1)
        grid.addWidget(BodyLabel("DRT type"), 3, 0)
        grid.addWidget(self.matlab_drt_type_combo, 3, 1)
        grid.addWidget(BodyLabel("Lambda"), 4, 0)
        grid.addWidget(self.matlab_lambda_edit, 4, 1)
        grid.addWidget(BodyLabel("Coeff"), 5, 0)
        grid.addWidget(self.matlab_coeff_edit, 5, 1)
        grid.addWidget(BodyLabel("Inductance"), 6, 0)
        grid.addWidget(self.matlab_inductance_combo, 6, 1)
        matlab_panel.layout().addLayout(grid)
        self.matlab_status = self._info_box()
        self.matlab_status.setMaximumHeight(100)
        matlab_panel.layout().addWidget(self.matlab_status)
        self.matlab_log = self._info_box()
        self.matlab_log.setMinimumHeight(200)
        matlab_panel.layout().addWidget(self.matlab_log)
        layout.addWidget(matlab_panel)
        return tab, layout
    def _tab_shell(self) -> tuple[QWidget, QVBoxLayout]:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.viewport().setObjectName("scrollAreaWidgetContents")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        scroll.setWidget(container)
        return scroll, layout
    def _panel(self, title: str, subtitle: str) -> SimpleCardWidget:
        card = SimpleCardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 15)
        layout.setSpacing(10)
        header = QVBoxLayout()
        t = StrongBodyLabel(title)
        s = CaptionLabel(subtitle)
        s.setWordWrap(True)
        header.addWidget(t)
        header.addWidget(s)
        layout.addLayout(header)
        return card
    def _configure_plots(self) -> None:
        if pg is None:
            return
        pg.setConfigOptions(antialias=False, foreground="#A0A0B0")
        for plot in self.findChildren(pg.PlotWidget):
            item = plot.getPlotItem()
            plot.setBackground("#121214")
            plot.setMenuEnabled(False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            item.setDownsampling(auto=True, mode="peak")
            item.setClipToView(True)
            item.getViewBox().setMouseEnabled(x=True, y=True)
            for axis in ("bottom", "left"):
                item.getAxis(axis).setPen(pg.mkPen("#cfd4dc"))
                item.getAxis(axis).setTextPen(pg.mkPen("#6e6e73"))
        if hasattr(self, "bode_phase_plot") and hasattr(self, "bode_mag_plot"):
            self.bode_phase_plot.setXLink(self.bode_mag_plot)
    def _info_box(self) -> QTextEdit:
        box = QTextEdit()
        box.setObjectName("infoPane")
        box.setReadOnly(True)
        return box
    def _chip(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("chipLabel")
        return label
    def _metric(self, label: str, value: str) -> dict[str, QWidget]:
        container = SimpleCardWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        l = CaptionLabel(label)
        v = StrongBodyLabel(value)
        v.setWordWrap(True)
        layout.addWidget(l)
        layout.addWidget(v)
        return {"container": container, "value": v}
    def _plot(self, title: str, x_label: str, y_label: str, *, log_x: bool = False, log_y: bool = False):
        plot = pg.PlotWidget(title=title, viewBox=InteractiveViewBox())
        plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.setLogMode(x=log_x, y=log_y)
        return plot
    def _table(self) -> TableWidget:
        table = TableWidget()
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return table
    def _import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import EIS Files", str(Path.cwd()), "EIS Files (*.bin *.txt *.csv)")
        if not paths:
            return
        spectra: list[SpectrumData] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    spectra.append(load_spectrum(path))
                except Exception as exc:
                    self._warn(f"Failed to import file:
{path}
{exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._merge_state(spectra)
    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import Folder", str(Path.cwd()))
        if not folder:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            spectra = load_spectra_from_folder(folder)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._warn(str(exc))
            return
        QApplication.restoreOverrideCursor()
        self._merge_state(spectra)
    def _replace_state(self, spectra: list[SpectrumData]) -> None:
        self.state = AppState(spectra=spectra)
        self._rebuild_file_list(0 if spectra else -1)
        self._refresh_import_text()
        self._refresh_batch_view()
        if not spectra:
            self._clear_views()
    def _merge_state(self, spectra: list[SpectrumData]) -> None:
        if not spectra:
            return
        existing_paths = {self._spectrum_path_key(spectrum) for spectrum in self.state.spectra}
        appended: list[SpectrumData] = []
        for spectrum in spectra:
            path_key = self._spectrum_path_key(spectrum)
            if path_key in existing_paths:
                continue
            existing_paths.add(path_key)
            appended.append(spectrum)
        if not appended:
            self._warn("No new spectra were added. The selected files are already loaded.")
            return
        self.state.spectra.extend(appended)
        self._invalidate_batch_outputs()
        new_index = len(self.state.spectra) - len(appended)
        self._rebuild_file_list(new_index)
        self._refresh_import_text()
        self._refresh_current_tab()
        self._update_global_status()
    def _rebuild_file_list(self, current_row: int) -> None:
        for table in self._queue_tables.values():
            blocker = QSignalBlocker(table)
            table.clearContents()
            table.setRowCount(len(self.state.spectra))
            for i, spectrum in enumerate(self.state.spectra):
                item = QTableWidgetItem(f"{spectrum.display_name}
{spectrum.metadata.source_format.upper()} ÃÂÃÂÃÂÃÂ· {spectrum.n_points} pt")
                table.setItem(i, 0, item)
            del blocker
        if self.state.spectra:
            row = min(max(current_row, 0), len(self.state.spectra) - 1)
            self._set_current_index(row)
        else:
            self.state.current_index = -1
            self._sync_queue_selection(-1)
    def _refresh_import_text(self) -> None:
        count = len(self.state.spectra)
        self.loaded_metric["value"].setText(str(count))
        if count == 0:
            self.brand_label.setText("EISMaster Pro")
            self.selection_chip.setText("Selection: none")
            self.instrument_chip.setText("Instrument: waiting for import")
        else:
            current = self._current_spectrum(silent=True) or self.state.spectra[0]
            self.brand_label.setText(f"EISMaster Pro - {current.display_name}")
    def _sync_queue_selection(self, row: int) -> None:
        for table in self._queue_tables.values():
            blocker = QSignalBlocker(table)
            if 0 <= row < table.rowCount():
                table.setCurrentCell(row, 0)
                table.selectRow(row)
                item = table.item(row, 0)
                if item is not None:
                    table.scrollToItem(item)
            else:
                table.clearSelection()
            del blocker
    def _set_current_index(self, row: int, *, refresh_page: bool = True) -> None:
        if row < 0 or row >= len(self.state.spectra):
            return
        self.state.current_index = row
        self._sync_queue_selection(row)
        spectrum = self.state.spectra[row]
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        self._prime_segment_controls(spectrum)
        self._refresh_sidebar(spectrum, quality)
        self._update_global_status()
        if refresh_page:
            self._refresh_current_tab()
    def _spectrum_path_key(self, spectrum: SpectrumData) -> str:
        return str(spectrum.metadata.file_path.resolve()).lower()
    def _invalidate_batch_outputs(self) -> None:
        self.state.batch_summary = None
        self._last_matlab_result = None
        self._last_drt_spectra = []
        if hasattr(self, "batch_progress"):
            self.batch_progress.setValue(0)
        if hasattr(self, "batch_progress_label"):
            self.batch_progress_label.setText("Batch fitting not started")
    def _remove_current_spectrum(self) -> None:
        row = self.state.current_index
        if row < 0 or row >= len(self.state.spectra):
            return
        spectrum = self.state.spectra.pop(row)
        self.state.qualities.pop(spectrum.display_name, None)
        for key in [key for key in self.state.fits if key[0] == spectrum.display_name]:
            self.state.fits.pop(key, None)
        self._invalidate_batch_outputs()
        self._rebuild_file_list(min(row, len(self.state.spectra) - 1))
        self._refresh_import_text()
        if not self.state.spectra:
            self._clear_views()
            return
        self._set_current_index(self.state.current_index)
    def _clear_all_spectra(self) -> None:
        if not self.state.spectra:
            return
        self._replace_state([])
    def _move_current_spectrum_up(self) -> None:
        row = self.state.current_index
        if row <= 0 or row >= len(self.state.spectra):
            return
        self.state.spectra[row - 1], self.state.spectra[row] = self.state.spectra[row], self.state.spectra[row - 1]
        self._invalidate_batch_outputs()
        self._rebuild_file_list(row - 1)
        self._refresh_import_text()
    def _move_current_spectrum_down(self) -> None:
        row = self.state.current_index
        if row < 0 or row >= len(self.state.spectra) - 1:
            return
        self.state.spectra[row + 1], self.state.spectra[row] = self.state.spectra[row], self.state.spectra[row + 1]
        self._invalidate_batch_outputs()
        self._rebuild_file_list(row + 1)
        self._refresh_import_text()
    def _clear_views(self) -> None:
        for meta_box in self._queue_meta_boxes.values():
            meta_box.setPlainText("Import data to inspect metadata and status.")
        self.quality_text.setPlainText("Quality report will appear here.")
        self.fit_text.setPlainText("Fit results will appear here.")
        self.batch_text.setPlainText("Batch summary will appear here.")
        self.matlab_status.setPlainText("MATLAB DRT status will appear here.")
        self.matlab_log.setPlainText("")
        if hasattr(self, "segment_split1_freq_edit"):
            self.segment_split1_freq_edit.clear()
            self.segment_split2_freq_edit.clear()
            self.segment_split1_spin.clear()
            self.segment_split2_spin.clear()
            self._update_split_info_labels(None)
        if hasattr(self, "batch_progress") and self.batch_progress:
            self.batch_progress.setValue(0)
        self.data_table.clearContents()
        self.data_table.setRowCount(0)
        self.batch_table.clearContents()
        self.batch_table.setRowCount(0)
        if pg is not None:
            plots = [
                getattr(self, "nyquist_plot", None), 
                getattr(self, "bode_mag_plot", None), 
                getattr(self, "bode_phase_plot", None),
                getattr(self, "fit_nyquist_plot", None),
                getattr(self, "fit_residual_plot", None),
                getattr(self, "batch_plot", None)
            ]
            for plot in plots:
                if plot:
                    self._reset_plot(plot)
            self._clear_segment_overlay()
        self._update_global_status()
    def _update_global_status(self) -> None:
        if not self.state.spectra:
            self.instrument_chip.setText("Instrument: waiting for import")
            self.selection_chip.setText("Sample: none")
            return
        current = self._current_spectrum(silent=True)
        self.loaded_metric["value"].setText(str(len(self.state.spectra)))
        if current:
            self.instrument_chip.setText(f"Instrument: {current.metadata.instrument_model or '-'}")
            self.selection_chip.setText(f"Current: {current.display_name}")
            self.brand_label.setText(f"EISMaster Pro - {current.display_name}")
        else:
            self.selection_chip.setText("Sample: none")
        self.fit_metric["value"].setText(self._status_text_for_current(current))
    def _status_text_for_current(self, current: SpectrumData | None) -> str:
        if self.state.drt_busy:
            return "DRT running"
        if self.state.batch_busy:
            return "Batch fitting running"
        if current is not None:
            fit = self._latest_fit_for_spectrum(current.display_name)
            if fit is not None:
                labels = {
                    "ok": "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ°ÃÂÃÂ±ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ",
                    "warn": "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ°ÃÂÃÂ±ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ­ÃÂÃÂ¦ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ",
                    "failed": "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ°ÃÂÃÂ±ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥",
                    "unavailable": "unavailable",
                }
                return labels.get(fit.status, fit.status)
        summary = self.state.batch_summary
        if summary is not None and summary.items:
            total = len(summary.items)
            done = sum(1 for item in summary.items if item.fit is not None)
            failed = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "failed")
            warned = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "warn")
            if failed:
                return f"Batch complete {done}/{total} with failures"
            if warned:
                return f"Batch complete {done}/{total} with warnings"
            return f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ {done}/{total}"
        return "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ°ÃÂÃÂ±ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂª"
    def _latest_fit_for_spectrum(self, display_name: str) -> FitOutcome | None:
        preferred_key = str(self.template_combo.currentData()) if hasattr(self, "template_combo") else None
        if preferred_key is not None:
            fit = self.state.fits.get((display_name, preferred_key))
            if fit is not None:
                return fit
        for key, fit in reversed(list(self.state.fits.items())):
            if key[0] == display_name:
                return fit
        return None
    def _on_queue_row_changed(self, page_key: str) -> None:
        table = self._queue_tables.get(page_key)
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self.state.spectra):
            return
        self._set_current_index(row)
    def _on_tab_changed(self, _: int) -> None:
        self._refresh_current_tab()
    def _current_interface_id(self) -> str:
        widget = self.stackedWidget.currentWidget()
        return widget.objectName() if widget is not None else ""
    def _refresh_current_tab(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        current_id = self._current_interface_id()
        if current_id == "inspectInterface":
            self._refresh_inspect_view(spectrum, quality)
        elif current_id == "fitInterface":
            self._refresh_fit_view(spectrum)
        elif current_id == "batchInterface":
            self._refresh_batch_view()
    def _refresh_sidebar(self, spectrum: SpectrumData, quality: QualityReport) -> None:
        lines = [
            f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¶: {spectrum.metadata.file_path.name}",
            f"Source: {spectrum.metadata.source_format.upper()} | {spectrum.n_points} points",
            f"ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ©ÃÂÃÂÃÂÃÂ: {spectrum.acquired_label}",
            f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¦ÃÂÃÂ£ÃÂÃÂÃÂÃÂ¦ÃÂÃÂ? {quality.status}",
        ]
        text = "
".join(lines)
        for meta_box in self._queue_meta_boxes.values():
            meta_box.setPlainText(text)
    def _refresh_inspect_view(self, spectrum: SpectrumData, quality: QualityReport) -> None:
        lines = [f"ÃÂÃÂ¦ÃÂÃÂ£ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ®? {quality.status.upper()}", f"ÃÂÃÂ©ÃÂÃÂ¢ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¦ÃÂÃÂ? {spectrum.n_points}"]
        lines.append("-" * 20)
        lines.extend(quality.summary_lines())
        self.quality_text.setPlainText("
".join(lines))
        headers = ["freq_hz", "z_real_ohm", "z_imag_ohm", "minus_z_imag_ohm", "z_mod_ohm", "phase_deg"]
        rows = [
            [f"{a:.8g}", f"{b:.8g}", f"{c:.8g}", f"{d:.8g}", f"{e:.8g}", f"{f:.8g}"]
            for a, b, c, d, e, f in zip(
                spectrum.freq_hz,
                spectrum.z_real_ohm,
                spectrum.z_imag_ohm,
                spectrum.minus_z_imag_ohm,
                spectrum.z_mod_ohm,
                spectrum.phase_deg,
            )
        ]
        self._fill_table(self.data_table, headers, rows)
        if pg is None:
            return
        pen = pg.mkPen("#0071e3", width=1.6)
        brush = pg.mkBrush(QColor("#5ac8fa"))
        for plot in (self.nyquist_plot, self.bode_mag_plot, self.bode_phase_plot):
            self._reset_plot(plot)
        self.nyquist_plot.plot(spectrum.z_real_ohm, spectrum.minus_z_imag_ohm, pen=pen, symbol="o", symbolSize=5, symbolBrush=brush)
        self.bode_mag_plot.plot(spectrum.freq_hz, spectrum.z_mod_ohm, pen=pen, symbol="o", symbolSize=4, symbolBrush=brush)
        self.bode_phase_plot.plot(spectrum.freq_hz, spectrum.phase_deg, pen=pen, symbol="o", symbolSize=4, symbolBrush=brush)
        self._set_nyquist_axes(self.nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
    def _toggle_quality_detail(self) -> None:
        pass
    def _run_kk_for_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            quality = assess_spectrum_quality(spectrum, run_kk=True)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._warn(f"KK ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{exc}")
            return
        QApplication.restoreOverrideCursor()
        self.state.qualities[spectrum.display_name] = quality
        self._refresh_sidebar(spectrum, quality)
        self._refresh_inspect_view(spectrum, quality)
        self._update_global_status()
    def _is_zview_template_selected(self) -> bool:
        return str(self.template_combo.currentData()).startswith("zview_")
    def _segment_min_points(self) -> int:
        return 6
    def _format_segment_frequency(self, value: float | None) -> str:
        if value is None or not np.isfinite(value) or value <= 0:
            return ""
        return f"{value:.6g}"
    def _segment_freq_for_index(self, spectrum: SpectrumData, index: int | None) -> float | None:
        if index is None or index < 0 or index >= spectrum.n_points:
            return None
        return float(spectrum.freq_hz[int(index)])
    def _segment_index_for_frequency(self, spectrum: SpectrumData, freq_hz: float | None) -> int | None:
        if freq_hz is None or not np.isfinite(freq_hz) or freq_hz <= 0:
            return None
        freqs = spectrum.freq_hz.astype(float, copy=False)
        safe_freqs = np.clip(freqs, np.finfo(float).tiny, None)
        target = float(np.clip(freq_hz, safe_freqs.min(), safe_freqs.max()))
        nearest = int(np.argmin(np.abs(np.log10(safe_freqs) - np.log10(target))))
        return nearest
    def _segment_index_bounds(
        self,
        spectrum: SpectrumData,
        resolved_mode: str,
        base_detection: SegmentDetection,
        split1: int | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int] | None]:
        min_points = self._segment_min_points()
        n_points = spectrum.n_points
        if resolved_mode == "single":
            lo = min_points - 1
            hi = max(lo, n_points - min_points)
            return (lo, hi), None
        peak2 = base_detection.peak_indices[1] if len(base_detection.peak_indices) > 1 else None
        split2_hi = max(min_points - 1, n_points - min_points)
        split1_lo = min_points - 1
        split1_hi = max(split1_lo, split2_hi - (min_points - 1))
        if peak2 is not None:
            split1_hi = max(split1_lo, min(split1_hi, peak2 - 1))
        split1_bounds = (split1_lo, split1_hi)
        effective_split1 = split1 if split1 is not None else base_detection.split_indices[0]
        effective_split1 = int(np.clip(effective_split1, split1_bounds[0], split1_bounds[1]))
        split2_lo = effective_split1 + (min_points - 1)
        if peak2 is not None:
            split2_lo = max(split2_lo, peak2)
        split2_lo = min(split2_lo, split2_hi)
        split2_bounds = (split2_lo, split2_hi)
        return split1_bounds, split2_bounds
    def _sanitize_segment_indices(
        self,
        spectrum: SpectrumData,
        requested_mode: str,
        split1: int | None,
        split2: int | None,
        base_detection: SegmentDetection | None = None,
    ) -> tuple[int | None, int | None, SegmentDetection]:
        detection = base_detection or detect_segments(spectrum, mode=requested_mode)
        resolved_mode = detection.resolved_mode if requested_mode == "auto" else requested_mode
        if resolved_mode == "single":
            bounds, _ = self._segment_index_bounds(spectrum, "single", detection)
            candidate = detection.split_indices[0] if split1 is None else split1
            split1_value = int(np.clip(candidate, bounds[0], bounds[1]))
            return split1_value, None, detection
        split1_bounds, split2_bounds = self._segment_index_bounds(spectrum, "double", detection, split1)
        default_split1 = detection.split_indices[0] if detection.split_indices else split1_bounds[0]
        split1_value = int(np.clip(default_split1 if split1 is None else split1, split1_bounds[0], split1_bounds[1]))
        split1_bounds, split2_bounds = self._segment_index_bounds(spectrum, "double", detection, split1_value)
        default_split2 = detection.split_indices[1] if len(detection.split_indices) > 1 else split2_bounds[0]
        split2_value = int(np.clip(default_split2 if split2 is None else split2, split2_bounds[0], split2_bounds[1]))
        return split1_value, split2_value, detection
    def _set_split2_controls_visible(self, visible: bool) -> None:
        for widget in getattr(self, "segment_split2_widgets", []):
            widget.setVisible(visible)
        if hasattr(self, "segment_arc2_label"):
            self.segment_arc2_label.setVisible(visible)
    def _sync_segment_controls_from_detection(self, spectrum: SpectrumData, detection: SegmentDetection) -> None:
        split_indices = list(detection.split_indices)
        split1 = split_indices[0] if split_indices else None
        split2 = split_indices[1] if len(split_indices) > 1 else None
        show_split2 = detection.resolved_mode == "double"
        self._set_split2_controls_visible(show_split2)
        self._segment_control_sync = True
        try:
            with QSignalBlocker(self.segment_split1_freq_edit):
                self.segment_split1_freq_edit.setText(self._format_segment_frequency(self._segment_freq_for_index(spectrum, split1)))
            with QSignalBlocker(self.segment_split1_spin):
                self.segment_split1_spin.setText("" if split1 is None else str(split1))
            with QSignalBlocker(self.segment_split2_freq_edit):
                self.segment_split2_freq_edit.setText(self._format_segment_frequency(self._segment_freq_for_index(spectrum, split2)))
            with QSignalBlocker(self.segment_split2_spin):
                self.segment_split2_spin.setText("" if split2 is None else str(split2))
        finally:
            self._segment_control_sync = False
        self.segment_split1_freq_label.setText("split Frequency [Hz]" if not show_split2 else "split1 Frequency [Hz]")
        if hasattr(self, "segment_split2_freq_label"):
            self.segment_split2_freq_label.setText("split2 Frequency [Hz]")
    def _refresh_segment_controls_state(self, spectrum: SpectrumData | None = None) -> None:
        enabled = self._is_zview_template_selected()
        for widget in (
            self.segment_mode_combo,
            self.segment_split1_freq_edit,
            self.segment_split2_freq_edit,
            self.segment_split1_spin,
            self.segment_split2_spin,
            self.segment_detect_btn,
        ):
            widget.setEnabled(enabled)
        if enabled and spectrum is not None:
            detection = detect_segments(spectrum, mode=str(self.segment_mode_combo.currentData() or "auto"))
            show_split2 = detection.resolved_mode == "double"
            self._set_split2_controls_visible(show_split2)
            freq_min = float(np.min(spectrum.freq_hz))
            freq_max = float(np.max(spectrum.freq_hz))
            self.segment_split1_freq_edit.setPlaceholderText(f"{freq_min:.3g} - {freq_max:.3g}")
            self.segment_split2_freq_edit.setPlaceholderText(f"{freq_min:.3g} - {freq_max:.3g}")
            self.segment_split2_freq_edit.setEnabled(enabled and show_split2)
        else:
            self._set_split2_controls_visible(False)
    def _prime_segment_controls(self, spectrum: SpectrumData) -> None:
        self._refresh_segment_controls_state(spectrum)
        if not self._is_zview_template_selected():
            return
        detection = detect_segments(spectrum, mode=str(self.segment_mode_combo.currentData() or "auto"))
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._update_split_info_labels(spectrum, detection)
    def _detect_segments_for_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        self._prime_segment_controls(spectrum)
        self._refresh_segment_preview_only(spectrum)
        self._fit_current()
    def _current_segment_hint(self, spectrum: SpectrumData) -> SegmentDetection | None:
        if not self._is_zview_template_selected():
            return None
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        base_detection = detect_segments(spectrum, mode=requested_mode)
        try:
            split1_freq = float(self.segment_split1_freq_edit.text()) if self.segment_split1_freq_edit.text() else None
        except ValueError:
            split1_freq = None
        try:
            split2_freq = float(self.segment_split2_freq_edit.text()) if self.segment_split2_freq_edit.text() else None
        except ValueError:
            split2_freq = None
        split1 = self._segment_index_for_frequency(spectrum, split1_freq)
        split2 = self._segment_index_for_frequency(spectrum, split2_freq)
        split1, split2, _ = self._sanitize_segment_indices(spectrum, requested_mode, split1, split2, base_detection)
        return detect_segments(
            spectrum,
            mode=requested_mode,
            manual_split1=split1,
            manual_split2=split2,
        )
    def _on_segment_frequency_edited(self, split_role: str) -> None:
        if self._segment_control_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or not self._is_zview_template_selected():
            return
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        base_detection = detect_segments(spectrum, mode=requested_mode)
        edit1 = self.segment_split1_freq_edit.text().strip()
        edit2 = self.segment_split2_freq_edit.text().strip()
        try:
            split1 = self._segment_index_for_frequency(spectrum, float(edit1)) if edit1 else None
        except ValueError:
            split1 = None
        try:
            split2 = self._segment_index_for_frequency(spectrum, float(edit2)) if edit2 else None
        except ValueError:
            split2 = None
        split1, split2, _ = self._sanitize_segment_indices(spectrum, requested_mode, split1, split2, base_detection)
        detection = detect_segments(spectrum, mode=requested_mode, manual_split1=split1, manual_split2=split2)
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._refresh_segment_preview_only(spectrum, detection)
        self._fit_current()
    def _refresh_fit_for_current_selection(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        self._refresh_segment_controls_state(spectrum)
        if spectrum is not None and self._is_zview_template_selected() and not self._segment_drag_active:
            self._prime_segment_controls(spectrum)
        else:
            self._update_split_info_labels(spectrum)
        if spectrum is not None and self._current_interface_id() == "fitInterface":
            self._refresh_fit_view(spectrum)
        self._update_global_status()
    def _update_split_info_labels(
        self,
        spectrum: SpectrumData | None = None,
        detection: SegmentDetection | None = None,
    ) -> None:
        if spectrum is None or not self._is_zview_template_selected():
            self.segment_mode_status.setText("Segment mode: unavailable")
            self.segment_peak_info_label.setText("Peaks: -")
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
            self._set_split2_controls_visible(False)
            return
        detection = detection or self._current_segment_hint(spectrum)
        if detection is None:
            self.segment_mode_status.setText("Segment mode: unavailable")
            self.segment_peak_info_label.setText("Peaks: -")
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
            self._set_split2_controls_visible(False)
            return
        self._set_split2_controls_visible(detection.resolved_mode == "double")
        peak_bits = [
            f"p{i + 1}=idx {peak} @ {self._format_segment_frequency(self._segment_freq_for_index(spectrum, peak))} Hz"
            for i, peak in enumerate(detection.peak_indices)
        ]
        self.segment_mode_status.setText(
            f"Segment mode: {detection.requested_mode} -> {detection.resolved_mode}"
        )
        self.segment_peak_info_label.setText("Peaks: " + (", ".join(peak_bits) if peak_bits else "-"))
        def region_text(name: str, start: int, stop: int) -> str:
            start_freq = self._format_segment_frequency(self._segment_freq_for_index(spectrum, start))
            stop_freq = self._format_segment_frequency(self._segment_freq_for_index(spectrum, stop))
            points = max(stop - start + 1, 0)
            return f"{name}: idx {start}-{stop} | {start_freq} -> {stop_freq} Hz | {points} pt"
        split_indices = list(detection.split_indices)
        if detection.resolved_mode == "double" and len(split_indices) >= 2:
            split1, split2 = split_indices[:2]
            self.segment_arc1_label.setText(region_text("Arc1", 0, split1))
            self.segment_arc2_label.setText(region_text("Arc2", split1, split2))
            self.segment_tail_label.setText(region_text("Tail", split2, spectrum.n_points - 1))
        elif split_indices:
            split1 = split_indices[0]
            self.segment_arc1_label.setText(region_text("Arc1", 0, split1))
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText(region_text("Tail", split1, spectrum.n_points - 1))
        else:
            self.segment_arc1_label.setText("Arc1: -")
            self.segment_arc2_label.setText("Arc2: -")
            self.segment_tail_label.setText("Tail: -")
    def _clear_segment_overlay(self, *, keep_handles: bool = False) -> None:
        if pg is None or not hasattr(self, "fit_nyquist_plot"):
            self._segment_overlay_items = []
            if not keep_handles:
                self._segment_handles = {}
            return
        seen: set[int] = set()
        items = list(self._segment_overlay_items)
        if not keep_handles:
            items.extend(self._segment_handles.values())
        for item in items:
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            try:
                self.fit_nyquist_plot.removeItem(item)
            except Exception:
                pass
        self._segment_overlay_items = []
        if not keep_handles:
            self._segment_handles = {}
    def _render_segment_overlay(self, spectrum: SpectrumData, segment_hint: SegmentDetection | None) -> None:
        self._clear_segment_overlay(keep_handles=True)
        if pg is None or segment_hint is None:
            self._clear_segment_overlay()
            return
        split_indices = list(segment_hint.split_indices)
        peak_indices = list(segment_hint.peak_indices)
        sections: list[tuple[int, int, str, float]] = []
        handle_specs: list[tuple[str, int, str]] = []
        if segment_hint.resolved_mode == "double" and len(split_indices) >= 2:
            split1, split2 = split_indices[:2]
            p1 = peak_indices[0] if len(peak_indices) > 0 else max(0, split1 // 2)
            p2 = peak_indices[1] if len(peak_indices) > 1 else max(split1 + 1, (split1 + split2) // 2)
            sections = [
                (0, split1, "#007aff", 2.2),      # Arc 1: Blue
                (split1, split2, "#af52de", 2.2), # Arc 2: Purple
                (split2, spectrum.n_points - 1, "#ff3b30", 1.8), # Tail: Red
            ]
            handle_specs = [
                ("peak1", p1, "#007aff"),
                ("split1", split1, "#5ac8fa"),
                ("peak2", p2, "#af52de"),
                ("split2", split2, "#ff453a"),
            ]
        elif split_indices:
            split1 = split_indices[0]
            p1 = peak_indices[0] if len(peak_indices) > 0 else max(0, split1 // 2)
            sections = [
                (0, split1, "#007aff", 2.2),
                (split1, spectrum.n_points - 1, "#ff3b30", 1.8),
            ]
            handle_specs = [
                ("peak1", p1, "#007aff"),
                ("split1", split1, "#5ac8fa"),
            ]
        for start, stop, color, width in sections:
            xs = spectrum.z_real_ohm[start : stop + 1]
            ys = spectrum.minus_z_imag_ohm[start : stop + 1]
            # Use solid lines for segments, with slightly different widths
            item = self.fit_nyquist_plot.plot(xs, ys, pen=pg.mkPen(color=color, width=width))
            self._segment_overlay_items.append(item)
        active_roles = {role for role, _, _ in handle_specs}
        for role, handle in list(self._segment_handles.items()):
            if role in active_roles:
                continue
            try:
                self.fit_nyquist_plot.removeItem(handle)
            except Exception:
                pass
            self._segment_handles.pop(role, None)
        for role, index, color in handle_specs:
            x = float(spectrum.z_real_ohm[index])
            y = float(spectrum.minus_z_imag_ohm[index])
            handle = self._segment_handles.get(role)
            is_peak = role.startswith("peak")
            if handle is None:
                brush_color = QColor(color)
                brush_color.setAlpha(180)
                handle = pg.TargetItem(
                    pos=(x, y),
                    size=14 if is_peak else 12,
                    symbol="d" if is_peak else "o",
                    pen=pg.mkPen(color=color, width=2),
                    hoverPen=pg.mkPen(color="#ffffff", width=2.5),
                    brush=pg.mkBrush(brush_color),
                    hoverBrush=pg.mkBrush(QColor("#ffffff")),
                    movable=True,
                )
                handle.sigPositionChanged.connect(lambda _, r=role: self._on_segment_handle_moved(r))
                handle.sigPositionChangeFinished.connect(lambda _, r=role: self._on_segment_handle_released(r))
                self.fit_nyquist_plot.addItem(handle)
                self._segment_handles[role] = handle
            self._segment_handle_sync = True
            try:
                handle.setPos((x, y))
            finally:
                self._segment_handle_sync = False
    def _refresh_segment_preview_only(
        self,
        spectrum: SpectrumData,
        detection: SegmentDetection | None = None,
    ) -> None:
        detection = detection or self._current_segment_hint(spectrum)
        if detection is None:
            self._clear_segment_overlay()
            return
        self._sync_segment_controls_from_detection(spectrum, detection)
        self._update_split_info_labels(spectrum, detection)
        if pg is not None:
            self._render_segment_overlay(spectrum, detection)
    def _ensure_fit_plot_click_handler(self) -> None:
        if pg is None or self._fit_plot_click_connected:
            return
        self.fit_nyquist_plot.scene().sigMouseClicked.connect(self._on_fit_plot_scene_clicked)
        self._fit_plot_click_connected = True
    def _move_segment_to_index(
        self,
        spectrum: SpectrumData,
        split_role: str,
        candidate_index: int,
        *,
        commit: bool,
    ) -> None:
        requested_mode = str(self.segment_mode_combo.currentData() or "auto")
        base_detection = detect_segments(spectrum, mode=requested_mode)
        current_hint = self._current_segment_hint(spectrum) or base_detection
        split1 = current_hint.split_indices[0] if current_hint.split_indices else None
        split2 = current_hint.split_indices[1] if len(current_hint.split_indices) > 1 else None
        peak1 = current_hint.peak_indices[0] if current_hint.peak_indices else None
        peak2 = current_hint.peak_indices[1] if len(current_hint.peak_indices) > 1 else None
        if split_role == "split1":
            split1 = candidate_index
        elif split_role == "split2":
            split2 = candidate_index
        elif split_role == "peak1":
            peak1 = candidate_index
        elif split_role == "peak2":
            peak2 = candidate_index
        detection = detect_segments(
            spectrum, 
            mode=requested_mode, 
            manual_split1=split1, 
            manual_split2=split2,
            manual_peak1=peak1,
            manual_peak2=peak2
        )
        self.state.segment_hints[spectrum.display_name] = detection
        self._segment_drag_active = not commit
        self._refresh_segment_preview_only(spectrum, detection)
        self._segment_drag_active = False
    def _nearest_segment_index_from_handle(self, spectrum: SpectrumData, split_role: str) -> int | None:
        handle = self._segment_handles.get(split_role)
        if handle is None:
            return None
        point = handle.pos()
        x_pos = float(point.x())
        y_pos = float(point.y())
        dists = (spectrum.z_real_ohm - x_pos) ** 2 + (spectrum.minus_z_imag_ohm - y_pos) ** 2
        return int(np.argmin(dists))
    def _on_segment_handle_moved(self, split_role: str) -> None:
        if self._segment_handle_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        nearest_index = self._nearest_segment_index_from_handle(spectrum, split_role)
        if nearest_index is None:
            return
        self._segment_handle_sync = True
        try:
            self._move_segment_to_index(spectrum, split_role, nearest_index, commit=False)
        finally:
            self._segment_handle_sync = False
    def _on_segment_handle_released(self, split_role: str) -> None:
        if self._segment_handle_sync:
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        nearest_index = self._nearest_segment_index_from_handle(spectrum, split_role)
        if nearest_index is None:
            return
        self._move_segment_to_index(spectrum, split_role, nearest_index, commit=True)
    def _refresh_fit_view(self, spectrum: SpectrumData) -> None:
        key = str(self.template_combo.currentData())
        fit = self.state.fits.get((spectrum.display_name, key))
        if fit is None:
            for stored_key, stored_fit in self.state.fits.items():
                if stored_key[0] == spectrum.display_name:
                    fit = stored_fit
                    break
        segment_hint = self._current_segment_hint(spectrum)
        if segment_hint is not None:
            self._sync_segment_controls_from_detection(spectrum, segment_hint)
                    f"Segment mode: {segment_hint.requested_mode} -> {segment_hint.resolved_mode}",
                    f"Peak indices: {', '.join(str(i) for i in segment_hint.peak_indices) or '-'}",
                    f"Split indices: {', '.join(str(i) for i in segment_hint.split_indices) or '-'}",
        if segment_hint is not None:
            detail_lines.extend(
                [
                    f"ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ¦ÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂ¥ÃÂÃÂ¼ÃÂÃÂ: {segment_hint.requested_mode} -> {segment_hint.resolved_mode}",
                    f"ÃÂÃÂ¥ÃÂÃÂ³ÃÂÃÂ°ÃÂÃÂ¤ÃÂÃÂ½ÃÂÃÂÃÂÃÂ§ÃÂÃÂ´ÃÂÃÂ¢ÃÂÃÂ¥ÃÂÃÂ¼ÃÂÃÂ: {', '.join(str(i) for i in segment_hint.peak_indices) or '-'}",
                    f"ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂ´ÃÂÃÂ¢ÃÂÃÂ¥ÃÂÃÂ¼ÃÂÃÂ: {', '.join(str(i) for i in segment_hint.split_indices) or '-'}",
                    "",
                ]
            )
        if fit is None:
            lines.extend(["No fit results are available for the current spectrum.", "", "Please run a fit first."])
            self.fit_text.setPlainText("
".join(lines))
            if pg is not None:
                self._reset_plot(self.fit_nyquist_plot, True)
                self.fit_nyquist_plot.plot(
                    spectrum.z_real_ohm,
                    spectrum.minus_z_imag_ohm,
                    pen=pg.mkPen("#0071e3", width=1.6),
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(QColor("#5ac8fa")),
                )
                self._ensure_fit_plot_click_handler()
                if segment_hint is not None:
                    self._refresh_segment_preview_only(spectrum, segment_hint)
                else:
                    self._clear_segment_overlay()
                self._set_nyquist_axes(self.fit_nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
            return
        lines.extend([f"ÃÂÃÂ¦ÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ: {fit.model_label}", f"ÃÂÃÂ§ÃÂÃÂÃÂÃÂ¶ÃÂÃÂ¦ÃÂÃÂ? {fit.status.upper()}", ""])
        if fit.preprocess_actions:
            lines.append("ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ®ÃÂÃÂ©ÃÂÃÂ¢ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ§ÃÂÃÂ?")
            for action in fit.preprocess_actions:
                lines.append(f"  ÃÂÃÂ¢ÃÂÃÂ?{action}")
            lines.append("")
        if fit.fallback_from:
            lines.append(f"Fallback used: started from template {fit.fallback_from}")
            lines.append("")
        lines.append("Primary parameters")
        for name in ("Rs", "Rsei", "Rct"):
            if name in fit.parameters:
                lines.append(f"  {name} = {fit.parameters[name]:.8g}")
        error_lines = self._fit_error_lines(fit)
        if error_lines:
            lines.extend(["", "ÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂ»ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ¯ÃÂÃÂ¥ÃÂÃÂ·ÃÂÃÂ®ÃÂÃÂ§ÃÂÃÂÃÂÃÂ"])
            lines.extend(error_lines)
        diag_start = fit.message.find("[ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ­:")
        if diag_start >= 0:
            diag_text = fit.message[diag_start:]
            plain_message = fit.message[:diag_start].strip().rstrip(";").strip()
            lines.extend(["", "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ´ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ", f"  {plain_message.split(';', 1)[0].strip()}"])
            lines.extend(["", "ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ­ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ", f"  {diag_text}"])
        else:
            lines.extend(["", "ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ´ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ", f"  {fit.message.split(';', 1)[0].strip()}"])
        detail_lines.extend(
            [
                "",
                f"ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ´ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ´ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ: {fit.message}",
                "",
                "ÃÂÃÂ¦ÃÂÃÂ¬ÃÂÃÂ¡ÃÂÃÂ§ÃÂÃÂºÃÂÃÂ§ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°",
            ]
        )
        for name in ("Rsei_global", "Rct_global", "Wo_R", "Wo_T", "Wo_P", "CPE_T", "CPE_P", "Q1", "n1", "Q2", "n2"):
            if name in fit.parameters:
                detail_lines.append(f"  {name} = {fit.parameters[name]:.8g}")
        detail_lines.extend(
            [
                "",
                "ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ®ÃÂÃÂ¡",
        self.fit_text.setPlainText("
".join(lines + ["", "Advanced parameters"] + detail_lines))
                f"  aic = {fit.statistics.get('aic', float('nan')):.8g}",
                f"  bic = {fit.statistics.get('bic', float('nan')):.8g}",
                f"  chi2_reduced = {fit.statistics.get('chi2_reduced', float('nan')):.8g}",
            ]
        )
        self.fit_text.setPlainText("
".join(lines + ["", "ÃÂÃÂ©ÃÂÃÂ«ÃÂÃÂÃÂÃÂ§ÃÂÃÂºÃÂÃÂ§ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°"] + detail_lines))
        if pg is None:
            self._update_global_status()
            return
        self._reset_plot(self.fit_nyquist_plot, True)
        self._ensure_legend(self.fit_nyquist_plot)
        self.fit_nyquist_plot.plot(
            name="Experimental",
            spectrum.minus_z_imag_ohm,
            pen=pg.mkPen("#0071e3", width=1.6),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(QColor("#5ac8fa")),
            name="ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂµÃÂÃÂ",
        )
        self._ensure_fit_plot_click_handler()
        if segment_hint is not None:
            self._refresh_segment_preview_only(spectrum, segment_hint)
        else:
                name="Fitted",
        if fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
            self.fit_nyquist_plot.plot(
                fit.predicted_real_ohm,
                -fit.predicted_imag_ohm,
                pen=pg.mkPen("#34c759", width=2.1),
                name="ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ",
            )
            self._set_nyquist_axes(
                self.fit_nyquist_plot,
                np.concatenate([spectrum.z_real_ohm, fit.predicted_real_ohm]),
                np.concatenate([spectrum.minus_z_imag_ohm, -fit.predicted_imag_ohm]),
            )
        else:
            self._set_nyquist_axes(self.fit_nyquist_plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
        self._update_global_status()
    def _on_fit_plot_scene_clicked(self, ev) -> None:
        if ev.button() != Qt.LeftButton:
            return
        if not self._is_zview_template_selected():
            return
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None:
            return
        segment_hint = self._current_segment_hint(spectrum)
        pos = self.fit_nyquist_plot.plotItem.vb.mapSceneToView(ev.scenePos())
        x_clicked = pos.x()
        y_clicked = pos.y()
        dists = (spectrum.z_real_ohm - x_clicked)**2 + (spectrum.minus_z_imag_ohm - y_clicked)**2
        nearest_idx = int(np.argmin(dists))
        if segment_hint is None:
            self._move_segment_to_index(spectrum, "split1", nearest_idx, commit=False)
            return
        # Find the geometrically closest handle to the click point
        min_dist = float("inf")
        best_role = "split1"
        all_points = []
        for i, idx in enumerate(segment_hint.peak_indices):
            all_points.append((f"peak{i+1}", idx))
        for i, idx in enumerate(segment_hint.split_indices):
            all_points.append((f"split{i+1}", idx))
        for role, idx in all_points:
            hx = spectrum.z_real_ohm[idx]
            hy = spectrum.minus_z_imag_ohm[idx]
            d = (hx - x_clicked)**2 + (hy - y_clicked)**2
            if d < min_dist:
                min_dist = d
                best_role = role
        self._move_segment_to_index(spectrum, best_role, nearest_idx, commit=False)
    def _toggle_fit_detail(self) -> None:
        visible = not self.fit_detail_text.isVisible()
        self.fit_detail_text.setVisible(visible)
        self.fit_detail_btn.setText("Hide details" if visible else "Show details")
    def _fit_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        template_key = str(self.template_combo.currentData())
        segment_hint = self._current_segment_hint(spectrum)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fit = fit_spectrum(spectrum, template_key, segment_hint=segment_hint)
        finally:
            QApplication.restoreOverrideCursor()
        self.state.fits[(spectrum.display_name, template_key)] = fit
        self._sync_fit_to_batch(spectrum, fit)
        self._refresh_fit_view(spectrum)
    def _fit_batch(self) -> None:
        if not self.state.spectra:
            self._warn("Import spectra before running batch fitting.")
            return
        if self.state.batch_busy:
            self._warn("Batch fitting is already running.")
            return
        self.state.batch_summary = None
        self._refresh_batch_view()
        self.batch_progress.setValue(0)
        self.batch_progress_label.setText(f"ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂ 0 / {len(self.state.spectra)}")
        self._set_batch_busy(True)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchFitWorker(list(self.state.spectra))
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_thread.finished.connect(self._batch_thread.deleteLater)
        self._batch_thread.start()
    def _refresh_batch_view(self) -> None:
        summary = self.state.batch_summary
        if summary is None:
            self.batch_text.setPlainText("Batch summary will appear here." if not self.state.batch_busy else "Batch fitting is running...")
            self.batch_table.clearContents()
            self.batch_table.setRowCount(0)
            self.batch_table.setColumnCount(0)
            self._batch_plot_cache = {}
            if pg is not None:
                self._reset_plot(self.batch_plot, True)
            return
        warn_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "warn")
        high_error_count = sum(1 for item in summary.items if item.fit is not None and self._fit_has_high_error(item.fit))
        failed_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "failed")
        ok_count = sum(1 for item in summary.items if item.fit is not None and item.fit.status == "ok")
        fallback_count = sum(1 for item in summary.items if item.fit is not None and item.fit.fallback_from is not None)
        lines = [
            "Batch summary:",
            f"ÃÂÃÂ¨ÃÂÃÂ°ÃÂÃÂ±ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¾ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ©ÃÂÃÂÃÂÃÂ: {len(summary.items)}",
            f"ÃÂÃÂ¢ÃÂÃÂ?ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ: {ok_count}    ÃÂÃÂ¢ÃÂÃÂ?ÃÂÃÂ¨ÃÂÃÂ­ÃÂÃÂ¦ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ: {warn_count}    ÃÂÃÂ¢ÃÂÃÂ?ÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥: {failed_count}",
            f"ÃÂÃÂ°ÃÂÃÂÃÂÃÂÃÂÃÂ ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂºÃÂÃÂ§ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂ: {fallback_count}    ÃÂÃÂ°ÃÂÃÂÃÂÃÂÃÂÃÂ ÃÂÃÂ©ÃÂÃÂ«ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ¯ÃÂÃÂ¥ÃÂÃÂ·? {high_error_count}",
        ]
        self.batch_text.setPlainText("
".join(lines))
        names: list[str] = []
        for item in summary.items:
            if item.fit is None:
                continue
            for name in item.fit.parameters:
                if name not in names:
                    names.append(name)
        essential_names = [name for name in ("Rs", "Rsei", "Rct") if name in names]
        headers = [
            "Index", "Spectrum", "Fit status", "Primary uncertainty", "Message", "Fallback", "Warnings", "Model",
            "Rs_error_pct", "Rsei_error_pct", "Rct_error_pct", *essential_names,
        ]
        rows: list[list[str]] = []
        for index, item in enumerate(summary.items, start=1):
            fit = item.fit
            stats = fit.statistics if fit is not None else {}
            msg = ""
            if fit and fit.message:
                diag_pos = fit.message.find("[ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ­:")
                msg = fit.message[:diag_pos].strip().rstrip(";").strip() if diag_pos >= 0 else fit.message
                msg = msg.split(";", 1)[0]
            fallback_label = "R(QR)ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂºÃÂÃÂ§" if fit and fit.fallback_from else ""
            preprocess_label = "; ".join(fit.preprocess_actions) if fit and fit.preprocess_actions else ""
            row = [
                str(index), item.spectrum.display_name, fit.model_label if fit else "-",
                msg,
                fit.status if fit else "not_run", fallback_label, preprocess_label,
                self._fit_error_summary(fit) if fit else "",
                self._format_stat(stats.get("Rs_stderr_pct")),
                self._format_stat(stats.get("Rsei_stderr_pct")),
                self._format_stat(stats.get("Rct_stderr_pct")),
            ]
            row.extend("" if fit is None else f"{fit.parameters.get(name, float('nan')):.8g}" for name in essential_names)
            rows.append(row)
        self._fill_table(self.batch_table, headers, rows)
        for row_index, item in enumerate(summary.items):
            fit = item.fit
            if fit is None:
                continue
            if fit.status == "failed" or self._fit_has_high_error(fit):
                self._color_batch_row(row_index, QColor("#fff1f0"), QColor("#b42318"))
            elif fit.status == "warn":
                self._color_batch_row(row_index, QColor("#fff7e6"), QColor("#b54708"))
        self._rebuild_batch_plot_cache(summary)
        self._refresh_batch_plot()
        self._update_global_status()
    def _rebuild_batch_plot_cache(self, summary: BatchSummary) -> None:
        x = np.arange(1, len(summary.items) + 1, dtype=float)
        cache: dict[str, dict[str, object]] = {}
        for name, color, error_key in (
            ("Rs", "#0071e3", "Rs_stderr_pct"),
            ("Rsei", "#ff9500", "Rsei_stderr_pct"),
            ("Rct", "#34c759", "Rct_stderr_pct"),
        ):
            values = np.asarray([np.nan if item.fit is None else item.fit.parameters.get(name, np.nan) for item in summary.items], dtype=float)
            ok_x: list[float] = []
            ok_y: list[float] = []
            bad_x: list[float] = []
            bad_y: list[float] = []
            for idx, item in enumerate(summary.items, start=1):
                fit = item.fit
                if fit is None:
                    continue
                value = fit.parameters.get(name)
                if value is None or not np.isfinite(float(value)):
                    continue
                err = fit.statistics.get(error_key)
                if err is not None and (not np.isfinite(float(err)) or float(err) > 20.0):
                    bad_x.append(float(idx))
                    bad_y.append(float(value))
                else:
                    ok_x.append(float(idx))
                    ok_y.append(float(value))
            cache[name] = {
                "x": x,
                "values": values,
                "ok_x": ok_x,
                "ok_y": ok_y,
                "bad_x": bad_x,
                "bad_y": bad_y,
                "color": color,
            }
        self._batch_plot_cache = cache
    def _refresh_batch_plot(self) -> None:
        if pg is None:
            return
        if self.state.batch_summary is None:
            self._reset_plot(self.batch_plot, True)
            return
        self.batch_plot.setUpdatesEnabled(False)
        try:
            self._reset_plot(self.batch_plot, True)
            self._ensure_legend(self.batch_plot)
            enabled = {
                "Rs": self.trend_rs_check.isChecked(),
                "Rsei": self.trend_rsei_check.isChecked(),
                "Rct": self.trend_rct_check.isChecked(),
            }
            for name in ("Rs", "Rsei", "Rct"):
                if not enabled[name]:
                    continue
                series = self._batch_plot_cache.get(name)
                if not series:
                    continue
                values = np.asarray(series["values"], dtype=float)
                if not np.any(np.isfinite(values)):
                    continue
                color = str(series["color"])
                self.batch_plot.plot(np.asarray(series["x"], dtype=float), values, pen=pg.mkPen(color=color, width=2), name=name)
                ok_x = list(series["ok_x"])
                ok_y = list(series["ok_y"])
                bad_x = list(series["bad_x"])
                bad_y = list(series["bad_y"])
                if ok_x:
                    scatter_ok = pg.ScatterPlotItem(ok_x, ok_y, symbol="o", size=5, brush=pg.mkBrush(QColor(color)), pen=pg.mkPen(color))
                    scatter_ok.sigClicked.connect(self._on_trend_point_clicked)
                    self.batch_plot.addItem(scatter_ok)
                if bad_x:
                    scatter_bad = pg.ScatterPlotItem(bad_x, bad_y, symbol="o", size=7, brush=pg.mkBrush(255, 255, 255, 0), pen=pg.mkPen(color, width=2))
                    scatter_bad.sigClicked.connect(self._on_trend_point_clicked)
                    self.batch_plot.addItem(scatter_bad)
        finally:
            self.batch_plot.setUpdatesEnabled(True)
    def _on_trend_point_clicked(self, scatter_item, points, ev=None) -> None:
        if not points:
            return
        try:
            clicked_x = int(round(points[0].pos().x()))
        except Exception:
            return
        summary = self.state.batch_summary
        if summary is None:
            return
        row_index = clicked_x - 1
        if 0 <= row_index < len(summary.items):
            item = summary.items[row_index]
            self._set_current_index(row_index, refresh_page=False)
            self.batch_table.selectRow(row_index)
            self.batch_table.scrollToItem(self.batch_table.item(row_index, 0))
            self.batch_progress_label.setText(
                f"ÃÂÃÂ¥ÃÂÃÂ·ÃÂÃÂ²ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂ­ÃÂÃÂ§ÃÂÃÂ¬?{clicked_x} ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¡ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{item.spectrum.display_name}"
            )
    def _on_batch_table_selection_changed(self) -> None:
        if self.state.batch_summary is None:
            return
        row = self.batch_table.currentRow()
        if row < 0 or row >= len(self.state.spectra):
            return
        self._set_current_index(row, refresh_page=False)
    def _browse_matlab_exe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ© matlab.exe", self.matlab_exe_edit.text() or str(Path.cwd()), "Executable (*.exe)")
        if path:
            self.matlab_exe_edit.setText(path)
    def _browse_drttools_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ© DRTtools ÃÂÃÂ§ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂ", self.drttools_dir_edit.text() or str(Path.cwd()))
        if path:
            self.drttools_dir_edit.setText(path)
    def _current_matlab_drt_config(self) -> MatlabDrtConfig:
        return MatlabDrtConfig(
            matlab_exe=self.matlab_exe_edit.text().strip(),
            drttools_dir=self.drttools_dir_edit.text().strip(),
            method_tag=str(self.matlab_method_combo.currentData()),
            drt_type=int(self.matlab_drt_type_combo.currentData()),
            lambda_value=float(self.matlab_lambda_edit.text().strip()),
            coeff_value=float(self.matlab_coeff_edit.text().strip()),
            inductance_mode=int(self.matlab_inductance_combo.currentData()),
        )
    def _set_drt_busy(self, busy: bool) -> None:
        self.state.drt_busy = busy
        for button in (self.run_drt_current_btn, self.run_drt_batch_btn):
            button.setEnabled(not busy)
        if busy:
            self.fit_metric["value"].setText("DRT running")
        else:
            self._update_global_status()
    def _set_batch_busy(self, busy: bool) -> None:
        self.state.batch_busy = busy
        for button in (self.fit_batch_btn, self.export_batch_btn):
            button.setEnabled(not busy)
        if busy:
        self.batch_progress_label.setText(f"Processing {index} / {total}: {display_name}")
        else:
            self._update_global_status()
    def _on_batch_progress(self, index: int, total: int, display_name: str) -> None:
        percent = int(index * 100 / max(total, 1))
            self.batch_progress_label.setText("Batch fitting failed")
        self.batch_progress_label.setText(f"ÃÂÃÂ¦ÃÂÃÂ­ÃÂÃÂ£ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ§ÃÂÃÂÃÂÃÂ {index} / {total}ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{display_name}")
    def _on_batch_finished(self, summary: BatchSummary | None, error: Exception | None) -> None:
        self._set_batch_busy(False)
        if error is not None:
            self.batch_progress.setValue(0)
            self.batch_progress_label.setText("ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥")
            self._warn(f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{error}")
        elif summary is not None:
            self.batch_progress_label.setText(f"Batch fitting complete: {len(summary.items)} / {len(summary.items)}")
            for item in summary.items:
                self.state.qualities[item.spectrum.display_name] = item.quality
                if item.fit is not None:
                    self.state.fits[(item.spectrum.display_name, item.fit.model_key)] = item.fit
            self.batch_progress.setValue(100)
            self.batch_progress_label.setText(f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{len(summary.items)} / {len(summary.items)}")
            self._refresh_batch_view()
        self._update_global_status()
        self._batch_worker = None
        self._batch_thread = None
    def _sync_fit_to_batch(self, spectrum: SpectrumData, fit: FitOutcome) -> None:
        summary = self.state.batch_summary
        if summary is None:
            return
        for item in summary.items:
            if item.spectrum.display_name == spectrum.display_name:
                item.fit = fit
                break
        if self._current_interface_id() == "batchInterface":
            self._refresh_batch_view()
        else:
            self._update_global_status()
    def _start_matlab_drt(self, spectra: list[SpectrumData], export_dir: Path) -> None:
        if self.state.drt_busy:
            self._warn("MATLAB DRT is already running.")
            return
        try:
            config = self._current_matlab_drt_config()
        except ValueError as exc:
            self._warn(f"MATLAB DRT ÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{exc}")
            return
        self._last_drt_spectra = list(spectra)
        self.matlab_status.setPlainText(f"MATLAB DRT job started
Spectra: {len(spectra)}
Output: {export_dir / 'results'}")
        self.matlab_log.setPlainText("MATLAB DRT job created. Waiting for completion...")
        self._set_drt_busy(True)
        self._matlab_thread = QThread(self)
        self._matlab_worker = MatlabDrtWorker(config, spectra, export_dir)
        self._matlab_worker.moveToThread(self._matlab_thread)
        self._matlab_thread.started.connect(self._matlab_worker.run)
        self._matlab_worker.finished.connect(self._on_matlab_drt_finished)
        self._matlab_worker.finished.connect(self._matlab_thread.quit)
        self._matlab_worker.finished.connect(self._matlab_worker.deleteLater)
        self._matlab_thread.finished.connect(self._matlab_thread.deleteLater)
        self._matlab_thread.start()
    def _on_matlab_drt_finished(self, result: MatlabDrtResult | None, error: Exception | None) -> None:
        self._set_drt_busy(False)
        if error is not None:
            self.fit_metric["value"].setText("DRT ÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥")
            self.matlab_log.setPlainText(f"MATLAB DRT ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ
{error}")
            self._warn(f"MATLAB DRT ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{error}")
        else:
            self._show_matlab_drt_result(result)
        self._matlab_worker = None
        self._matlab_thread = None
    def _run_matlab_drt_current(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ© MATLAB DRT ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂ", str(Path.cwd()))
        if folder:
            self._start_matlab_drt([spectrum], Path(folder) / f"{spectrum.metadata.file_path.stem}_matlab_drt")
    def _run_matlab_drt_batch(self) -> None:
        if not self.state.spectra:
            self._warn("Import spectra before running MATLAB DRT.")
            return
        folder = QFileDialog.getExistingDirectory(self, "ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ© MATLAB DRT ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ©ÃÂÃÂÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂ", str(Path.cwd()))
        if folder:
            self._start_matlab_drt(self.state.spectra, Path(folder) / "matlab_drt_batch")
    def _show_matlab_drt_result(self, result: MatlabDrtResult | None) -> None:
        if result is None:
            return
        self._last_matlab_result = result
        try:
            from eismaster.exporters import write_drt_only_export
            write_drt_only_export(result.output_dir / "drt_matrix.csv", self._last_drt_spectra, result.output_dir, fmt="csv")
        except Exception:
            pass
        lines = [
            f"ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ½ÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¤ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{' '.join(result.command)}",
            "",
        ]
        self.matlab_status.setPlainText(
            "
".join(
                [
                    f"ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ§ÃÂÃÂ ÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{result.returncode}",
                    f"ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{result.output_dir}",
                    f"Staging directory: {result.staging_dir}",
                    f"ÃÂÃÂ§ÃÂÃÂ»ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¶ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{len(result.output_files)}",
                    "Output files:",
                    *[f"  {path.name}" for path in result.output_files[:12]],
                ]
            )
        )
        if result.stdout.strip():
            lines.extend(["", "Standard output:", result.stdout.strip()])
        if result.stderr.strip():
            lines.extend(["", "Standard error:", result.stderr.strip()])
        self.matlab_log.setPlainText("
".join(lines))
        if result.returncode == 0:
            self.fit_metric["value"].setText("DRT complete")
            QMessageBox.information(self, "MATLAB DRT ÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ", f"ÃÂÃÂ¥ÃÂÃÂ·ÃÂÃÂ²ÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂ?{len(result.output_files)} ÃÂÃÂ¤ÃÂÃÂ¸?DRT ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¶ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¹ÃÂÃÂ¶ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂÃÂÃÂ©ÃÂÃÂ©ÃÂÃÂÃÂÃÂµÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¶ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ
{result.output_dir}")
        else:
            self.fit_metric["value"].setText("DRT ÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥")
            self._warn(f"MATLAB DRT ÃÂÃÂ¨ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂ±ÃÂÃÂ¨ÃÂÃÂ´ÃÂÃÂ¥ÃÂÃÂ£ÃÂÃÂÃÂÃÂ
ÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂ·ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ§ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¸ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¹ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¥ÃÂÃÂ¿ÃÂÃÂÃÂÃÂ£ÃÂÃÂÃÂÃÂ
ÃÂÃÂ¨ÃÂÃÂ¾ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ§ÃÂÃÂÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ{result.output_dir}")
    def _export_current_bundle(self) -> None:
        spectrum = self._current_spectrum()
        if spectrum is None:
            return
        output_path, fmt = self._select_export_target("ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ¥ÃÂÃÂ½ÃÂÃÂÃÂÃÂ¥ÃÂÃÂÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ®", spectrum.metadata.file_path.stem)
        if output_path is None:
            return
        quality = self.state.qualities.get(spectrum.display_name) or assess_spectrum_quality(spectrum, run_kk=False)
        self.state.qualities[spectrum.display_name] = quality
        fit = self.state.fits.get((spectrum.display_name, str(self.template_combo.currentData())))
        drt_dir = self._last_matlab_result.output_dir if self._last_matlab_result is not None else None
        paths = export_spectrum_bundle(output_path, spectrum, fit=fit, quality=quality, fmt=fmt, drt_source_dir=drt_dir)
        QMessageBox.information(self, "ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ¥ÃÂÃÂ®ÃÂÃÂÃÂÃÂ¦ÃÂÃÂÃÂÃÂ", f"ÃÂÃÂ¦ÃÂÃÂÃÂÃÂÃÂÃÂ¤ÃÂÃÂ»ÃÂÃÂ¶ÃÂÃÂ¥ÃÂÃÂ·ÃÂÃÂ²ÃÂÃÂ¥ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ¥ÃÂÃÂÃÂÃÂºÃÂÃÂ¥ÃÂÃÂÃÂÃÂ°ÃÂÃÂ¯ÃÂÃÂ¼ÃÂÃÂ
{self._export_result_location(paths)}")
    def _export_batch_bundle(self) -> None:
        if self.state.batch_summary is None:
            self._warn("Run batch fitting before exporting the batch bundle.")
            return
        output_path, fmt = self._select_export_target("Export batch summary", "batch_export")
        if output_path is None:
            return
        drt_dir = self._last_matlab_result.output_dir if self._last_matlab_result is not None else None
        paths = export_batch_summary(output_path, self.state.batch_summary, fmt=fmt, drt_source_dir=drt_dir)
        QMessageBox.information(self, "Export complete", f"Files written to:
{self._export_result_location(paths)}")
    def _select_export_target(self, title: str, default_stem: str) -> tuple[Path | None, str]:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            str(Path.cwd() / default_stem),
            "Excel Workbook (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not path:
            return None, "txt"
        selected_filter = selected_filter.lower()
        if "xlsx" in selected_filter or path.lower().endswith(".xlsx"):
            return Path(path), "xlsx"
        if "csv" in selected_filter or path.lower().endswith(".csv"):
            return Path(path), "csv"
        return Path(path), "txt"
    def _export_result_location(self, paths: dict[str, Path]) -> Path:
        first = next(iter(paths.values()))
        return first if first.suffix.lower() == ".xlsx" else first.parent
    def _set_nyquist_axes(self, plot, x_values: np.ndarray, y_values: np.ndarray) -> None:
        if pg is not None:
            class InteractiveViewBox(pg.ViewBox):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            x = np.asarray(x_values, dtype=float)
            y = np.asarray(y_values, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                return
            x = x[mask]
            y = y[mask]
            x_min = min(float(np.min(x)), 0.0)
            y_min = min(float(np.min(y)), 0.0)
            x_max = float(np.max(x))
            y_max = float(np.max(y))
            span = max(x_max - x_min, y_max - y_min, 1.0)
            pad = span * 0.08
            plot.setAspectLocked(False)
            plot.setXRange(x_min - pad, x_min + span + pad, padding=0.0)
            plot.setYRange(y_min - pad, y_min + span + pad, padding=0.0)
    def _reset_current_nyquist(self, plot) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or plot is None:
            return
        fit = None
        if plot is getattr(self, "fit_plot", None):
            fit = self.state.fits.get((spectrum.display_name, str(self.template_combo.currentData())))
        if fit is not None and fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
            self._set_nyquist_axes(
                plot,
                np.concatenate([spectrum.z_real_ohm, fit.predicted_real_ohm]),
                np.concatenate([spectrum.minus_z_imag_ohm, -fit.predicted_imag_ohm]),
            )
        else:
            self._set_nyquist_axes(plot, spectrum.z_real_ohm, spectrum.minus_z_imag_ohm)
    def _reset_bode(self) -> None:
        spectrum = self._current_spectrum(silent=True)
        if spectrum is None or pg is None:
            return
        x_min = float(np.min(spectrum.freq_hz))
        x_max = float(np.max(spectrum.freq_hz))
        self.bode_mag_plot.setXRange(np.log10(x_min), np.log10(x_max), padding=0.0)
        self.bode_phase_plot.setXRange(np.log10(x_min), np.log10(x_max), padding=0.0)
    def _reset_plot(self, plot, keep_legend: bool = False) -> None:
        if pg is None:
            return
        plot.clear()
        if not keep_legend and plot.plotItem.legend:
            plot.plotItem.legend.setParentItem(None)
            plot.plotItem.legend = None
    def _ensure_legend(self, plot) -> None:
        if pg is not None and plot.plotItem.legend is None:
            plot.addLegend(offset=(12, 12))
    def _plot_segment_preview(self, spectrum: SpectrumData, segment_hint: SegmentDetection) -> None:
        self._render_segment_overlay(spectrum, segment_hint)
    def _fit_error_lines(self, fit: FitOutcome) -> list[str]:
        lines: list[str] = []
        for key, label in self._primary_error_pairs(fit):
            value = fit.statistics.get(key)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                lines.append(f"  {label}: ÃÂÃÂ¦ÃÂÃÂÃÂÃÂ ÃÂÃÂ¦ÃÂÃÂ³ÃÂÃÂÃÂÃÂ¨ÃÂÃÂ¯ÃÂÃÂÃÂÃÂ¤ÃÂÃÂ¼ÃÂÃÂ°")
            elif value > 20.0:
                lines.append(f"  {label}: {value:.2f}%  ÃÂÃÂ¥ÃÂÃÂ»ÃÂÃÂºÃÂÃÂ¨ÃÂÃÂ®ÃÂÃÂ®ÃÂÃÂ¥ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ¦ÃÂÃÂ ÃÂÃÂ¸")
            elif value > 10.0:
                lines.append(f"  {label}: {value:.2f}%  ÃÂÃÂ¨ÃÂÃÂ­ÃÂÃÂ¦ÃÂÃÂ¥ÃÂÃÂÃÂÃÂ")
            else:
                lines.append(f"  {label}: {value:.2f}%")
        return lines
    def _fit_has_high_error(self, fit: FitOutcome, threshold: float = 20.0) -> bool:
        return any(value > threshold for _, value in self._primary_error_values(fit))
    def _fit_error_summary(self, fit: FitOutcome | None) -> str:
        if fit is None:
            return ""
        values = self._primary_error_values(fit)
        if not values:
            return "n/a"
        high_parts = [f"{label} {value:.1f}%" for label, value in values if value > 20.0]
        if high_parts:
            return "High: " + ", ".join(high_parts)
        return ", ".join(f"{label} {value:.1f}%" for label, value in values)
    def _primary_error_pairs(self, fit: FitOutcome) -> list[tuple[str, str]]:
        if fit.model_key == "zview_double_rq_qrwo":
            return [("Rs_stderr_pct", "Rs"), ("Rsei_stderr_pct", "Rsei"), ("Rct_stderr_pct", "Rct")]
        return [("Rs_stderr_pct", "Rs"), ("Rct_stderr_pct", "Rct")]
    def _primary_error_values(self, fit: FitOutcome) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        for key, label in self._primary_error_pairs(fit):
            value = fit.statistics.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isfinite(value):
                values.append((label, value))
        return values
    def _fill_table(self, table, headers: list[str], rows: list[list[str]]) -> None:
        table.setUpdatesEnabled(False)
        table.clearContents()
        table.setSortingEnabled(False)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(value))
        if table is self.data_table:
            table.resizeColumnsToContents()
        elif table is self.batch_table:
            self._apply_batch_table_widths()
        table.setUpdatesEnabled(True)
    def _apply_batch_table_widths(self) -> None:
        widths = {
            0: 68,
            1: 220,
            2: 110,
            3: 250,
            4: 90,
            5: 90,
            6: 120,
            7: 160,
            8: 90,
            9: 90,
            10: 90,
            11: 90,
            12: 90,
            13: 90,
        }
        header = self.batch_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        for column, width in widths.items():
            if column < self.batch_table.columnCount():
                self.batch_table.setColumnWidth(column, width)
    def _color_batch_row(self, row_index: int, background: QColor, foreground: QColor) -> None:
        for column in range(self.batch_table.columnCount()):
            item = self.batch_table.item(row_index, column)
            if item is not None:
                item.setBackground(background)
                item.setForeground(foreground)
    def _format_stat(self, value) -> str:
        if value is None:
            return ""
        value = float(value)
        if not np.isfinite(value):
            return "nan"
        return f"{value:.2f}"
    def _current_spectrum(self, silent: bool = False) -> SpectrumData | None:
        row = self.state.current_index
        if 0 <= row < len(self.state.spectra):
            return self.state.spectra[row]
        if not silent:
            self._warn("Please select a spectrum first.")
        return None
    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, "EISMaster", message)
    def _open_circuit_builder(self) -> None:
        try:
            from eismaster.ui.circuit_builder.graphics import CircuitBuilderWindow
            self.circuit_builder = CircuitBuilderWindow(self)
            self.circuit_builder.show()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Graph-based logic editor is under construction.")
