from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, PushButton, TableWidget
from qfluentwidgets import FluentIcon as FIF

from eismaster.cc_curves import CCCurve, merge_cc_segments
from eismaster.cc_exporters import (
    build_cc_curve_table,
    build_cc_segment_summary,
    export_cc_curves,
)
from eismaster.io.cc import load_cc_segments_from_folder

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None


class CCCurvePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CC曲线")
        self.curves: list[CCCurve] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        action_row = QHBoxLayout()
        self.select_folder_btn = PrimaryPushButton(FIF.FOLDER, "选择 CC 文件夹")
        self.select_folder_btn.clicked.connect(self._choose_folder)
        self.export_btn = PushButton(FIF.SHARE, "导出 CC 曲线")
        self.export_btn.clicked.connect(self._choose_export_target)
        action_row.addWidget(self.select_folder_btn)
        action_row.addWidget(self.export_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        layout.addWidget(CaptionLabel("读取 CP 或 ISTEP/Multi-Current Steps 文件；EIS、静置和不完整文件不会进入曲线。"))
        self.status_label = BodyLabel("尚未加载 CC 文件。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        if pg is None:
            raise RuntimeError("pyqtgraph is required for the CC curve page")
        self.plot = pg.PlotWidget(title="时间–电压曲线")
        self.plot.setLabel("bottom", "时间", units="s")
        self.plot.setLabel("left", "电压", units="V")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend()
        layout.addWidget(self.plot, 2)

        self.curve_table = self._table()
        self.segment_table = self._table()
        layout.addWidget(BodyLabel("曲线明细"))
        layout.addWidget(self.curve_table, 1)
        layout.addWidget(BodyLabel("分段汇总"))
        layout.addWidget(self.segment_table, 1)

    def load_folder(self, folder: str | Path) -> None:
        try:
            segments = load_cc_segments_from_folder(folder)
            self.curves = merge_cc_segments(segments)
        except Exception as exc:
            self.curves = []
            self._clear_views()
            self.status_label.setText(f"加载失败：{exc}")
            return

        self._refresh_views()
        point_count = sum(curve.n_points for curve in self.curves)
        self.status_label.setText(f"已加载 {len(segments)} 个 CC 分段，共 {point_count} 个数据点。")

    def export_current(self, target: str | Path, fmt: str) -> dict[str, Path]:
        paths = export_cc_curves(target, self.curves, fmt=fmt)
        self.status_label.setText(f"已导出 CC 曲线：{paths['curve']}")
        return paths

    def _choose_folder(self) -> None:
        default = Path.cwd() / "L"
        folder = QFileDialog.getExistingDirectory(self, "选择 CC 文件夹", str(default if default.exists() else Path.cwd()))
        if folder:
            self.load_folder(folder)

    def _choose_export_target(self) -> None:
        if not self.curves:
            self.status_label.setText("请先加载 CC 文件。")
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出 CC 曲线",
            str(Path.cwd() / "cc_curve.xlsx"),
            "Excel (*.xlsx);;Text (*.txt)",
        )
        if not path:
            return
        fmt = "xlsx" if "xlsx" in selected_filter.lower() else "txt"
        self.export_current(path, fmt)

    def _refresh_views(self) -> None:
        self._clear_views()
        colors = ["#4FC3F7", "#FF8A65", "#A5D6A7", "#CE93D8"]
        curve_table = build_cc_curve_table(self.curves)
        segment_table = build_cc_segment_summary(self.curves)
        self._fill_table(self.curve_table, curve_table)
        self._fill_table(self.segment_table, segment_table)
        for index, curve in enumerate(self.curves):
            self.plot.plot(
                curve.total_time_s,
                curve.voltage_v,
                pen=pg.mkPen(colors[index % len(colors)], width=2),
                name=f"{curve.sample_id} {curve.mode}",
            )
        self.plot.autoRange()

    def _clear_views(self) -> None:
        self.plot.clear()
        self.plot.addLegend()
        self.curve_table.clearContents()
        self.curve_table.setRowCount(0)
        self.segment_table.clearContents()
        self.segment_table.setRowCount(0)

    @staticmethod
    def _table() -> TableWidget:
        table = TableWidget()
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        return table

    @staticmethod
    def _fill_table(table: TableWidget, frame: pd.DataFrame) -> None:
        table.setUpdatesEnabled(False)
        table.clearContents()
        table.setColumnCount(len(frame.columns))
        table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        table.setRowCount(len(frame))
        for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
            for column_index, value in enumerate(row):
                text = f"{value:.8g}" if isinstance(value, float) else str(value)
                table.setItem(row_index, column_index, QTableWidgetItem(text))
        table.setUpdatesEnabled(True)
