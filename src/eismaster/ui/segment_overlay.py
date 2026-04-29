from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import Qt

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

from eismaster.models import SpectrumData


@dataclass(frozen=True)
class SegmentBoundaries:
    requested_mode: Literal["single", "double", "auto", "auto_detect"]
    resolved_mode: Literal["single", "double"]
    split_indices: tuple[int, ...]
    peak_indices: tuple[int, ...] = ()
    source: Literal["auto", "manual"] = "manual"


class SegmentOverlay:
    """Render segment feedback on the Nyquist fit plot."""

    ARC1_COLOR = "#FFB74D"
    ARC2_COLOR = "#4DD0E1"
    TAIL_COLOR = "#6B7280"
    GUIDE_COLOR = "#8B8BA7"
    HANDLE_FILL = "#F8FAFC"

    def __init__(self, plot_widget) -> None:
        self._plot = plot_widget
        self._items: list[object] = []

    def clear(self) -> None:
        if pg is None or self._plot is None:
            self._items = []
            return
        for item in self._items:
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._items = []

    def render(
        self,
        spectrum: SpectrumData,
        boundaries: SegmentBoundaries | None,
    ) -> None:
        self.clear()
        if pg is None or self._plot is None or boundaries is None:
            return

        for start, stop, color, label in self._segment_specs(spectrum, boundaries):
            if stop < start:
                continue
            x = spectrum.z_real_ohm[start : stop + 1]
            y = spectrum.minus_z_imag_ohm[start : stop + 1]
            if x.size == 0:
                continue
            curve = self._plot.plot(
                x,
                y,
                pen=pg.mkPen(color, width=3),
                name=label,
            )
            scatter = pg.ScatterPlotItem(
                x,
                y,
                symbol="o",
                size=6,
                brush=pg.mkBrush(color),
                pen=pg.mkPen(color, width=1.2),
            )
            self._plot.addItem(scatter)
            self._items.extend([curve, scatter])

        for split in boundaries.split_indices:
            if not 0 <= split < spectrum.n_points:
                continue
            x_split = float(spectrum.z_real_ohm[split])
            y_split = float(spectrum.minus_z_imag_ohm[split])
            handle = self._plot.plot(
                [x_split],
                [y_split],
                pen=None,
                symbol="o",
                symbolSize=12,
                symbolBrush=pg.mkBrush(self.HANDLE_FILL),
                symbolPen=pg.mkPen(self.GUIDE_COLOR, width=2),
            )
            self._items.append(handle)

    def _segment_specs(
        self,
        spectrum: SpectrumData,
        boundaries: SegmentBoundaries,
    ) -> list[tuple[int, int, str, str]]:
        n = spectrum.n_points
        if n <= 0:
            return []
        if boundaries.resolved_mode == "double" and len(boundaries.split_indices) >= 2:
            s1, s2 = boundaries.split_indices[:2]
            return [
                (0, s1, self.ARC1_COLOR, "半圆1区域"),
                (s1 + 1, s2, self.ARC2_COLOR, "半圆2区域"),
                (s2 + 1, n - 1, self.TAIL_COLOR, "尾部区域"),
            ]
        if boundaries.split_indices:
            split = boundaries.split_indices[0]
            return [
                (0, split, self.ARC1_COLOR, "半圆区域"),
                (split + 1, n - 1, self.TAIL_COLOR, "尾部区域"),
            ]
        return [(0, n - 1, self.TAIL_COLOR, "数据区域")]
