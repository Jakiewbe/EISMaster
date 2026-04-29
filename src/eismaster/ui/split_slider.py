"""Split-point slider for arc/tail boundary selection.

Supports two modes:
  - "single": one thumb = split point between arc and tail
  - "double": two thumbs = split1 (arc1/arc2) and split2 (arc2/tail)
"""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt, QSize, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QMouseEvent, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import QWidget, QToolTip


class SplitSlider(QWidget):
    """分界点滑块（单/双模式）。

    Signals:
        valueChanged:  Emitted with (split,) in single mode.
        valuesChanged: Emitted with (split1, split2) in double mode.
    """

    valueChanged = Signal(int)
    valuesChanged = Signal(int, int)

    # ── Visual constants ────────────────────────────────────────────
    _TRACK_HEIGHT = 4
    _THUMB_RADIUS = 8
    _TICK_MAJOR_HEIGHT = 8
    _TICK_MINOR_HEIGHT = 4
    _LABEL_FONT_SIZE = 8
    _MARGIN_H = 10
    _MARGIN_TOP = 18
    _MARGIN_BOTTOM = 4

    # Colors
    _COLOR_TRACK_BG = QColor("#2A2A30")
    _COLOR_ARC = QColor("#30d158")       # arc region highlight
    _COLOR_TAIL = QColor("#2A2A30")      # tail region (same as bg)
    _COLOR_THUMB_NORMAL = QColor("#E0E0E0")
    _COLOR_THUMB_HOVER = QColor("#FFFFFF")
    _COLOR_THUMB_BORDER = QColor("#555566")
    _COLOR_THUMB_GLOW = QColor(48, 209, 88, 80)
    _COLOR_TICK_MAJOR = QColor("#666680")
    _COLOR_TICK_MINOR = QColor("#44445A")
    _COLOR_LABEL = QColor("#8888A0")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = "single"   # "single" | "double"
        self._min = 0
        self._max = 100
        self._split = 50
        self._split2 = 75
        self._drag = None       # None | "split" | "split2"
        self._hover = None
        self._total_height = self._MARGIN_TOP + self._TRACK_HEIGHT + self._MARGIN_BOTTOM + 2 * self._THUMB_RADIUS
        self.setFixedHeight(self._total_height)
        self.setMouseTracking(True)
        self._label_font = QFont("Segoe UI, Microsoft YaHei", self._LABEL_FONT_SIZE)
        self._label_font.setPixelSize(self._LABEL_FONT_SIZE + 2)

    # ── Public API ──────────────────────────────────────────────────

    def setMode(self, mode: str) -> None:
        """Switch between 'single' (one split) and 'double' (two splits)."""
        if mode not in {"single", "double"}:
            raise ValueError(f"Unsupported split slider mode: {mode}")
        self._mode = mode
        if self._mode == "double" and self._split2 <= self._split:
            self._split2 = min(self._max, self._split + 1)
            if self._split2 <= self._split:
                self._split = max(self._min, self._max - 1)
                self._split2 = self._max
        self._drag = None
        self._hover = None
        self.update()

    def setRange(self, minimum: int, maximum: int) -> None:
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        self._min = minimum
        self._max = maximum
        if self._split < minimum:
            self._split = minimum
        if self._split > maximum:
            self._split = maximum
        if self._split2 < minimum:
            self._split2 = minimum
        if self._split2 > maximum:
            self._split2 = maximum
        if self._mode == "double" and maximum > minimum:
            if self._split >= maximum:
                self._split = maximum - 1
            if self._split2 <= self._split:
                self._split2 = min(maximum, self._split + 1)
        self.update()

    def setValue(self, split: int) -> None:
        split = max(self._min, min(split, self._max))
        if split == self._split:
            return
        self._split = split
        self.valueChanged.emit(self._split)
        self.update()

    def setValues(self, split1: int, split2: int) -> None:
        split1 = max(self._min, min(split1, self._max))
        split2 = max(self._min, min(split2, self._max))
        if split1 > split2:
            split1, split2 = split2, split1
        if self._max > self._min and split1 == split2:
            if split2 < self._max:
                split2 += 1
            else:
                split1 -= 1
        if split1 == self._split and split2 == self._split2:
            return
        self._split = split1
        self._split2 = split2
        self.valuesChanged.emit(self._split, self._split2)
        self.update()

    def value(self) -> int:
        return self._split

    def values(self) -> tuple[int, int]:
        return self._split, self._split2

    # ── Geometry helpers ────────────────────────────────────────────

    def _track_rect(self) -> QRect:
        w = self.width()
        track_y = self._MARGIN_TOP + self._THUMB_RADIUS - self._TRACK_HEIGHT // 2
        return QRect(self._MARGIN_H, track_y, max(w - 2 * self._MARGIN_H, 1), self._TRACK_HEIGHT)

    def _thumb_center_y(self) -> int:
        return self._MARGIN_TOP + self._THUMB_RADIUS

    def _thumb_pos(self, value: int) -> QPoint:
        track = self._track_rect()
        span = self._max - self._min
        if span <= 0:
            return QPoint(track.x(), self._thumb_center_y())
        frac = (value - self._min) / span
        x = track.x() + int(frac * track.width())
        return QPoint(x, self._thumb_center_y())

    def _hit_thumb(self, pos: QPoint) -> str | None:
        r = self._THUMB_RADIUS + 4
        thumbs = [("split", self._split)]
        if self._mode == "double":
            thumbs.append(("split2", self._split2))
        for role, val in thumbs:
            tp = self._thumb_pos(val)
            if (pos.x() - tp.x()) ** 2 + (pos.y() - tp.y()) ** 2 <= r ** 2:
                return role
        return None

    def _value_for_x(self, x: int) -> int:
        track = self._track_rect()
        span = self._max - self._min
        if span <= 0:
            return self._min
        frac = (x - track.x()) / max(track.width(), 1)
        frac = max(0.0, min(1.0, frac))
        return self._min + int(round(frac * span))

    # ── Tick calculation ────────────────────────────────────────────

    def _compute_ticks(self) -> tuple[list[int], list[int]]:
        span = self._max - self._min
        if span <= 0:
            return [], []
        major_values = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            major_values.append(self._min + int(round(frac * span)))
        major_values = sorted(set(major_values))
        minor_values = []
        if span >= 10:
            for frac_pct in range(10, 100, 10):
                frac = frac_pct / 100.0
                val = self._min + int(round(frac * span))
                if val not in major_values:
                    minor_values.append(val)
        return major_values, minor_values

    # ── Events ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.pos()
        thumb = self._hit_thumb(pos)
        if thumb:
            self._drag = thumb
            event.accept()
            return
        # click on track → move closest thumb
        split_pos = self._thumb_pos(self._split)
        d_split = (pos.x() - split_pos.x()) ** 2 + (pos.y() - split_pos.y()) ** 2
        if self._mode == "double":
            split2_pos = self._thumb_pos(self._split2)
            d_split2 = (pos.x() - split2_pos.x()) ** 2 + (pos.y() - split2_pos.y()) ** 2
            if d_split2 < d_split:
                self._drag = "split2"
            else:
                self._drag = "split"
        else:
            self._drag = "split"
        self._move_to_pos(pos)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        if self._drag:
            self._move_to_pos(pos)
            val = self._value_for_x(pos.x())
            label = "split2" if self._drag == "split2" else "分界"
            QToolTip.showText(event.globalPosition().toPoint(), f"{label}: {val}", self)
            event.accept()
            return
        hover = self._hit_thumb(pos)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None
        QToolTip.hideText()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        super().leaveEvent(event)

    # ── Painting ────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self._label_font)

        track = self._track_rect()
        split_p = self._thumb_pos(self._split)
        r = self._THUMB_RADIUS

        # ── 1. Draw tick marks ──
        self._draw_ticks(painter, track)

        # ── 2. Draw track background ──
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._COLOR_TRACK_BG)
        painter.drawRoundedRect(track, 3, 3)

        # ── 3. Draw arc region (from left to split) ──
        if split_p.x() > track.x():
            painter.setBrush(self._COLOR_ARC)
            painter.drawRoundedRect(
                QRect(track.x(), track.y(), max(split_p.x() - track.x(), 1), track.height()), 3, 3
            )

        if self._mode == "double":
            split2_p = self._thumb_pos(self._split2)
            # Draw arc2 region (between split1 and split2)
            if split2_p.x() > split_p.x() + 1:
                painter.setBrush(self._COLOR_ARC)
                painter.drawRoundedRect(
                    QRect(split_p.x(), track.y(), max(split2_p.x() - split_p.x(), 1), track.height()), 3, 3
                )

        # ── 4. Draw thumbs ──
        thumbs = [("split", self._split, split_p)]
        if self._mode == "double":
            thumbs.append(("split2", self._split2, self._thumb_pos(self._split2)))

        for role, val, tp in thumbs:
            active = role == self._drag or role == self._hover
            if active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(self._COLOR_THUMB_GLOW)
                painter.drawEllipse(tp, r + 4, r + 4)

            thumb_r = r + (1 if active else 0)
            grad = QLinearGradient(tp.x(), tp.y() - thumb_r, tp.x(), tp.y() + thumb_r)
            grad.setColorAt(0.0, self._COLOR_THUMB_HOVER if active else self._COLOR_THUMB_NORMAL)
            grad.setColorAt(1.0, QColor("#C0C0C0") if not active else QColor("#E8E8E8"))
            painter.setPen(QPen(self._COLOR_THUMB_BORDER if not active else self._COLOR_THUMB_HOVER, 1.5))
            painter.setBrush(grad)
            painter.drawEllipse(tp, thumb_r, thumb_r)

            painter.setPen(Qt.NoPen)
            painter.setBrush(self._COLOR_THUMB_BORDER if not active else QColor("#30d158"))
            painter.drawEllipse(tp, 2, 2)

    def _draw_ticks(self, painter: QPainter, track: QRect) -> None:
        span = self._max - self._min
        if span <= 0:
            return
        major_vals, minor_vals = self._compute_ticks()
        tick_base_y = track.y() - 3

        painter.setPen(QPen(self._COLOR_TICK_MINOR, 1))
        for val in minor_vals:
            frac = (val - self._min) / span
            x = track.x() + int(frac * track.width())
            painter.drawLine(x, tick_base_y - self._TICK_MINOR_HEIGHT, x, tick_base_y)

        painter.setPen(QPen(self._COLOR_TICK_MAJOR, 1.5))
        fm = painter.fontMetrics()
        for val in major_vals:
            frac = (val - self._min) / span
            x = track.x() + int(frac * track.width())
            painter.drawLine(x, tick_base_y - self._TICK_MAJOR_HEIGHT, x, tick_base_y)
            label = str(val)
            label_w = fm.horizontalAdvance(label)
            label_x = x - label_w // 2
            label_x = max(2, min(label_x, self.width() - label_w - 2))
            label_y = tick_base_y - self._TICK_MAJOR_HEIGHT - 2
            painter.setPen(self._COLOR_LABEL)
            painter.drawText(label_x, label_y, label)
            painter.setPen(QPen(self._COLOR_TICK_MAJOR, 1.5))

    # ── Internal ────────────────────────────────────────────────────

    def _move_to_pos(self, pos: QPoint) -> None:
        new_val = self._value_for_x(pos.x())

        if self._drag == "split":
            if self._mode == "double":
                new_val = min(new_val, self._split2 - 1)
            self._split = new_val
            if self._mode == "single":
                self.valueChanged.emit(self._split)
            else:
                self.valuesChanged.emit(self._split, self._split2)
        elif self._drag == "split2":
            new_val = max(new_val, self._split + 1)
            self._split2 = new_val
            self.valuesChanged.emit(self._split, self._split2)
        self.update()
