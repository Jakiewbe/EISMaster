"""Lightweight dual-thumb range slider with tick marks for fitting range selection."""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt, QSize, QRect, QPoint
from PySide6.QtGui import QPainter, QPainterPath, QColor, QMouseEvent, QFont, QLinearGradient, QPen
from PySide6.QtWidgets import QWidget, QToolTip


class RangeSlider(QWidget):
    """双拇指范围选择滑块（带刻度定位点）。

    Signals:
        rangeChanged: Emitted with (start, end) when the range changes.
    """

    rangeChanged = Signal(int, int)

    # ── Visual constants ────────────────────────────────────────────
    _TRACK_HEIGHT = 4
    _THUMB_RADIUS = 8
    _TICK_MAJOR_HEIGHT = 8
    _TICK_MINOR_HEIGHT = 4
    _LABEL_FONT_SIZE = 8
    _MARGIN_H = 10          # horizontal margin for track
    _MARGIN_TOP = 18        # space above track for labels + ticks
    _MARGIN_BOTTOM = 4

    # Colors
    _COLOR_TRACK_BG = QColor("#2A2A30")
    _COLOR_TRACK_ACTIVE = QColor("#30d158")
    _COLOR_THUMB_NORMAL = QColor("#E0E0E0")
    _COLOR_THUMB_HOVER = QColor("#FFFFFF")
    _COLOR_THUMB_BORDER = QColor("#555566")
    _COLOR_THUMB_GLOW = QColor(48, 209, 88, 80)   # green glow
    _COLOR_TICK_MAJOR = QColor("#666680")
    _COLOR_TICK_MINOR = QColor("#44445A")
    _COLOR_LABEL = QColor("#8888A0")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._start = 0
        self._end = 100
        self._drag = None  # None | "start" | "end"
        self._hover = None  # None | "start" | "end"
        self._total_height = self._MARGIN_TOP + self._TRACK_HEIGHT + self._MARGIN_BOTTOM + 2 * self._THUMB_RADIUS
        self.setFixedHeight(self._total_height)
        self.setMouseTracking(True)
        self._label_font = QFont("Segoe UI, Microsoft YaHei", self._LABEL_FONT_SIZE)
        self._label_font.setPixelSize(self._LABEL_FONT_SIZE + 2)

    # ── Public API ──────────────────────────────────────────────────

    def setRange(self, minimum: int, maximum: int) -> None:
        self._min = minimum
        self._max = maximum
        if self._start < minimum:
            self._start = minimum
        if self._end > maximum:
            self._end = maximum
        if self._start > self._end:
            self._start, self._end = self._end, self._start
        self.update()

    def setValue(self, start: int, end: int) -> None:
        start = max(self._min, min(start, self._max))
        end = max(self._min, min(end, self._max))
        if start > end:
            start, end = end, start
        if start == self._start and end == self._end:
            return
        self._start = start
        self._end = end
        self.rangeChanged.emit(self._start, self._end)
        self.update()

    def value(self) -> tuple[int, int]:
        return self._start, self._end

    # ── Geometry helpers ────────────────────────────────────────────

    def _track_rect(self) -> QRect:
        w, h = self.width(), self.height()
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
        r = self._THUMB_RADIUS + 4  # hit margin
        for role in ("start", "end"):
            tp = self._thumb_pos(getattr(self, f"_{role}"))
            if (pos.x() - tp.x()) ** 2 + (pos.y() - tp.y()) ** 2 <= r ** 2:
                return role
        return None

    def _value_for_x(self, x: int) -> int:
        """Convert pixel x to slider value."""
        track = self._track_rect()
        span = self._max - self._min
        if span <= 0:
            return self._min
        frac = (x - track.x()) / max(track.width(), 1)
        frac = max(0.0, min(1.0, frac))
        return self._min + int(round(frac * span))

    # ── Tick calculation ────────────────────────────────────────────

    def _compute_ticks(self) -> tuple[list[int], list[int]]:
        """Return (major_values, minor_values) for tick marks."""
        span = self._max - self._min
        if span <= 0:
            return [], []

        # Major ticks: 0%, 25%, 50%, 75%, 100%
        major_values = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            major_values.append(self._min + int(round(frac * span)))
        # Deduplicate
        major_values = sorted(set(major_values))

        # Minor ticks: every 10% (excluding majors)
        minor_values = []
        if span >= 10:
            for frac_pct in range(10, 100, 10):
                frac = frac_pct / 100.0
                val = self._min + int(round(frac * span))
                if val not in major_values:
                    minor_values.append(val)

        return major_values, minor_values

    # ── Events ──────────────────────────────────────────────────────

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
        start_pos = self._thumb_pos(self._start)
        end_pos = self._thumb_pos(self._end)
        d_start = (pos.x() - start_pos.x()) ** 2 + (pos.y() - start_pos.y()) ** 2
        d_end = (pos.x() - end_pos.x()) ** 2 + (pos.y() - end_pos.y()) ** 2
        if d_start <= d_end:
            self._drag = "start"
            self._move_to_pos(pos)
        else:
            self._drag = "end"
            self._move_to_pos(pos)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        if self._drag:
            self._move_to_pos(pos)
            # Show tooltip with current value
            val = self._value_for_x(pos.x())
            QToolTip.showText(event.globalPosition().toPoint(), f"idx: {val}", self)
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
        start_p = self._thumb_pos(self._start)
        end_p = self._thumb_pos(self._end)
        r = self._THUMB_RADIUS

        # ── 1. Draw tick marks and labels above the track ──
        self._draw_ticks(painter, track)

        # ── 2. Draw track background ──
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._COLOR_TRACK_BG)
        painter.drawRoundedRect(track, 3, 3)

        # ── 3. Draw active (selected) region ──
        active_left = min(start_p.x(), end_p.x())
        active_right = max(start_p.x(), end_p.x())
        if active_right > active_left:
            grad = QLinearGradient(active_left, 0, active_right, 0)
            grad.setColorAt(0.0, QColor("#26A65B"))
            grad.setColorAt(1.0, QColor("#30d158"))
            painter.setBrush(grad)
            painter.drawRoundedRect(
                QRect(active_left, track.y(), max(active_right - active_left, 1), track.height()), 3, 3
            )

        # ── 4. Draw thumbs ──
        for role in ("start", "end"):
            tp = self._thumb_pos(getattr(self, f"_{role}"))
            active = role == self._drag or role == self._hover

            # Glow effect on hover/drag
            if active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(self._COLOR_THUMB_GLOW)
                painter.drawEllipse(tp, r + 4, r + 4)

            # Thumb body with gradient
            thumb_r = r + (1 if active else 0)
            grad = QLinearGradient(tp.x(), tp.y() - thumb_r, tp.x(), tp.y() + thumb_r)
            grad.setColorAt(0.0, self._COLOR_THUMB_HOVER if active else self._COLOR_THUMB_NORMAL)
            grad.setColorAt(1.0, QColor("#C0C0C0") if not active else QColor("#E8E8E8"))

            painter.setPen(QPen(self._COLOR_THUMB_BORDER if not active else self._COLOR_THUMB_HOVER, 1.5))
            painter.setBrush(grad)
            painter.drawEllipse(tp, thumb_r, thumb_r)

            # Inner dot for better grab visual
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._COLOR_THUMB_BORDER if not active else QColor("#30d158"))
            painter.drawEllipse(tp, 2, 2)

    def _draw_ticks(self, painter: QPainter, track: QRect) -> None:
        """Draw tick marks and numeric labels above the track."""
        span = self._max - self._min
        if span <= 0:
            return

        major_vals, minor_vals = self._compute_ticks()

        tick_base_y = track.y() - 3  # just above the track

        # Minor ticks
        painter.setPen(QPen(self._COLOR_TICK_MINOR, 1))
        for val in minor_vals:
            frac = (val - self._min) / span
            x = track.x() + int(frac * track.width())
            painter.drawLine(x, tick_base_y - self._TICK_MINOR_HEIGHT, x, tick_base_y)

        # Major ticks + labels
        painter.setPen(QPen(self._COLOR_TICK_MAJOR, 1.5))
        fm = painter.fontMetrics()
        for val in major_vals:
            frac = (val - self._min) / span
            x = track.x() + int(frac * track.width())
            painter.drawLine(x, tick_base_y - self._TICK_MAJOR_HEIGHT, x, tick_base_y)

            # Draw label
            label = str(val)
            label_w = fm.horizontalAdvance(label)
            label_x = x - label_w // 2
            # Clamp to widget bounds
            label_x = max(2, min(label_x, self.width() - label_w - 2))
            label_y = tick_base_y - self._TICK_MAJOR_HEIGHT - 2
            painter.setPen(self._COLOR_LABEL)
            painter.drawText(label_x, label_y, label)
            painter.setPen(QPen(self._COLOR_TICK_MAJOR, 1.5))

    # ── Internal ────────────────────────────────────────────────────

    def _move_to_pos(self, pos: QPoint) -> None:
        new_val = self._value_for_x(pos.x())

        if self._drag == "start":
            new_start = min(new_val, self._end)
            self._start = new_start
            self.rangeChanged.emit(self._start, self._end)
        else:
            new_end = max(new_val, self._start)
            self._end = new_end
            self.rangeChanged.emit(self._start, self._end)
        self.update()
