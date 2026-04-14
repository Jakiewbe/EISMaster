"""Lightweight dual-thumb range slider for fitting range selection."""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt, QSize, QRect, QPoint
from PySide6.QtGui import QPainter, QPainterPath, QColor, QMouseEvent
from PySide6.QtWidgets import QWidget


class RangeSlider(QWidget):
    """双拇指范围选择滑块。

    Signals:
        rangeChanged: Emitted with (start, end) when the range changes.
    """

    rangeChanged = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._start = 0
        self._end = 100
        self._drag = None  # None | "start" | "end"
        self._hover = None  # None | "start" | "end"
        self.setFixedHeight(20)
        self.setMouseTracking(True)

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
        m = 4  # margin
        w, h = self.width(), self.height()
        return QRect(m, h // 2 - 2, max(w - 2 * m, 1), 4)

    def _thumb_radius(self) -> int:
        return 7

    def _thumb_pos(self, value: int) -> QPoint:
        track = self._track_rect()
        span = self._max - self._min
        if span <= 0:
            return QPoint(track.x(), track.y() + track.height() // 2)
        frac = (value - self._min) / span
        x = track.x() + int(frac * track.width())
        return QPoint(x, track.y() + track.height() // 2)

    def _hit_thumb(self, pos: QPoint) -> str | None:
        r = self._thumb_radius() + 4  # hit margin
        for role in ("start", "end"):
            tp = self._thumb_pos(getattr(self, f"_{role}"))
            if (pos.x() - tp.x()) ** 2 + (pos.y() - tp.y()) ** 2 <= r ** 2:
                return role
        return None

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
            event.accept()
            return
        hover = self._hit_thumb(pos)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        super().leaveEvent(event)

    # ── Painting ────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        track = self._track_rect()
        start_p = self._thumb_pos(self._start)
        end_p = self._thumb_pos(self._end)
        r = self._thumb_radius()

        # background (inactive region)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#3a3a3c"))
        painter.drawRoundedRect(track, 3, 3)

        # active (selected) region — green bar
        active_left = min(start_p.x(), end_p.x()) - r // 2
        active_right = max(start_p.x(), end_p.x()) + r // 2
        painter.setBrush(QColor("#30d158"))
        painter.drawRoundedRect(
            QRect(active_left, track.y(), max(active_right - active_left, 1), track.height()), 3, 3
        )

        # thumbs
        for role in ("start", "end"):
            tp = self._thumb_pos(getattr(self, f"_{role}"))
            active = role == self._drag or role == self._hover
            thumb_r = r + (2 if active else 0)
            painter.setPen(QColor("#ffffff") if active else QColor("#555555"))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(tp, thumb_r, thumb_r)

    # ── Internal ────────────────────────────────────────────────────

    def _move_to_pos(self, pos: QPoint) -> None:
        track = self._track_rect()
        span = self._max - self._min
        if span <= 0:
            return
        frac = (pos.x() - track.x()) / max(track.width(), 1)
        frac = max(0.0, min(1.0, frac))
        new_val = self._min + int(round(frac * span))

        if self._drag == "start":
            new_start = min(new_val, self._end)
            self._start = new_start
            self.rangeChanged.emit(self._start, self._end)
        else:
            new_end = max(new_val, self._start)
            self._end = new_end
            self.rangeChanged.emit(self._start, self._end)
        self.update()
