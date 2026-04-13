from __future__ import annotations
from typing import Optional
import typing









from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eismaster.ui.theme import NANOBANANA_ACCENT, NANOBANANA_BG_DARK, NANOBANANA_BG_PANEL, NANOBANANA_BORDER


class Socket(QGraphicsItem):
    Type = QGraphicsItem.UserType + 1

    def __init__(self, parent_node: "ComponentNode", is_input: bool):
        super().__init__(parent_node)
        self.parent_node = parent_node
        self.is_input = is_input
        self.radius = 6.0
        self.setAcceptHoverEvents(True)
        self.edges: list["Wire"] = []
        self.setZValue(1)

    def boundingRect(self) -> QRectF:
        return QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setBrush(QBrush(QColor(NANOBANANA_ACCENT)))
        painter.setPen(QPen(QColor(NANOBANANA_BG_DARK), 2))
        painter.drawEllipse(self.boundingRect())

    def connect_edge(self, edge: "Wire") -> None:
        if edge not in self.edges:
            self.edges.append(edge)

    def remove_edge(self, edge: "Wire") -> None:
        if edge in self.edges:
            self.edges.remove(edge)


class ComponentNode(QGraphicsObject):
    Type = QGraphicsItem.UserType + 2
    deleted = Signal(object)

    def __init__(self, c_type: str, label: str):
        super().__init__()
        self.c_type = c_type
        self.label = label
        self.width = 100
        self.height = 60
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        self.in_socket = Socket(self, True)
        self.in_socket.setPos(0, self.height / 2)
        self.out_socket = Socket(self, False)
        self.out_socket.setPos(self.width, self.height / 2)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        path = QPainterPath()
        path.addRoundedRect(self.boundingRect(), 8, 8)
        painter.setPen(QPen(QColor(NANOBANANA_ACCENT if self.isSelected() else NANOBANANA_BORDER), 2))
        painter.setBrush(QBrush(QColor(NANOBANANA_BG_PANEL)))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(NANOBANANA_ACCENT), 2))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(self.boundingRect(), Qt.AlignCenter, f"{self.c_type}\n{self.label}")

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.in_socket.edges:
                edge.update_positions()
            for edge in self.out_socket.edges:
                edge.update_positions()
        return super().itemChange(change, value)


class Wire(QGraphicsPathItem):
    Type = QGraphicsItem.UserType + 3

    def __init__(self, source: Socket, dest: Socket):
        super().__init__()
        self.source = source
        self.dest = dest
        self.source.connect_edge(self)
        self.dest.connect_edge(self)
        self.setZValue(-1)
        self.update_positions()

    def update_positions(self) -> None:
        source_pos = self.source.scenePos()
        dest_pos = self.dest.scenePos()
        path = QPainterPath(source_pos)
        dx = abs(dest_pos.x() - source_pos.x()) * 0.5
        ctrl1 = QPointF(source_pos.x() + max(dx, 30), source_pos.y())
        ctrl2 = QPointF(dest_pos.x() - max(dx, 30), dest_pos.y())
        path.cubicTo(ctrl1, ctrl2, dest_pos)
        self.setPath(path)
        self.setPen(QPen(QColor(NANOBANANA_BORDER), 3))

    def remove(self) -> None:
        self.source.remove_edge(self)
        self.dest.remove_edge(self)
        if self.scene():
            self.scene().removeItem(self)


class CircuitScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor(NANOBANANA_BG_DARK)))
        self.active_wire: Optional[QGraphicsPathItem] = None
        self.wire_start_socket: Optional[Socket] = None

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if isinstance(item, Socket):
            self.wire_start_socket = item
            self.active_wire = QGraphicsPathItem()
            self.active_wire.setPen(QPen(QColor(NANOBANANA_ACCENT), 2, Qt.DashLine))
            self.active_wire.setZValue(-1)
            self.addItem(self.active_wire)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.active_wire and self.wire_start_socket:
            start_pos = self.wire_start_socket.scenePos()
            end_pos = event.scenePos()
            path = QPainterPath(start_pos)
            dx = abs(end_pos.x() - start_pos.x()) * 0.5
            ctrl1 = QPointF(start_pos.x() + dx, start_pos.y())
            ctrl2 = QPointF(end_pos.x() - dx, end_pos.y())
            path.cubicTo(ctrl1, ctrl2, end_pos)
            self.active_wire.setPath(path)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.active_wire:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, Socket) and item != self.wire_start_socket and self.wire_start_socket is not None:
                if self.wire_start_socket.is_input != item.is_input:
                    source = self.wire_start_socket if not self.wire_start_socket.is_input else item
                    dest = item if not item.is_input else self.wire_start_socket
                    self.addItem(Wire(source, dest))
            self.removeItem(self.active_wire)
            self.active_wire = None
            self.wire_start_socket = None
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.selectedItems():
                if isinstance(item, ComponentNode):
                    if item.c_type in ("INPUT", "OUTPUT"):
                        continue
                    for edge in list(item.in_socket.edges) + list(item.out_socket.edges):
                        edge.remove()
                    self.removeItem(item)
                elif isinstance(item, Wire):
                    item.remove()
        super().keyPressEvent(event)


class CircuitBuilderWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EISMaster Circuit Builder")
        self.resize(1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        palette_widget = QWidget()
        palette_widget.setMaximumWidth(200)
        palette_layout = QVBoxLayout(palette_widget)
        palette_layout.setContentsMargins(15, 15, 15, 15)
        palette_layout.setSpacing(10)

        title = QLabel("Components")
        title.setObjectName("heroSubtitle")
        palette_layout.addWidget(title)

        components = [
            ("Resistor", "R"),
            ("Capacitor", "C"),
            ("Const. Phase (CPE)", "Q"),
            ("Warburg Open", "W"),
            ("Inductor", "L"),
        ]

        self.counters = {k: 0 for _, k in components}
        for name, key in components:
            btn = QPushButton(name)
            btn.setObjectName("secondaryButton")
            btn.clicked.connect(lambda _, k=key: self.add_node(k))
            palette_layout.addWidget(btn)

        palette_layout.addStretch(1)

        compile_btn = QPushButton("Compile Circuit")
        compile_btn.setObjectName("primaryButton")
        compile_btn.clicked.connect(self._compile_circuit)
        palette_layout.addWidget(compile_btn)
        layout.addWidget(palette_widget)

        self.scene = CircuitScene(self)
        self.scene.setSceneRect(0, 0, 2000, 2000)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        layout.addWidget(self.view, 1)

        self.input_node = ComponentNode("INPUT", "Term 1")
        self.input_node.in_socket.hide()
        self.input_node.setPos(100, 300)

        self.output_node = ComponentNode("OUTPUT", "Term 2")
        self.output_node.out_socket.hide()
        self.output_node.setPos(900, 300)

        self.scene.addItem(self.input_node)
        self.scene.addItem(self.output_node)

    def add_node(self, comp_key: str) -> None:
        self.counters[comp_key] += 1
        node = ComponentNode(comp_key, f"{comp_key}{self.counters[comp_key]}")
        visible_center = self.view.mapToScene(self.view.viewport().rect().center())
        node.setPos(visible_center.x() - 50, visible_center.y() - 30)
        self.scene.addItem(node)

    def _compile_circuit(self) -> None:
        from .logic import build_cdc_from_scene

        cdc = build_cdc_from_scene(self.scene)
        QMessageBox.information(self, "Circuit Compiled", f"Generated CDC String:\n{cdc}")
