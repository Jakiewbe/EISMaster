from typing import List
import typing






"""
NanoBanana Modern UI Theme System
A sleek, high-fidelity dark mode with neon accents and glassmorphism hints.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

NANOBANANA_ACCENT = "#2AE88A"  # Neon Green/Teal
NANOBANANA_ACCENT_HOVER = "#5CFFA8"
NANOBANANA_BG_DARK = "#121214"
NANOBANANA_BG_PANEL = "#1E1E22"
NANOBANANA_BG_INPUT = "#29292F"
NANOBANANA_TEXT = "#E2E2E5"
NANOBANANA_TEXT_MUTED = "#8B8B9B"
NANOBANANA_BORDER = "#33333C"


def apply_nanobanana_theme(app: QApplication) -> None:
    """Applies the deep immersive NanoBanana visual styles to the entire application."""

    # 1. Update general Application Palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(NANOBANANA_BG_DARK))
    palette.setColor(QPalette.WindowText, QColor(NANOBANANA_TEXT))
    palette.setColor(QPalette.Base, QColor(NANOBANANA_BG_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(NANOBANANA_BG_INPUT))
    palette.setColor(QPalette.ToolTipBase, QColor(NANOBANANA_BG_DARK))
    palette.setColor(QPalette.ToolTipText, QColor(NANOBANANA_TEXT))
    palette.setColor(QPalette.Text, QColor(NANOBANANA_TEXT))
    palette.setColor(QPalette.Button, QColor(NANOBANANA_BG_PANEL))
    palette.setColor(QPalette.ButtonText, QColor(NANOBANANA_TEXT))
    palette.setColor(QPalette.BrightText, Qt.red if hasattr(Qt, 'red') else QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(NANOBANANA_ACCENT))
    palette.setColor(QPalette.Highlight, QColor(NANOBANANA_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor(NANOBANANA_BG_DARK))
    app.setPalette(palette)
    
    # 2. Modern Font (with Chinese fallback)
    font = QFont("Segoe UI Variable Display, Segoe UI, Roboto, Helvetica, Microsoft YaHei, SimHei", 10)
    app.setFont(font)

    # 3. Conservative global stylesheet.
    # Avoid styling every widget too aggressively; just force the common
    # containers and inputs into a stable dark theme.
    qss = f"""
    QMainWindow, QDialog, MSFluentWindow {{
        background-color: {NANOBANANA_BG_DARK};
        color: {NANOBANANA_TEXT};
    }}
    QWidget {{
        color: {NANOBANANA_TEXT};
    }}
    QScrollArea, QAbstractScrollArea {{
        background-color: transparent;
        border: none;
    }}
    SimpleCardWidget, HeaderCardWidget, QFrame {{
        background-color: {NANOBANANA_BG_PANEL};
        border: 1px solid {NANOBANANA_BORDER};
        border-radius: 8px;
    }}
    QLabel, BodyLabel, CaptionLabel, StrongBodyLabel, TitleLabel {{
        color: {NANOBANANA_TEXT};
    }}
    QTextEdit, QPlainTextEdit, QListWidget, QTableWidget {{
        background-color: {NANOBANANA_BG_PANEL};
        color: {NANOBANANA_TEXT};
        border: 1px solid {NANOBANANA_BORDER};
        border-radius: 6px;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {NANOBANANA_BG_INPUT};
        color: {NANOBANANA_TEXT};
        border: 1px solid {NANOBANANA_BORDER};
        border-radius: 6px;
        padding: 5px 8px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {NANOBANANA_ACCENT};
    }}
    QPushButton, ToolButton, PushButton, PrimaryPushButton, TransparentPushButton {{
        background-color: {NANOBANANA_BG_INPUT};
        border: 1px solid {NANOBANANA_BORDER};
        border-radius: 6px;
        padding: 6px 16px;
        color: {NANOBANANA_TEXT};
    }}
    QPushButton:hover, ToolButton:hover, PushButton:hover, PrimaryPushButton:hover, TransparentPushButton:hover {{
        background-color: {NANOBANANA_BORDER};
    }}
    QPushButton:pressed, ToolButton:pressed, PushButton:pressed, PrimaryPushButton:pressed, TransparentPushButton:pressed {{
        background-color: {NANOBANANA_ACCENT};
        color: {NANOBANANA_BG_DARK};
    }}
    PrimaryPushButton {{
        background-color: {NANOBANANA_ACCENT};
        color: {NANOBANANA_BG_DARK};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView, QListView, QMenu {{
        background-color: {NANOBANANA_BG_INPUT};
        color: {NANOBANANA_TEXT};
        border: 1px solid {NANOBANANA_BORDER};
    }}
    QHeaderView::section {{
        background-color: {NANOBANANA_BG_INPUT};
        color: {NANOBANANA_TEXT};
        padding: 5px;
        border-right: 1px solid {NANOBANANA_BORDER};
        border-bottom: 1px solid {NANOBANANA_BORDER};
    }}
    QProgressBar {{
        background-color: {NANOBANANA_BG_INPUT};
        border: 1px solid {NANOBANANA_BORDER};
        border-radius: 4px;
        text-align: center;
        color: {NANOBANANA_TEXT};
    }}
    QProgressBar::chunk {{
        background-color: {NANOBANANA_ACCENT};
        border-radius: 3px;
    }}
    QSplitter::handle {{
        background-color: {NANOBANANA_BG_DARK};
    }}
    QWidget#scrollAreaWidgetContents,
    QWidget#inspectInterface,
    QWidget#fitInterface,
    QWidget#batchInterface {{
        background: transparent;
    }}
    """
    app.setStyleSheet(qss)
