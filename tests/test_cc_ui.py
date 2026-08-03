from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from eismaster.ui.cc_page import CCCurvePage
from eismaster.ui.main_window import MainWindow
from tests.cc_fixtures import write_cc_bin


APP = QApplication.instance() or QApplication([])


def test_main_window_registers_cc_curve_page() -> None:
    window = MainWindow()
    try:
        assert window.cc_widget.objectName() == "ccInterface"
        assert window.cc_widget.windowTitle() == "CC曲线"
    finally:
        window.close()


def test_cc_page_loads_folder_and_shows_curve(tmp_path: Path) -> None:
    write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 4.0], current_raw=-18128)

    page = CCCurvePage()
    page.load_folder(tmp_path)

    assert len(page.curves) == 1
    assert page.status_label.text().startswith("已加载")
