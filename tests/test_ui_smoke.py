from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from eismaster.io.chi import load_spectrum
from eismaster.models import FitOutcome
from eismaster.ui.main_window import MainWindow


ROOT = Path(__file__).resolve().parents[1]
TXT_SAMPLE = ROOT / "Ag_EIS_OCV.txt"
APP = QApplication.instance() or QApplication([])


class UiSmokeTests(unittest.TestCase):
    def test_main_window_can_load_sample_spectrum(self) -> None:
        window = MainWindow()
        try:
            spectrum = load_spectrum(TXT_SAMPLE)
            window._merge_state([spectrum])
            self.assertEqual(len(window.state.spectra), 1)
            self.assertEqual(window.state.current_index, 0)
            self.assertEqual(window._current_spectrum(silent=True).display_name, TXT_SAMPLE.name)
            window.navigationInterface.setCurrentItem(window.fit_widget.objectName())
            window._refresh_current_tab()
            self.assertTrue(window.fit_text.toPlainText())
        finally:
            window.close()

    def test_fit_view_renders_plot_items_after_fit(self) -> None:
        window = MainWindow()
        try:
            spectrum = load_spectrum(TXT_SAMPLE)
            window._merge_state([spectrum])
            fit = FitOutcome(
                model_key="zview_segmented_rq_rwo",
                model_label="Render Demo",
                status="ok",
                message="fit rendered",
                parameters={"Rs": 1.0, "Rct": 2.0},
                statistics={},
                predicted_real_ohm=spectrum.z_real_ohm.copy(),
                predicted_imag_ohm=spectrum.z_imag_ohm.copy(),
            )
            window.state.fits[(spectrum.display_name, fit.model_key)] = fit
            window.navigationInterface.setCurrentItem(window.fit_widget.objectName())
            window._refresh_fit_view(spectrum)

            item_names = [item.name() for item in window.fit_nyquist_plot.listDataItems()]
            self.assertGreaterEqual(len(item_names), 2)
            self.assertTrue(any(name == "拟合曲线" for name in item_names if name))
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
