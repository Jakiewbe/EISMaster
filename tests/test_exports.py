from __future__ import annotations

import unittest
from pathlib import Path
import shutil

from eismaster.analysis.fitting import FitOutcome
from eismaster.analysis.quality import assess_spectrum_quality
from eismaster.exporters import export_batch_summary, export_fit_results, export_spectrum_bundle
from eismaster.io.chi import load_spectrum
from eismaster.models import BatchItemResult, BatchSummary


ROOT = Path(__file__).resolve().parents[1]
TXT_SAMPLE = ROOT / "Ag_EIS_OCV.txt"


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "_tmp_exports"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_txt_bundle_has_expected_headers(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        paths = export_spectrum_bundle(self.tmp_root / "bundle", spectrum)
        plot_text = paths["raw_plot"].read_text(encoding="utf-8").splitlines()[0]
        rs_rct_text = paths["rs_rct"].read_text(encoding="utf-8").splitlines()[0]
        fit_text = paths["fit_overlay"].read_text(encoding="utf-8").splitlines()[0]
        fit_second = paths["fit_overlay"].read_text(encoding="utf-8").splitlines()[1]
        self.assertEqual(plot_text, "OCV\t")
        self.assertEqual(rs_rct_text, "label\tfile\tRs\tRct\tRsei")
        self.assertEqual(fit_text, "OCV\t\t\t")
        self.assertEqual(fit_second, "z_real_exp\timag_exp_pos\tz_real_fit\timag_fit_pos")

    def test_fit_and_batch_exports(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        quality = assess_spectrum_quality(spectrum)
        fit = FitOutcome(
            model_key="demo",
            model_label="Demo Model",
            status="ok",
            message="ok",
            parameters={"Rs": 5.8, "Rct": 120.0},
            statistics={"rss": 1.2, "aic": 2.3, "bic": 3.4, "chi2_reduced": 0.12},
            predicted_real_ohm=spectrum.z_real_ohm.copy(),
            predicted_imag_ohm=spectrum.z_imag_ohm.copy(),
        )
        summary = BatchSummary(model_key="demo", items=[BatchItemResult(spectrum=spectrum, quality=quality, fit=fit)])
        fit_paths = export_fit_results(self.tmp_root / "fit", spectrum, quality, fit)
        summary_paths = export_batch_summary(self.tmp_root / "fit", summary)
        params_header = fit_paths["rs_rct"].read_text(encoding="utf-8").splitlines()[0]
        batch_header = summary_paths["rs_rct"].read_text(encoding="utf-8").splitlines()[0]
        curve_header = fit_paths["fit_overlay"].read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("label\tfile\tRs\tRct", params_header)
        self.assertIn("label\tfile\tRs\tRct", batch_header)
        self.assertEqual(curve_header, "OCV\t\t\t")

    def test_xlsx_export_creates_workbook(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        quality = assess_spectrum_quality(spectrum)
        fit = FitOutcome(
            model_key="demo",
            model_label="Demo Model",
            status="ok",
            message="ok",
            parameters={"Rs": 5.8, "Rct": 120.0},
            statistics={"rss": 1.2, "aic": 2.3, "bic": 3.4, "chi2_reduced": 0.12},
            predicted_real_ohm=spectrum.z_real_ohm.copy(),
            predicted_imag_ohm=spectrum.z_imag_ohm.copy(),
        )
        xlsx_path = self.tmp_root / "export.xlsx"
        paths = export_spectrum_bundle(xlsx_path, spectrum, fit=fit, quality=quality, fmt="xlsx")
        self.assertIn("workbook", paths)
        self.assertTrue(paths["workbook"].exists())
        self.assertEqual(paths["workbook"].suffix, ".xlsx")

    def test_xlsx_batch_export(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        quality = assess_spectrum_quality(spectrum)
        fit = FitOutcome(
            model_key="demo",
            model_label="Demo Model",
            status="ok",
            message="ok",
            parameters={"Rs": 5.8, "Rct": 120.0},
            statistics={"rss": 1.2, "aic": 2.3, "bic": 3.4, "chi2_reduced": 0.12},
            predicted_real_ohm=spectrum.z_real_ohm.copy(),
            predicted_imag_ohm=spectrum.z_imag_ohm.copy(),
        )
        summary = BatchSummary(model_key="demo", items=[BatchItemResult(spectrum=spectrum, quality=quality, fit=fit)])
        xlsx_path = self.tmp_root / "batch.xlsx"
        paths = export_batch_summary(xlsx_path, summary, fmt="xlsx")
        self.assertIn("workbook", paths)
        self.assertTrue(paths["workbook"].exists())


if __name__ == "__main__":
    unittest.main()
