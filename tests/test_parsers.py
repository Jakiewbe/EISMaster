from __future__ import annotations

import unittest
from pathlib import Path
import shutil

import numpy as np

from eismaster.analysis.fitting import fit_spectrum
from eismaster.analysis.quality import assess_spectrum_quality
from eismaster.analysis.segmentation import detect_segments
from eismaster.io.chi import load_spectrum, load_spectra_from_folder


ROOT = Path(__file__).resolve().parents[1]
TXT_SAMPLE = ROOT / "Ag_EIS_OCV.txt"
BIN_SAMPLE = ROOT / "Ag_EIS_OCV.bin"
DOUBLE_SAMPLE = next(ROOT.glob("*放电完后.txt"))


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "_tmp_parsers"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_txt_sample_parses(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        self.assertEqual(spectrum.metadata.instrument_model, "CHI660F")
        self.assertEqual(spectrum.n_points, 85)
        self.assertAlmostEqual(float(spectrum.freq_hz[0]), 9.668e4, places=0)
        self.assertAlmostEqual(float(spectrum.z_real_ohm[0]), 5.740, places=3)
        self.assertAlmostEqual(float(spectrum.z_imag_ohm[0]), -4.091e-1, places=4)

    def test_bin_sample_matches_txt_within_binary_precision(self) -> None:
        txt_spectrum = load_spectrum(TXT_SAMPLE)
        bin_spectrum = load_spectrum(BIN_SAMPLE)
        self.assertEqual(bin_spectrum.n_points, txt_spectrum.n_points)
        np.testing.assert_allclose(bin_spectrum.freq_hz, txt_spectrum.freq_hz, rtol=1e-4, atol=5.0)
        np.testing.assert_allclose(bin_spectrum.z_real_ohm, txt_spectrum.z_real_ohm, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(bin_spectrum.z_imag_ohm, txt_spectrum.z_imag_ohm, rtol=1e-3, atol=0.5)

    def test_fitting_smoke_for_single_semicircle_model(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        fit = fit_spectrum(spectrum, "zview_segmented_rq_rwo")
        self.assertIn(fit.status, {"ok", "warn"})
        self.assertGreater(fit.parameters["Rs"], 0.0)
        self.assertGreater(fit.parameters["Rct"], 0.0)
        self.assertTrue(np.isfinite(fit.statistics["rss"]))

    def test_fitting_smoke_for_double_semicircle_model(self) -> None:
        spectrum = load_spectrum(DOUBLE_SAMPLE)
        fit = fit_spectrum(spectrum, "zview_double_rq_qrwo")
        self.assertIn(fit.status, {"ok", "warn"})
        self.assertGreater(fit.parameters["Rs"], 0.0)
        self.assertGreater(fit.parameters["Rsei"], 0.0)
        self.assertGreater(fit.parameters["Rct"], 0.0)
        self.assertIn("Rsei_stderr_pct", fit.statistics)
        self.assertIn("Rct_stderr_pct", fit.statistics)
        self.assertIn("Rsei_global_stderr_pct", fit.statistics)
        self.assertIn("Rct_global_stderr_pct", fit.statistics)

    def test_double_segmentation_accepts_manual_peaks_and_splits(self) -> None:
        spectrum = load_spectrum(DOUBLE_SAMPLE)
        detection = detect_segments(
            spectrum,
            mode="double",
            manual_peak1=10,
            manual_split1=20,
            manual_peak2=30,
            manual_split2=45,
        )
        self.assertEqual(detection.resolved_mode, "double")
        self.assertEqual(detection.peak_indices, (10, 30))
        self.assertEqual(detection.split_indices, (20, 45))
        self.assertLess(detection.peak_indices[0], detection.split_indices[0])
        self.assertLess(detection.split_indices[0], detection.peak_indices[1])
        self.assertLess(detection.peak_indices[1], detection.split_indices[1])

    def test_plain_numeric_txt_with_tabs_parses(self) -> None:
        sample = self.tmp_root / "plain_numeric.txt"
        sample.write_text(
            "1000\t5.1\t-0.2\n100\t5.4\t-0.5\n10\t6.2\t-1.1\n",
            encoding="utf-8",
        )
        spectrum = load_spectrum(sample)
        self.assertEqual(spectrum.n_points, 3)
        self.assertAlmostEqual(float(spectrum.freq_hz[0]), 1000.0)
        self.assertAlmostEqual(float(spectrum.z_real_ohm[1]), 5.4)
        self.assertAlmostEqual(float(spectrum.z_imag_ohm[2]), -1.1)

    def test_folder_import_skips_bad_supported_file_when_others_are_valid(self) -> None:
        good = self.tmp_root / "good.txt"
        bad = self.tmp_root / "bad.txt"
        good.write_text(
            "1000,5.1,-0.2\n100,5.4,-0.5\n10,6.2,-1.1\n",
            encoding="utf-8",
        )
        bad.write_text("this is not an eis file\njust notes\n", encoding="utf-8")
        spectra = load_spectra_from_folder(self.tmp_root)
        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].metadata.file_path.name, "good.txt")

    def test_quality_report_summary_lines_are_human_readable(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        quality = assess_spectrum_quality(spectrum, run_kk=False)
        self.assertEqual(spectrum.acquired_label, "未知")
        self.assertEqual(quality.kk_message, "KK/Z-HIT 未执行。")
        self.assertIn("状态: pass", quality.summary_lines())
        self.assertIn("未发现明显质量问题。", quality.summary_lines())


if __name__ == "__main__":
    unittest.main()
