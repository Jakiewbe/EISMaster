from __future__ import annotations

import unittest
from pathlib import Path
import shutil

import numpy as np

from eismaster.analysis.fitting import fit_spectrum
from eismaster.analysis.quality import assess_spectrum_quality
from eismaster.analysis.segmentation import detect_segments
from eismaster.io.chi import load_spectrum, load_spectra_from_folder
from eismaster.models import SpectrumData, SpectrumMetadata


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

    def test_single_segmentation_prefers_local_arc_peak_over_terminal_maximum(self) -> None:
        z_real = np.array([
            3.1370325088500977, 3.2019760608673096, 3.294358491897583, 3.3983049392700195,
            3.5116286277770996, 3.6723814010620117, 3.832101345062256, 4.0781755447387695,
            4.306181907653809, 4.6339192390441895, 5.016920566558838, 5.525825023651123,
            5.7781853675842285, 6.295176982879639, 6.982400417327881, 7.756028175354004,
            8.5746488571167, 9.696084976196289, 10.711566925048828, 12.151483535766602,
            13.457901954650879, 15.180721282958984, 16.912742614746094, 19.167888641357422,
            21.18239402770996, 23.007064819335938, 25.105937957763672, 27.067062377929688,
            28.77385139465332, 30.79508399963379, 32.360267639160156, 34.263851165771484,
            35.80744934082031, 37.72715377807617, 39.44247817993164, 41.6188850402832,
            43.52027893066406, 45.17778396606445, 47.10660171508789, 48.8984375,
            50.31928634643555, 52.078575134277344, 53.31765365600586, 54.66347885131836,
            55.58536911010742, 56.807029724121094, 57.501121520996094, 58.45831298828125,
            59.83451843261719, 60.372886657714844, 61.02216339111328, 61.62940979003906,
            61.88295364379883, 62.55939865112305, 62.986942291259766, 63.44816207885742,
            63.63958740234375, 64.47903442382812, 64.53456115722656, 66.55062866210938,
            69.5946273803711, 72.46876525878906, 74.60115814208984, 76.15950775146484,
            77.40392303466797, 78.64726257324219, 79.95594787597656, 81.32293701171875,
            82.96575164794922, 85.24642944335938, 88.54199981689453, 93.36370849609375,
            101.17423248291016, 112.24366760253906, 126.74373626708984, 146.2497100830078,
            172.62820434570312, 209.0454864501953, 257.45794677734375, 319.511962890625,
            398.80987548828125, 496.889892578125, 614.0899658203125, 750.5719604492188,
            906.01904296875,
        ])
        minus_z_imag = np.array([
            0.18646788597106934, 0.48757070302963257, 0.8274839520454407, 1.1661933660507202,
            1.4834896326065063, 1.8960415124893188, 2.2580952644348145, 2.7365012168884277,
            3.144876480102539, 3.6845641136169434, 4.217316627502441, 4.932314872741699,
            5.212133407592773, 5.827437400817871, 6.597487926483154, 7.372109889984131,
            8.124319076538086, 9.05472469329834, 9.852801322937012, 10.767127990722656,
            11.505534172058105, 12.374295234680176, 12.975096702575684, 13.620817184448242,
            13.879697799682617, 14.126628875732422, 14.246830940246582, 14.267672538757324,
            14.2318696975708, 14.218637466430664, 14.18437385559082, 14.16822624206543,
            14.160116195678711, 14.121678352355957, 14.093001365661621, 13.976531028747559,
            13.609014511108398, 13.505398750305176, 13.068739891052246, 12.632678985595703,
            12.155242919921875, 11.578140258789062, 11.133113861083984, 10.559755325317383,
            10.051218032836914, 9.426355361938477, 9.14096736907959, 8.710587501525879,
            6.740202903747559, 7.163536071777344, 7.236416816711426, 7.653050422668457,
            8.120189666748047, 8.981240272521973, 9.912817001342773, 11.278128623962402,
            12.491313934326172, 14.03664493560791, 16.3352108001709, 19.377609252929688,
            23.29266929626465, 28.00611686706543, 33.663387298583984, 40.42706298828125,
            48.50446319580078, 58.0033073425293, 69.14427947998047, 82.22943115234375,
            97.49341583251953, 115.40673065185547, 136.2225341796875, 160.459228515625,
            191.00949096679688, 226.89418029785156, 266.76470947265625, 312.3882141113281,
            364.2469787597656, 423.3614501953125, 488.6627502441406, 558.0294799804688,
            628.3460693359375, 696.71044921875, 760.9888916015625, 815.5950317382812,
            858.78662109375,
        ])
        freq = np.geomspace(96679.7, 0.01, z_real.size)
        z_imag = -minus_z_imag
        spectrum = SpectrumData(
            metadata=SpectrumMetadata(file_path=ROOT / "synthetic_cfx_like.txt"),
            freq_hz=freq,
            z_real_ohm=z_real,
            z_imag_ohm=z_imag,
            z_mod_ohm=np.sqrt(z_real**2 + z_imag**2),
            phase_deg=np.degrees(np.arctan2(z_imag, z_real)),
        )

        detection = detect_segments(spectrum, mode="auto")

        self.assertEqual(detection.resolved_mode, "single")
        self.assertLess(detection.peak_indices[0], 40)
        self.assertLess(detection.split_indices[0], 60)
        self.assertGreater(detection.split_indices[0], detection.peak_indices[0])

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
