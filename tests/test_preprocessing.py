from __future__ import annotations

import unittest

import numpy as np

from eismaster.analysis.preprocessing import preprocess_for_fitting
from eismaster.models import SpectrumData


def _make_spectrum(freq: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray) -> SpectrumData:
    z_mod = np.sqrt(z_real**2 + z_imag**2)
    phase = np.arctan2(z_imag, z_real) * 180.0 / np.pi
    return SpectrumData(
        freq_hz=freq,
        z_real_ohm=z_real,
        z_imag_ohm=z_imag,
        z_mod_ohm=z_mod,
        phase_deg=phase,
        metadata=type("Meta", (), {"file_path": type("Path", (), {"stem": "test"})(), "technique": "", "instrument_model": ""})(),  # type: ignore[arg-type]
    )


class PreprocessingTests(unittest.TestCase):
    def test_nan_inf_removal(self) -> None:
        freq = np.array([1e4, 1e3, 1e2, 1e1])
        z_real = np.array([5.0, np.nan, 6.0, 7.0])
        z_imag = np.array([-0.1, -0.5, np.inf, -2.0])
        spectrum = _make_spectrum(freq, z_real, z_imag)
        result = preprocess_for_fitting(spectrum, mask_inductive=False, mask_outliers=False)
        self.assertEqual(int(result.mask.sum()), 2)
        self.assertTrue(result.mask[0])
        self.assertTrue(result.mask[3])
        self.assertIn("NaN/Inf", result.actions[0])

    def test_inductive_loop_masking(self) -> None:
        freq = np.array([1e5, 1e4, 1e3, 1e2])
        z_real = np.array([5.0, 5.1, 5.5, 6.0])
        z_imag = np.array([0.05, 0.01, -0.5, -1.0])  # first two are inductive (Z'' > 0)
        spectrum = _make_spectrum(freq, z_real, z_imag)
        result = preprocess_for_fitting(spectrum, mask_nan=False, mask_outliers=False)
        self.assertEqual(int(result.mask.sum()), 2)
        self.assertFalse(result.mask[0])
        self.assertFalse(result.mask[1])
        self.assertIn("感抗", result.actions[0])

    def test_no_cleaning_when_all_valid(self) -> None:
        freq = np.array([1e4, 1e3, 1e2])
        z_real = np.array([5.0, 5.5, 6.0])
        z_imag = np.array([-0.1, -0.5, -1.0])
        spectrum = _make_spectrum(freq, z_real, z_imag)
        result = preprocess_for_fitting(spectrum)
        self.assertTrue(result.mask.all())
        self.assertEqual(len(result.actions), 0)

    def test_mad_outlier_detection_masks_isolated_spike(self) -> None:
        freq = np.logspace(5, 0, 11)
        z_real = np.linspace(5.0, 8.0, 11)
        z_imag = -np.array([0.1, 0.25, 0.5, 0.9, 1.2, 5.5, 1.1, 0.85, 0.6, 0.3, 0.1])
        spectrum = _make_spectrum(freq, z_real, z_imag)
        result = preprocess_for_fitting(spectrum, mask_nan=False, mask_inductive=False, mask_outliers=True)
        self.assertFalse(result.mask[5])
        self.assertIn(5, result.outlier_indices)
        self.assertTrue(any("异常点" in action for action in result.actions))


if __name__ == "__main__":
    unittest.main()
