from __future__ import annotations

import unittest

import numpy as np

from eismaster.analysis.native_drt import compute_drt, find_drt_peaks
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


class NativeDrtTests(unittest.TestCase):
    def test_compute_drt_returns_valid_arrays(self) -> None:
        freq = np.logspace(4, -2, 50)
        z_real = np.ones(50) * 10.0 + np.linspace(0, 50, 50)
        z_imag = -np.linspace(0, 30, 50)
        spectrum = _make_spectrum(freq, z_real, z_imag)
        tau, gamma, rs = compute_drt(spectrum)
        self.assertEqual(tau.shape, gamma.shape)
        self.assertTrue(np.all(tau > 0))
        self.assertTrue(np.all(gamma >= 0))
        self.assertTrue(np.isfinite(rs))

    def test_compute_drt_can_auto_choose_lambda(self) -> None:
        freq = np.logspace(4, -2, 40)
        z_real = np.ones(40) * 4.0 + np.linspace(0, 20, 40)
        z_imag = -np.linspace(0, 12, 40)
        spectrum = _make_spectrum(freq, z_real, z_imag)
        tau, gamma, rs = compute_drt(spectrum, lambda_reg=None, mode="fast")
        self.assertEqual(tau.shape, gamma.shape)
        self.assertTrue(np.all(gamma >= 0))
        self.assertTrue(np.isfinite(rs))

    def test_find_drt_peaks_with_single_artificial_peak(self) -> None:
        tau = np.logspace(-6, 2, 100)
        gamma = np.zeros(100)
        gamma[30] = 5.0  # inject a sharp peak
        peaks = find_drt_peaks(tau, gamma)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0]["tau"], float(tau[30]), places=0)
        self.assertGreater(peaks[0]["R"], 0.0)
        self.assertGreater(peaks[0]["n"], 0.0)

    def test_find_drt_peaks_returns_empty_for_flat_signal(self) -> None:
        tau = np.logspace(-6, 2, 50)
        gamma = np.zeros(50)
        peaks = find_drt_peaks(tau, gamma)
        self.assertEqual(len(peaks), 0)

    def test_drt_peak_info_has_required_keys(self) -> None:
        tau = np.logspace(-6, 2, 100)
        gamma = np.zeros(100)
        gamma[50] = 10.0
        peaks = find_drt_peaks(tau, gamma)
        self.assertIn("tau", peaks[0])
        self.assertIn("R", peaks[0])
        self.assertIn("n", peaks[0])


if __name__ == "__main__":
    unittest.main()
