from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eismaster.analysis.fitting import (
    _attach_diagnosis,
    _adaptive_weight_floor,
    _cnls_diagnostics,
    _cnls_selection_score,
    _estimate_cpe_n,
    _zview_warburg_open,
    _zview_residual,
)
from eismaster.analysis.segmentation import SegmentDetection
from eismaster.models import FitOutcome, SpectrumData, SpectrumMetadata


class FittingAlgorithmTests(unittest.TestCase):
    def test_adaptive_weight_floor_scales_with_data(self) -> None:
        z_exp = np.array([0.01 + 0.02j, 0.02 + 0.01j, 0.015 + 0.015j], dtype=complex)
        floor = _adaptive_weight_floor(z_exp)
        self.assertGreaterEqual(floor, 1e-6)
        self.assertLess(floor, 1e-3)

    def test_calc_unit_residual_is_unweighted(self) -> None:
        freq = np.array([1.0, 10.0])
        z_exp = np.array([1.0 + 2.0j, 2.0 + 3.0j])
        params = np.array([0.0])

        def model_fn(_freq: np.ndarray, _params: np.ndarray) -> np.ndarray:
            return np.array([1.5 + 1.0j, 1.0 + 5.0j])

        residual = _zview_residual(freq, z_exp, params, model_fn, "calc-unit")
        np.testing.assert_allclose(residual, np.array([0.5, -1.0, -1.0, 2.0]))

    def test_estimate_cpe_n_returns_reasonable_range(self) -> None:
        freq = np.logspace(5, 1, 20)
        z_imag_neg = 1.0 / np.sqrt(freq)
        n_est = _estimate_cpe_n(freq, z_imag_neg)
        self.assertGreaterEqual(n_est, 0.4)
        self.assertLessEqual(n_est, 1.0)

    def test_calc_modulus_zview_uses_model_magnitude(self) -> None:
        freq = np.array([1.0, 10.0])
        z_exp = np.array([1.0 + 0.0j, 4.0 + 0.0j])
        params = np.array([0.0])

        def model_fn(_freq: np.ndarray, _params: np.ndarray) -> np.ndarray:
            return np.array([2.0 + 0.0j, 2.0 + 0.0j])

        residual = _zview_residual(freq, z_exp, params, model_fn, "calc-modulus-zview")
        np.testing.assert_allclose(residual, np.array([0.5, -1.0, 0.0, 0.0]))

    def test_zview_warburg_open_remains_finite_near_zero(self) -> None:
        omega = np.array([1e-15, 1e-9, 1.0, 1e9], dtype=float)
        z = _zview_warburg_open(omega, wo_r=10.0, wo_t=1.0, wo_p=0.5)
        self.assertTrue(np.all(np.isfinite(z.real)))
        self.assertTrue(np.all(np.isfinite(z.imag)))

    def test_cnls_selection_score_penalizes_systematic_runs(self) -> None:
        jac = np.eye(8)
        alternating = SimpleNamespace(fun=np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float), jac=jac)
        systematic = SimpleNamespace(fun=np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float), jac=jac)
        alt_score = _cnls_selection_score(alternating, 2)
        sys_score = _cnls_selection_score(systematic, 2)
        self.assertGreater(sys_score, alt_score)

    def test_cnls_diagnostics_produces_ci_and_correlation(self) -> None:
        jac = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 1.5],
            ]
        )
        result = SimpleNamespace(fun=np.array([0.1, -0.2, 0.05, -0.05]), jac=jac, x=np.array([2.0, 4.0]))
        stats, warnings = _cnls_diagnostics(result, 2, ["Rs", "Rct"])
        self.assertIn("Rs_ci95_low", stats)
        self.assertIn("Rct_ci95_high", stats)
        self.assertIn("correlation_matrix_max", stats)
        self.assertIsInstance(warnings, list)

    def test_attach_diagnosis_populates_structured_fields(self) -> None:
        spectrum = SpectrumData(
            metadata=SpectrumMetadata(file_path=Path("synthetic.txt")),
            freq_hz=np.array([1000.0, 100.0, 10.0, 1.0]),
            z_real_ohm=np.array([1.0, 2.0, 3.0, 4.0]),
            z_imag_ohm=np.array([-0.1, -0.2, -0.3, -0.4]),
            z_mod_ohm=np.array([1.0, 2.0, 3.0, 4.0]),
            phase_deg=np.array([-5.0, -6.0, -7.0, -8.0]),
        )
        outcome = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="warn",
            message="split mismatch",
        )
        hint = SegmentDetection(requested_mode="single", resolved_mode="single", peak_indices=(1,), split_indices=(2,))
        diagnosed = _attach_diagnosis(outcome, spectrum, outcome.model_key, hint)
        self.assertEqual(diagnosed.diagnosis_type, "data_quality")
        self.assertEqual(diagnosed.diagnosis_severity, "error")
        self.assertTrue(diagnosed.diagnosis_explanation)
        self.assertIsInstance(diagnosed.diagnosis_suggestions, list)


if __name__ == "__main__":
    unittest.main()
