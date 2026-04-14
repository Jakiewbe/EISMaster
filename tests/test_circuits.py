from __future__ import annotations

import unittest

import numpy as np

from eismaster.analysis.circuits import (
    CircuitTemplate,
    TEMPLATES,
    get_circuit_templates,
    _parallel,
    _cpe,
    _warburg,
)


class CircuitTests(unittest.TestCase):
    def test_templates_exist(self) -> None:
        self.assertIn("zview_segmented_rq_rwo", TEMPLATES)
        self.assertIn("zview_double_rq_qrwo", TEMPLATES)

    def test_get_circuit_templates_returns_copy(self) -> None:
        templates = get_circuit_templates()
        self.assertIsInstance(templates, dict)
        self.assertIsNot(templates, TEMPLATES)

    def test_circuit_template_structure(self) -> None:
        tmpl = TEMPLATES["zview_segmented_rq_rwo"]
        self.assertIsInstance(tmpl, CircuitTemplate)
        self.assertEqual(tmpl.label, "Single-arc R(QRWo)")
        self.assertEqual(len(tmpl.parameter_names), 8)
        self.assertIn("Rs", tmpl.primary_exports)
        self.assertIn("Rct", tmpl.primary_exports)

    def test_parallel_impedance(self) -> None:
        z1 = np.array([1.0 + 0j, 2.0 + 0j])
        z2 = np.array([1.0 + 0j, 2.0 + 0j])
        result = _parallel(z1, z2)
        expected = np.array([0.5 + 0j, 1.0 + 0j])
        np.testing.assert_array_almost_equal(result, expected)

    def test_cpe_impedance(self) -> None:
        omega = np.array([2 * np.pi * 1000.0])
        z = _cpe(q=1e-5, n=0.9, omega=omega)
        self.assertEqual(z.shape, (1,))
        self.assertTrue(np.isfinite(z.real) and np.isfinite(z.imag))

    def test_warburg_impedance(self) -> None:
        omega = np.array([2 * np.pi * 1000.0])
        z = _warburg(sigma=100.0, omega=omega)
        self.assertEqual(z.shape, (1,))
        self.assertTrue(np.isfinite(z.real) and np.isfinite(z.imag))


if __name__ == "__main__":
    unittest.main()
