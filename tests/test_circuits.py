from __future__ import annotations

import unittest

from eismaster.analysis.circuits import (
    CircuitTemplate,
    TEMPLATES,
    get_circuit_templates,
)


class CircuitTests(unittest.TestCase):
    def test_templates_exist(self) -> None:
        self.assertIn("zview_segmented_rq_rwo", TEMPLATES)
        self.assertIn("zview_double_rq_qrwo", TEMPLATES)
        self.assertEqual(len(TEMPLATES), 2)

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

    def test_double_template_structure(self) -> None:
        tmpl = TEMPLATES["zview_double_rq_qrwo"]
        self.assertIsInstance(tmpl, CircuitTemplate)
        self.assertEqual(tmpl.label, "Double-arc R(QR)(Q(RWo))")
        self.assertIn("Rs", tmpl.primary_exports)
        self.assertIn("Rsei", tmpl.primary_exports)
        self.assertIn("Rct", tmpl.primary_exports)


if __name__ == "__main__":
    unittest.main()
