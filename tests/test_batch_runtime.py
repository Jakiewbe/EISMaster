from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from eismaster.analysis.batch import analyze_batch_auto
from eismaster.io.chi import load_spectrum


ROOT = Path(__file__).resolve().parents[1]
TXT_SAMPLE = ROOT / "Ag_EIS_OCV.txt"


class BrokenExecutor:
    def __init__(self, *args, **kwargs) -> None:
        raise PermissionError("executor blocked")


class BatchRuntimeTests(unittest.TestCase):
    def test_analyze_batch_auto_falls_back_to_sequential_when_pool_unavailable(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        with patch("eismaster.analysis.batch.concurrent.futures.ProcessPoolExecutor", BrokenExecutor):
            summary = analyze_batch_auto([spectrum])
        self.assertEqual(summary.model_key, "auto")
        self.assertEqual(len(summary.items), 1)
        self.assertIsNotNone(summary.items[0].fit)
        self.assertIn(summary.items[0].fit.status, {"ok", "warn", "failed"})


if __name__ == "__main__":
    unittest.main()
