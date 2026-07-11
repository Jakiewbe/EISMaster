from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eismaster.analysis.batch import analyze_batch_auto
from eismaster.io.chi import load_spectrum
from tests.fixture_factory import write_single_arc_txt


class BrokenExecutor:
    def __init__(self, *args, **kwargs) -> None:
        raise PermissionError("executor blocked")


class BatchRuntimeTests(unittest.TestCase):
    def test_analyze_batch_auto_falls_back_to_sequential_when_pool_unavailable(self) -> None:
        with TemporaryDirectory() as tmp:
            sample = write_single_arc_txt(Path(tmp) / "Ag_EIS_OCV.txt")
            spectrum = load_spectrum(sample)
            with patch("eismaster.analysis.batch.concurrent.futures.ProcessPoolExecutor", BrokenExecutor):
                summary = analyze_batch_auto([spectrum])
        self.assertEqual(summary.model_key, "auto")
        self.assertEqual(len(summary.items), 1)
        self.assertIsNotNone(summary.items[0].fit)
        self.assertIn(summary.items[0].fit.status, {"ok", "warn", "failed"})


if __name__ == "__main__":
    unittest.main()
