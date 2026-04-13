from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from eismaster.io.chi import load_spectrum
from eismaster.matlab_drt import MatlabDrtConfig, _build_matlab_batch_call, stage_matlab_drt_inputs


ROOT = Path(__file__).resolve().parents[1]
TXT_SAMPLE = ROOT / "Ag_EIS_OCV.txt"


class MatlabDrtTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "_tmp_matlab_drt"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_stage_inputs_uses_raw_z_imag(self) -> None:
        spectrum = load_spectrum(TXT_SAMPLE)
        staging = stage_matlab_drt_inputs([spectrum], self.tmp_root)
        lines = (staging / "Ag_EIS_OCV.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("96680"))
        self.assertIn("-0.4091", lines[0])

    def test_build_batch_call_points_to_repo_runner(self) -> None:
        config = MatlabDrtConfig()
        call = _build_matlab_batch_call(
            runner_path=ROOT / "matlab_bridge" / "eismaster_batch_drt.m",
            input_dir=self.tmp_root / "in",
            output_dir=self.tmp_root / "out",
            config=config,
        )
        self.assertIn("eismaster_batch_drt(", call)
        self.assertIn("matlab_bridge", call)
        self.assertIn("matlab-DRTtools-local", call)


if __name__ == "__main__":
    unittest.main()
