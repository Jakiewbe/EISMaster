from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from eismaster.io.chi import load_spectrum
from eismaster.matlab_drt import (
    MatlabDrtConfig,
    _build_matlab_batch_call,
    run_matlab_drt,
    stage_matlab_drt_inputs,
    validate_matlab_drt_config,
)
from tests.fixture_factory import write_single_arc_txt

ROOT = Path(__file__).resolve().parents[1]


class MatlabDrtTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "_tmp_matlab_drt"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.txt_sample = write_single_arc_txt(self.tmp_root / "Ag_EIS_OCV.txt")

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_stage_inputs_uses_raw_z_imag(self) -> None:
        spectrum = load_spectrum(self.txt_sample)
        staging = stage_matlab_drt_inputs([spectrum], self.tmp_root)
        lines = (staging / "Ag_EIS_OCV.txt").read_text(encoding="utf-8").splitlines()
        first = lines[0].split("\t")
        self.assertAlmostEqual(float(first[0]), float(spectrum.freq_hz[0]))
        self.assertAlmostEqual(float(first[2]), float(spectrum.z_imag_ohm[0]))

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

    def test_missing_matlab_executable_is_rejected(self) -> None:
        config = MatlabDrtConfig(matlab_exe=str(self.tmp_root / "missing.exe"), drttools_dir=str(self.tmp_root))
        with self.assertRaisesRegex(FileNotFoundError, "MATLAB executable"):
            validate_matlab_drt_config(config)

    def test_nonzero_matlab_exit_is_rejected(self) -> None:
        matlab = self.tmp_root / "matlab.exe"
        matlab.write_bytes(b"")
        drttools = self.tmp_root / "drttools"
        drttools.mkdir()
        runner = self.tmp_root / "matlab_bridge" / "eismaster_batch_drt.m"
        runner.parent.mkdir()
        runner.write_text("function eismaster_batch_drt(varargin)\nend\n", encoding="utf-8")
        staging_dir = self.tmp_root / "staging"
        staging_dir.mkdir()
        output_dir = self.tmp_root / "results"
        valid_config = MatlabDrtConfig(matlab_exe=str(matlab), drttools_dir=str(drttools))
        completed = subprocess.CompletedProcess(["matlab"], returncode=1, stdout="", stderr="DRT failed")
        with patch("eismaster.matlab_drt._matlab_runner_path", return_value=runner), patch(
            "eismaster.matlab_drt.subprocess.run", return_value=completed
        ), self.assertRaisesRegex(RuntimeError, "DRT failed"):
            run_matlab_drt(valid_config, staging_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
