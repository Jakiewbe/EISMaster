from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from eismaster.models import SpectrumData


_COMMON_MATLAB_PATHS = [
    r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe",
    r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe",
    r"C:\Program Files\MATLAB\R2023b\bin\matlab.exe",
    r"C:\Program Files\MATLAB\R2023a\bin\matlab.exe",
    r"D:\Matlabs\bin\matlab.exe",
]


def _find_matlab_exe() -> str:
    """Locate MATLAB executable via env var, PATH, or common install paths."""
    env_exe = os.environ.get("EISMASTER_MATLAB_EXE")
    if env_exe and Path(env_exe).is_file():
        return env_exe
    path_exe = shutil.which("matlab")
    if path_exe:
        return path_exe
    for candidate in _COMMON_MATLAB_PATHS:
        if Path(candidate).is_file():
            return candidate
    return ""


@dataclass()
class MatlabDrtConfig:
    matlab_exe: str = field(default_factory=_find_matlab_exe)
    drttools_dir: str = field(default_factory=lambda: str((_resource_root() / "matlab-DRTtools-local").resolve()))
    method_tag: str = "simple"
    drt_type: int = 2
    lambda_value: float = 1e-3
    coeff_value: float = 0.5
    derivative_order: str = "1st-order"
    data_used: str = "Combined Re-Im Data"
    inductance_mode: int = 1
    shape_control: str = "FWHM Coefficient"


@dataclass()
class MatlabDrtResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_files: list[Path]
    staging_dir: Path
    output_dir: Path


def stage_matlab_drt_inputs(spectra: Iterable[SpectrumData], base_output_dir: str | Path) -> Path:
    base = Path(base_output_dir)
    staging_dir = base / "matlab_drt_inputs"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for spectrum in spectra:
        target = staging_dir / f"{spectrum.metadata.file_path.stem}.txt"
        _write_raw_impedance_input(target, spectrum)
    return staging_dir


def run_matlab_drt(config: MatlabDrtConfig, staging_dir: str | Path, output_dir: str | Path) -> MatlabDrtResult:
    staging = Path(staging_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    runner_path = _matlab_runner_path()
    matlab_call = _build_matlab_batch_call(
        runner_path=runner_path,
        input_dir=staging,
        output_dir=output,
        config=config,
    )
    command = [config.matlab_exe, "-batch", matlab_call]
    completed = subprocess.run(command, capture_output=True, text=True)
    output_files = sorted(output.glob("*_DRT.txt"))
    return MatlabDrtResult(
        command=command,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_files=output_files,
        staging_dir=staging,
        output_dir=output,
    )


def _write_raw_impedance_input(path: Path, spectrum: SpectrumData) -> None:
    lines = []
    for freq, z_real, z_imag in zip(spectrum.freq_hz, spectrum.z_real_ohm, spectrum.z_imag_ohm):
        lines.append(f"{freq:.12g}\t{z_real:.12g}\t{z_imag:.12g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matlab_runner_path() -> Path:
    return _resource_root() / "matlab_bridge" / "eismaster_batch_drt.m"


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parents[2]


def _build_matlab_batch_call(
    *,
    runner_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: MatlabDrtConfig,
) -> str:
    runner_dir = runner_path.parent.as_posix()
    runner_name = runner_path.stem
    args = [
        input_dir.as_posix(),
        output_dir.as_posix(),
        Path(config.drttools_dir).as_posix(),
        config.method_tag,
        str(int(config.drt_type)),
        f"{config.lambda_value:.12g}",
        f"{config.coeff_value:.12g}",
        config.derivative_order,
        config.data_used,
        str(int(config.inductance_mode)),
        config.shape_control,
    ]
    matlab_args = ", ".join(_matlab_quote(arg) for arg in args)
    return f"addpath('{runner_dir}'); {runner_name}({matlab_args});"


def _matlab_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"