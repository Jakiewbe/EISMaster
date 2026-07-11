from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pytest

from eismaster.exporters import build_drt_matrices, write_drt_only_export
from eismaster.models import SpectrumData, SpectrumMetadata


def _spectrum(path: Path) -> SpectrumData:
    freq = np.array([1000.0, 100.0, 10.0])
    real = np.array([5.0, 6.0, 8.0])
    imag = np.array([-0.2, -0.7, -1.1])
    return SpectrumData(
        metadata=SpectrumMetadata(file_path=path),
        freq_hz=freq,
        z_real_ohm=real,
        z_imag_ohm=imag,
        z_mod_ohm=np.hypot(real, imag),
        phase_deg=np.degrees(np.arctan2(imag, real)),
    )


def _write_drt_result(directory: Path, stem: str) -> None:
    (directory / f"{stem}_DRT.txt").write_text(
        "tau\tgamma(tau)\n1\t1\n10\t2\n100\t3\n",
        encoding="utf-8",
    )


def test_drt_workbook_contract(tmp_path: Path) -> None:
    spectrum = _spectrum(tmp_path / "Ag_S01_EIS_T5M.txt")
    _write_drt_result(tmp_path, spectrum.metadata.file_path.stem)
    paths = write_drt_only_export(tmp_path / "drt.xlsx", [spectrum], tmp_path, fmt="xlsx")
    workbook = openpyxl.load_workbook(paths["drt"], data_only=True)
    try:
        assert workbook.sheetnames == ["drt_line_plot", "drt_cloud_density", "drt_quant_area"]
        assert workbook["drt_line_plot"]["A2"].value in {"logtau", "tau/s"}
        assert workbook["drt_cloud_density"]["A2"].value == "logtau"
        assert workbook["drt_quant_area"]["A1"].value == "sample"
    finally:
        workbook.close()


def test_drt_line_axis_rejects_unknown_value(tmp_path: Path) -> None:
    spectrum = _spectrum(tmp_path / "Ag_S01_EIS_T5M.txt")
    _write_drt_result(tmp_path, spectrum.metadata.file_path.stem)
    with pytest.raises(ValueError, match="line_x_axis"):
        build_drt_matrices([spectrum], tmp_path, line_x_axis="frequency")
