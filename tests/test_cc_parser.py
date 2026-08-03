from __future__ import annotations

from pathlib import Path

import pytest

from eismaster.io.cc import load_cc_segments_from_folder, parse_cc_bin
from tests.cc_fixtures import write_cc_bin, write_eis_marker


def test_parse_cc_bin_decodes_header_interval_direction_and_voltage(tmp_path: Path) -> None:
    path = write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 3.2, 4.0], current_raw=-18128)

    segment = parse_cc_bin(path)

    assert segment.sample_id == "S01"
    assert segment.target_voltage_v == pytest.approx(4.0)
    assert segment.mode == "charge"
    assert segment.sample_interval_s == pytest.approx(0.01)
    assert segment.voltage_v.tolist() == pytest.approx([3.0, 3.2, 4.0])


def test_parse_cc_bin_marks_positive_current_as_discharge(tmp_path: Path) -> None:
    path = write_cc_bin(tmp_path / "L_S03_CC_2p00V.bin", [2.2, 2.1, 2.0], current_raw=14640)

    segment = parse_cc_bin(path)

    assert segment.sample_id == "S03"
    assert segment.target_voltage_v == pytest.approx(2.0)
    assert segment.mode == "discharge"


def test_folder_loader_skips_eis_and_unknown_bin(tmp_path: Path) -> None:
    write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [3.0, 4.0], current_raw=-18128)
    write_eis_marker(tmp_path / "L_S01_EIS_4p00V.bin")
    (tmp_path / "notes.bin").write_bytes(b"unknown")

    segments = load_cc_segments_from_folder(tmp_path)

    assert [item.source_path.name for item in segments] == ["L_S01_CC_4p00V.bin"]


def test_parse_cc_bin_rejects_voltage_direction_conflict(tmp_path: Path) -> None:
    path = write_cc_bin(tmp_path / "L_S01_CC_4p00V.bin", [4.0, 3.5], current_raw=-18128)

    with pytest.raises(ValueError, match="voltage direction"):
        parse_cc_bin(path)
