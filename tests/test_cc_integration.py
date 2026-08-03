from __future__ import annotations

from pathlib import Path

import pytest

from eismaster.cc_curves import merge_cc_segments
from eismaster.io.cc import load_cc_segments_from_folder

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "L").is_dir(), reason="user-provided L folder is absent")
def test_real_l_folder_extracts_only_cc_and_merges_continuous_curves() -> None:
    segments = load_cc_segments_from_folder(ROOT / "L")
    curves = merge_cc_segments(segments)

    assert len(segments) == 10
    assert all("_EIS_" not in segment.source_path.name for segment in segments)
    assert {curve.mode for curve in curves} == {"charge", "discharge"}
    assert {curve.sample_id for curve in curves} == {"S01", "S03"}
    for curve in curves:
        assert (curve.total_time_s[1:] > curve.total_time_s[:-1]).all()
        assert all("_EIS_" not in name for name in curve.source_file)
