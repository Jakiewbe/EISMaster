from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eismaster.cc_curves import merge_cc_segments
from eismaster.io.cc import CCSegment


def segment(sample_id: str, mode: str, target_voltage: float, voltages: list[float]) -> CCSegment:
    return CCSegment(
        source_path=Path(f"{sample_id}_CC_{target_voltage:.2f}".replace(".", "p") + "V.bin"),
        sample_id=sample_id,
        target_voltage_v=target_voltage,
        mode=mode,
        sample_interval_s=0.01,
        voltage_v=np.asarray(voltages, dtype=float),
        current_sign_raw=-1 if mode == "charge" else 1,
    )


def test_charge_segments_are_sorted_by_target_and_time_is_continuous() -> None:
    segments = [
        segment("S01", "charge", 4.2, [4.0, 4.2]),
        segment("S01", "charge", 4.0, [3.0, 4.0]),
    ]

    curve = merge_cc_segments(segments)[0]

    assert curve.source_file == [
        "S01_CC_4p00V.bin",
        "S01_CC_4p00V.bin",
        "S01_CC_4p20V.bin",
        "S01_CC_4p20V.bin",
    ]
    assert curve.total_time_s.tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert curve.segment_index.tolist() == [1, 1, 2, 2]
    assert curve.voltage_v.tolist() == pytest.approx([3.0, 4.0, 4.0, 4.2])


def test_discharge_segments_are_sorted_by_target_descending_and_separated_from_charge() -> None:
    curves = merge_cc_segments(
        [
            segment("S01", "charge", 4.0, [3.0, 4.0]),
            segment("S03", "discharge", 4.2, [4.6, 4.2]),
            segment("S03", "discharge", 2.0, [2.2, 2.0]),
        ]
    )

    assert {(curve.sample_id, curve.mode) for curve in curves} == {("S01", "charge"), ("S03", "discharge")}
    discharge = next(curve for curve in curves if curve.mode == "discharge")
    assert discharge.source_file[0].endswith("4p20V.bin")
    assert discharge.source_file[2].endswith("2p00V.bin")
