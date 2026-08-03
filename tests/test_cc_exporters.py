from __future__ import annotations

from pathlib import Path

import openpyxl

from eismaster.cc_curves import merge_cc_segments
from eismaster.cc_exporters import (
    build_cc_curve_table,
    build_cc_segment_summary,
    export_cc_curves,
)
from tests.test_cc_curves import segment


def test_curve_export_contains_traceability_columns() -> None:
    curves = merge_cc_segments([segment("S01", "charge", 4.0, [3.0, 4.0])])

    table = build_cc_curve_table(curves)

    assert list(table.columns) == [
        "sample_id",
        "mode",
        "total_time_s",
        "segment_time_s",
        "voltage_v",
        "segment_index",
        "source_file",
    ]
    assert table["source_file"].notna().all()


def test_segment_summary_contains_duration_and_endpoints() -> None:
    curves = merge_cc_segments([segment("S01", "charge", 4.0, [3.0, 4.0])])

    summary = build_cc_segment_summary(curves)

    assert list(summary.columns) == [
        "sample_id",
        "mode",
        "segment_index",
        "source_file",
        "target_voltage_v",
        "start_voltage_v",
        "end_voltage_v",
        "point_count",
        "duration_s",
        "order",
    ]
    assert summary.loc[0, "duration_s"] == 0.01


def test_xlsx_export_has_curve_and_segment_sheets(tmp_path: Path) -> None:
    curves = merge_cc_segments([segment("S01", "charge", 4.0, [3.0, 4.0])])

    paths = export_cc_curves(tmp_path / "cc.xlsx", curves, fmt="xlsx")

    workbook = openpyxl.load_workbook(paths["curve"], read_only=True)
    assert workbook.sheetnames == ["cc_curve", "cc_segments"]
