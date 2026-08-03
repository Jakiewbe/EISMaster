from __future__ import annotations

from pathlib import Path

import pandas as pd

from eismaster.cc_curves import CCCurve


CC_CURVE_COLUMNS = [
    "sample_id",
    "mode",
    "total_time_s",
    "segment_time_s",
    "voltage_v",
    "segment_index",
    "source_file",
]
CC_SEGMENT_COLUMNS = [
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


def build_cc_curve_table(curves: list[CCCurve]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for curve in curves:
        for index in range(curve.n_points):
            rows.append(
                {
                    "sample_id": curve.sample_id,
                    "mode": curve.mode,
                    "total_time_s": float(curve.total_time_s[index]),
                    "segment_time_s": float(curve.segment_time_s[index]),
                    "voltage_v": float(curve.voltage_v[index]),
                    "segment_index": int(curve.segment_index[index]),
                    "source_file": curve.source_file[index],
                }
            )
    return pd.DataFrame(rows, columns=CC_CURVE_COLUMNS)


def build_cc_segment_summary(curves: list[CCCurve]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    order = 0
    for curve in curves:
        for segment_index, segment in enumerate(curve.segments, start=1):
            order += 1
            rows.append(
                {
                    "sample_id": curve.sample_id,
                    "mode": curve.mode,
                    "segment_index": segment_index,
                    "source_file": segment.source_path.name,
                    "target_voltage_v": segment.target_voltage_v,
                    "start_voltage_v": float(segment.voltage_v[0]),
                    "end_voltage_v": float(segment.voltage_v[-1]),
                    "point_count": segment.n_points,
                    "duration_s": segment.duration_s,
                    "order": order,
                }
            )
    return pd.DataFrame(rows, columns=CC_SEGMENT_COLUMNS)


def export_cc_curves(target: str | Path, curves: list[CCCurve], fmt: str = "xlsx") -> dict[str, Path]:
    if not curves:
        raise ValueError("No CC curves to export")
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    curve_table = build_cc_curve_table(curves)
    segment_table = build_cc_segment_summary(curves)

    if fmt.lower() == "xlsx":
        with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
            curve_table.to_excel(writer, sheet_name="cc_curve", index=False)
            segment_table.to_excel(writer, sheet_name="cc_segments", index=False)
        return {"curve": target_path, "segments": target_path}

    if fmt.lower() in {"txt", "csv"}:
        suffix = target_path.suffix or ".txt"
        stem = target_path.with_suffix("")
        curve_path = stem.with_name(stem.name + "_curve").with_suffix(suffix)
        segment_path = stem.with_name(stem.name + "_segments").with_suffix(suffix)
        curve_table.to_csv(curve_path, index=False)
        segment_table.to_csv(segment_path, index=False)
        return {"curve": curve_path, "segments": segment_path}

    raise ValueError(f"Unsupported CC export format: {fmt}")
