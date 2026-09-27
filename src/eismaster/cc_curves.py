from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from eismaster.io.cc import CCSegment


@dataclass(frozen=True)
class CCCurve:
    sample_id: str
    mode: str
    segments: tuple[CCSegment, ...]
    total_time_s: np.ndarray
    segment_time_s: np.ndarray
    voltage_v: np.ndarray
    segment_index: np.ndarray
    source_file: list[str]

    @property
    def n_points(self) -> int:
        return int(self.voltage_v.size)

    @property
    def duration_s(self) -> float:
        return float(self.total_time_s[-1]) if self.n_points else 0.0


def merge_cc_segments(segments: list[CCSegment]) -> list[CCCurve]:
    grouped: dict[tuple[str, str], list[CCSegment]] = defaultdict(list)
    for segment in segments:
        if segment.mode not in {"charge", "discharge"}:
            raise ValueError(f"Unsupported CC mode: {segment.mode}")
        grouped[(segment.sample_id, segment.mode)].append(segment)

    curves: list[CCCurve] = []
    for (sample_id, mode), group in sorted(grouped.items()):
        if any(segment.sequence_key is not None for segment in group):
            ordered = sorted(
                group,
                key=lambda segment: float("inf") if segment.sequence_key is None else segment.sequence_key,
            )
        else:
            ordered = sorted(
                group,
                key=lambda segment: segment.target_voltage_v,
                reverse=mode == "discharge",
            )
        total_time_parts: list[np.ndarray] = []
        segment_time_parts: list[np.ndarray] = []
        voltage_parts: list[np.ndarray] = []
        segment_index_parts: list[np.ndarray] = []
        source_files: list[str] = []
        offset_s = 0.0
        for index, segment in enumerate(ordered, start=1):
            segment_time = np.arange(segment.n_points, dtype=float) * segment.sample_interval_s
            total_time_parts.append(segment_time + offset_s)
            segment_time_parts.append(segment_time)
            voltage_parts.append(segment.voltage_v.copy())
            segment_index_parts.append(np.full(segment.n_points, index, dtype=int))
            source_files.extend([segment.source_path.name] * segment.n_points)
            offset_s += segment.n_points * segment.sample_interval_s

        curves.append(
            CCCurve(
                sample_id=sample_id,
                mode=mode,
                segments=tuple(ordered),
                total_time_s=np.concatenate(total_time_parts),
                segment_time_s=np.concatenate(segment_time_parts),
                voltage_v=np.concatenate(voltage_parts),
                segment_index=np.concatenate(segment_index_parts),
                source_file=source_files,
            )
        )
    return curves
