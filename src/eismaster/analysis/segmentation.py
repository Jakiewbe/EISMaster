from __future__ import annotations
from typing import Optional








from dataclasses import dataclass

import numpy as np

from eismaster.models import SpectrumData


@dataclass(frozen=True)
class SegmentDetection:
    requested_mode: str
    resolved_mode: str
    peak_indices: tuple[int, ...]
    split_indices: tuple[int, ...]


def detect_segments(
    spectrum: SpectrumData,
    mode: str = "auto",
    manual_split1: Optional[int] = None,
    manual_split2: Optional[int] = None,
    manual_peak1: Optional[int] = None,
    manual_peak2: Optional[int] = None,
) -> SegmentDetection:
    raw_y = spectrum.minus_z_imag_ohm.astype(float, copy=False)
    y = _smooth_trace(raw_y)
    peaks = _significant_peaks(y)
    if mode == "double" and len(peaks) < 2:
        peaks = _fallback_double_peaks(raw_y)
    elif mode == "auto" and len(peaks) < 2:
        fallback = _fallback_double_peaks(raw_y)
        if len(fallback) >= 2:
            peaks = fallback

    if mode == "single":
        resolved_mode = "single"
    elif mode == "double":
        resolved_mode = "double"
    else:
        resolved_mode = "double" if len(peaks) >= 2 else "single"

    if resolved_mode == "double" and len(peaks) < 2:
        # If explicitly requested double arc, or auto decided double but later failed (unlikely),
        # force 2 peaks to prevent crash and respect the double mode selection.
        if len(peaks) == 1:
            peaks = [peaks[0], min(peaks[0] + 5, len(y) - 2)]
        else:
            peaks = [len(y) // 3, 2 * len(y) // 3]

    if resolved_mode == "single":
        peak = peaks[0] if peaks else int(np.argmax(y))
        split1 = _valley_after(y, peak, len(y) - 4)
        if manual_peak1 is not None and 0 <= manual_peak1 < len(y):
            peak = int(manual_peak1)
        if manual_split1 is not None and 1 <= manual_split1 < len(y) - 3:
            split1 = int(manual_split1)
        peak, split1 = _sanitize_single_controls(len(y), peak, split1)
        return SegmentDetection(
            requested_mode=mode,
            resolved_mode="single",
            peak_indices=(peak,),
            split_indices=(split1,),
        )

    peak1, peak2 = peaks[:2]
    split1 = _valley_between(y, peak1, peak2)
    split2 = _valley_after(y, peak2, len(y) - 4)
    if manual_peak1 is not None and 0 <= manual_peak1 < len(y):
        peak1 = int(manual_peak1)
    if manual_peak2 is not None and 0 <= manual_peak2 < len(y):
        peak2 = int(manual_peak2)
    if manual_split1 is not None and 1 <= manual_split1 < len(y) - 2:
        split1 = int(manual_split1)
    if manual_split2 is not None and 1 <= manual_split2 < len(y) - 1:
        split2 = int(manual_split2)
    peak1, split1, peak2, split2 = _sanitize_double_controls(len(y), peak1, split1, peak2, split2)
    return SegmentDetection(
        requested_mode=mode,
        resolved_mode="double",
        peak_indices=(peak1, peak2),
        split_indices=(split1, split2),
    )


def _smooth_trace(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values.astype(float, copy=True)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _significant_peaks(y: np.ndarray) -> list[int]:
    if y.size < 5:
        return [int(np.argmax(y))] if y.size else []
    indices = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if indices.size == 0:
        return [int(np.argmax(y))]
    kept: list[int] = []
    for index in indices:
        left = y[max(0, index - 5) : index + 1]
        right = y[index : min(len(y), index + 6)]
        baseline = max(float(np.min(left)), float(np.min(right)))
        prominence = float(y[index]) - baseline
        if prominence < max(float(y[index]) * 0.01, 0.2):
            continue
        if kept and index - kept[-1] < 4:
            if y[index] > y[kept[-1]]:
                kept[-1] = int(index)
            continue
        kept.append(int(index))
    if kept:
        return kept

    # Broad, shallow semicircles can have a very weak local prominence in a
    # small neighborhood even though they still define the correct arc peak.
    # Falling back to the global maximum pushes the peak to the low-frequency
    # end for continuously rising tails, which then forces a pathological split.
    fallback: list[int] = []
    for index in indices:
        if fallback and index - fallback[-1] < 4:
            if y[index] > y[fallback[-1]]:
                fallback[-1] = int(index)
            continue
        fallback.append(int(index))
    return fallback or [int(np.argmax(y))]


def _valley_between(y: np.ndarray, start: int, stop: int) -> int:
    if stop <= start + 1:
        return min(start + 1, len(y) - 2)
    offset = int(np.argmin(y[start:stop + 1]))
    return start + offset


def _valley_after(y: np.ndarray, start: int, stop: int) -> int:
    search_end = min(stop, len(y) - 1)
    if search_end <= start + 1:
        return min(start + 1, len(y) - 2)
    segment = y[start + 1 : search_end + 1]
    offset = int(np.argmin(segment))
    candidate = start + 1 + offset
    # If the detected valley is right at the peak edge (arc keeps growing with
    # no tail valley) and the tail region would be too small for a stable fit,
    # place the split further along to guarantee enough tail points.
    tail_needed = 6
    if candidate <= start + 2 and (search_end - candidate) < tail_needed:
        fallback = search_end - tail_needed
        if fallback > start + 2:
            return fallback
    return candidate


def _fallback_double_peaks(y: np.ndarray) -> list[int]:
    if y.size < 12:
        return [int(np.argmax(y))] if y.size else []
    indices = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if indices.size == 0:
        return [int(np.argmax(y))]
    stop = max(int(len(y) * 0.85), 2)
    candidates = [int(index) for index in indices if 2 <= index <= stop]
    if len(candidates) < 2:
        candidates = [int(index) for index in indices]
    ranked = sorted(candidates, key=lambda index: float(y[index]), reverse=True)
    selected: list[int] = []
    for index in ranked:
        if not selected:
            selected.append(index)
            continue
        if all(abs(index - prev) >= 6 for prev in selected):
            selected.append(index)
        if len(selected) == 2:
            break
    return sorted(selected) if len(selected) >= 2 else ([int(np.argmax(y))] if y.size else [])


def _sanitize_single_controls(n_points: int, peak: int, split1: int) -> tuple[int, int]:
    if n_points <= 6:
        peak = int(np.clip(peak, 0, max(n_points - 2, 0)))
        split1 = int(np.clip(split1, min(peak + 1, n_points - 1), max(n_points - 2, 0)))
        return peak, split1
    peak = int(np.clip(peak, 1, n_points - 5))
    split1 = int(np.clip(split1, peak + 1, n_points - 4))
    return peak, split1


def _sanitize_double_controls(
    n_points: int,
    peak1: int,
    split1: int,
    peak2: int,
    split2: int,
) -> tuple[int, int, int, int]:
    if n_points <= 10:
        peak1 = int(np.clip(peak1, 0, max(n_points - 4, 0)))
        split1 = int(np.clip(split1, peak1 + 1, max(n_points - 3, 1)))
        peak2 = int(np.clip(peak2, split1 + 1, max(n_points - 2, split1 + 1)))
        split2 = int(np.clip(split2, peak2 + 1, max(n_points - 1, peak2 + 1)))
        return peak1, split1, peak2, split2

    peak1 = int(np.clip(peak1, 1, n_points - 7))
    split1 = int(np.clip(split1, peak1 + 1, n_points - 6))
    peak2 = int(np.clip(peak2, split1 + 1, n_points - 5))
    split2 = int(np.clip(split2, peak2 + 1, n_points - 4))
    return peak1, split1, peak2, split2


@dataclass(frozen=True)
class ArcRange:
    """User-defined arc range from a RangeSlider.

    Attributes:
        start: First index of the arc region.
        end:   Last index of the arc region (also serves as the split point
               to the next region).
    """
    start: int
    end: int
