from __future__ import annotations
from typing import Optional, Union, List, Dict, Any, Tuple
import typing







"""Pre-fitting data guard — generates a point mask without mutating the
original ``SpectrumData``.

Three optional cleaning passes:

1. **NaN / Inf removal** — unconditional.
2. **Inductive-loop masking** — high-frequency points where Z'' > 0.
3. **Statistical outlier masking** — curvature + local-slope deviation.
"""


from dataclasses import dataclass, field

import numpy as np

from eismaster.models import SpectrumData


@dataclass(frozen=True)
class PreprocessResult:
    mask: np.ndarray  # bool, True = keep
    actions: list[str]
    outlier_indices: tuple[int, ...]


def preprocess_for_fitting(
    spectrum: SpectrumData,
    existing_mask: Optional[np.ndarray] = None,
    *,
    mask_nan: bool = True,
    mask_inductive: bool = True,
    mask_outliers: bool = True,
) -> PreprocessResult:
    """Return a refined *point_mask* and a human-readable action log."""

    n = spectrum.n_points
    mask = np.ones(n, dtype=bool) if existing_mask is None else existing_mask.copy()
    actions: list[str] = []
    outlier_indices: list[int] = []

    # --- pass 1: NaN / Inf ------------------------------------------------
    if mask_nan:
        bad = (
            ~np.isfinite(spectrum.freq_hz)
            | ~np.isfinite(spectrum.z_real_ohm)
            | ~np.isfinite(spectrum.z_imag_ohm)
        )
        count = int((mask & bad).sum())
        if count:
            mask &= ~bad
            actions.append(f"mask äº?{count} ä¸ªå« NaN/Inf çç¹")

    # --- pass 2: inductive loop (Z'' > 0 at high-freq end) ----------------
    if mask_inductive:
        inductive = spectrum.z_imag_ohm > 0
        # only mask a leading run of inductive points (high-freq end)
        leading_count = 0
        for i in range(n):
            if inductive[i] and mask[i]:
                leading_count += 1
            else:
                break
        if leading_count > 0:
            mask[:leading_count] = False
            actions.append(f"mask äºé«é¢ç«¯ {leading_count} ä¸ªææç¹ (Z'' > 0)")

    # --- pass 3: statistical outliers -------------------------------------
    if mask_outliers:
        flagged = _detect_outliers_enhanced(spectrum, mask)
        new_outliers = np.flatnonzero(flagged & mask)
        if new_outliers.size:
            mask[new_outliers] = False
            outlier_indices.extend(int(i) for i in new_outliers)
            actions.append(f"mask äº?{new_outliers.size} ä¸ªç»è®¡ç¦»ç¾¤ç¹")

    return PreprocessResult(
        mask=mask,
        actions=actions,
        outlier_indices=tuple(outlier_indices),
    )


def _detect_outliers_enhanced(
    spectrum: SpectrumData, mask: np.ndarray
) -> np.ndarray:
    """Curvature + local-slope outlier detection on the Nyquist trace."""
    flags = np.zeros(spectrum.n_points, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size < 7:
        return flags

    x = spectrum.z_real_ohm[idx]
    y = -spectrum.z_imag_ohm[idx]

    # curvature-based
    curv = np.zeros(idx.size)
    curv[1:-1] = np.abs(y[:-2] - 2 * y[1:-1] + y[2:])
    med_c = np.median(curv)
    std_c = np.std(curv)
    thresh_c = med_c + 4.0 * max(std_c, 1e-12)

    # slope-jump based
    dx = np.diff(x)
    dy = np.diff(y)
    slopes = np.arctan2(dy, np.where(np.abs(dx) < 1e-30, 1e-30, dx))
    slope_diff = np.zeros(idx.size)
    slope_diff[1:-1] = np.abs(np.diff(slopes))
    med_s = np.median(slope_diff)
    std_s = np.std(slope_diff)
    thresh_s = med_s + 4.0 * max(std_s, 1e-12)

    for k in range(1, idx.size - 1):
        if curv[k] > thresh_c or slope_diff[k] > thresh_s:
            flags[idx[k]] = True

    return flags
