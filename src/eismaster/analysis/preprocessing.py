from __future__ import annotations

"""Pre-fitting data guard that generates a conservative point mask."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from eismaster.models import SpectrumData


@dataclass(frozen=True)
class PreprocessResult:
    mask: np.ndarray
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
    """Return a refined point mask and a human-readable action log."""

    n = spectrum.n_points
    mask = np.ones(n, dtype=bool) if existing_mask is None else existing_mask.copy()
    actions: list[str] = []
    outlier_indices: list[int] = []

    if mask_nan:
        bad = (
            ~np.isfinite(spectrum.freq_hz)
            | ~np.isfinite(spectrum.z_real_ohm)
            | ~np.isfinite(spectrum.z_imag_ohm)
        )
        count = int((mask & bad).sum())
        if count:
            mask &= ~bad
            actions.append(f"mask {count} 个 NaN/Inf 点")

    if mask_inductive:
        inductive = spectrum.z_imag_ohm > 0
        leading_count = 0
        for i in range(n):
            if inductive[i] and mask[i]:
                leading_count += 1
            else:
                break
        if leading_count > 0:
            mask[:leading_count] = False
            actions.append(f"mask 高频端 {leading_count} 个感抗点 (Z'' > 0)")

    if mask_outliers:
        flagged = _detect_outliers_enhanced(spectrum, mask)
        new_outliers = np.flatnonzero(flagged & mask)
        if new_outliers.size:
            mask[new_outliers] = False
            outlier_indices.extend(int(i) for i in new_outliers)
            actions.append(f"mask {new_outliers.size} 个高置信异常点")

    return PreprocessResult(mask=mask, actions=actions, outlier_indices=tuple(outlier_indices))


def _scaled_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def _detect_outliers_enhanced(spectrum: SpectrumData, mask: np.ndarray) -> np.ndarray:
    """Conservative curvature/slope/log-frequency outlier detection."""

    flags = np.zeros(spectrum.n_points, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size < 7:
        return flags

    x = spectrum.z_real_ohm[idx].astype(float)
    y = (-spectrum.z_imag_ohm[idx]).astype(float)
    log_f = np.log10(np.maximum(spectrum.freq_hz[idx].astype(float), 1e-30))

    curv = np.zeros(idx.size, dtype=float)
    curv[1:-1] = np.abs(y[:-2] - 2.0 * y[1:-1] + y[2:])
    med_c = float(np.median(curv))
    mad_c = _scaled_mad(curv)
    thresh_c = med_c + 5.0 * max(mad_c, 1e-12)

    dx = np.diff(x)
    dy = np.diff(y)
    slopes = np.arctan2(dy, np.where(np.abs(dx) < 1e-30, 1e-30, dx))
    slope_diff = np.zeros(idx.size, dtype=float)
    if slopes.size >= 2:
        slope_diff[1:-1] = np.abs(np.diff(slopes))
    med_s = float(np.median(slope_diff))
    mad_s = _scaled_mad(slope_diff)
    thresh_s = med_s + 5.0 * max(mad_s, 1e-12)

    log_grad = np.gradient(y, log_f, edge_order=1)
    smooth_jump = np.zeros(idx.size, dtype=float)
    if log_grad.size >= 3:
        second_grad = np.abs(np.diff(log_grad, n=2))
        if second_grad.size:
            smooth_jump[1:-1] = second_grad
    med_g = float(np.median(smooth_jump))
    mad_g = _scaled_mad(smooth_jump)
    thresh_g = med_g + 6.0 * max(mad_g, 1e-12)

    for k in range(1, idx.size - 1):
        score = 0
        if curv[k] > thresh_c:
            score += 1
        if slope_diff[k] > thresh_s:
            score += 1
        if smooth_jump[k] > thresh_g:
            score += 1
        if score >= 2:
            flags[idx[k]] = True

    return flags
