from __future__ import annotations

import math
from typing import Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict

import numpy as np
from scipy import optimize, signal

from eismaster.models import SpectrumData


class DrtPeakInfo(TypedDict):
    tau: float
    R: float
    n: float


def compute_drt(
    spectrum: SpectrumData,
    lambda_reg: Optional[float] = None,
    tau_points_per_decade: int = 15,
    mode: str = "fast",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute DRT using non-negative ridge regression with optional lambda search."""

    freq_hz = np.asarray(spectrum.freq_hz, dtype=float)
    z_real = np.asarray(spectrum.z_real_ohm, dtype=float)
    z_imag = np.asarray(spectrum.z_imag_ohm, dtype=float)

    omega = 2.0 * np.pi * freq_hz
    max_omega = float(np.max(omega))
    min_omega = float(np.min(omega))
    min_tau = max((1.0 / max_omega) * 0.1, 1e-7)
    max_tau = min((1.0 / min_omega) * 10.0, 1e7)
    decades = math.log10(max_tau / min_tau)
    num_tau = max(int(decades * tau_points_per_decade), 20)
    tau = np.logspace(math.log10(min_tau), math.log10(max_tau), num=num_tau)

    a_w, b_w, d1 = _build_drt_system(omega, z_real, z_imag, tau)
    if lambda_reg is None:
        lambda_reg = _choose_lambda_lcurve(a_w, b_w, d1, mode=mode)

    gamma, rs = _solve_drt_system(a_w, b_w, d1, float(lambda_reg))
    return tau, gamma, rs


def _build_drt_system(
    omega: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(omega)
    m = len(tau)

    a_real = np.zeros((n, m))
    a_imag = np.zeros((n, m))
    for i, w in enumerate(omega):
        wt = w * tau
        denom = 1.0 + wt**2
        a_real[i, :] = 1.0 / denom
        a_imag[i, :] = -wt / denom

    a = np.vstack([np.column_stack([a_real, np.ones(n)]), np.column_stack([a_imag, np.zeros(n)])])
    b = np.concatenate([z_real, z_imag])

    modulus_sq = z_real**2 + z_imag**2
    w_vec = 1.0 / np.maximum(modulus_sq, 1e-8)
    sqrt_w = np.sqrt(np.concatenate([w_vec, w_vec]))[:, None]

    a_w = a * sqrt_w
    b_w = b * sqrt_w[:, 0]

    d1 = np.diag(-np.ones(m)) + np.diag(np.ones(m - 1), 1)
    d1 = d1[:-1, :]
    return a_w, b_w, d1


def _solve_drt_system(a_w: np.ndarray, b_w: np.ndarray, d1: np.ndarray, lambda_reg: float) -> tuple[np.ndarray, float]:
    m = d1.shape[1]
    l_aug = np.zeros((m - 1, m + 1))
    l_aug[:, :m] = math.sqrt(lambda_reg) * d1
    a_aug = np.vstack([a_w, l_aug])
    b_aug = np.concatenate([b_w, np.zeros(m - 1)])
    x, _ = optimize.nnls(a_aug, b_aug)
    gamma = x[:m]
    rs = float(x[m])
    return gamma, rs


def _choose_lambda_lcurve(a_w: np.ndarray, b_w: np.ndarray, d1: np.ndarray, mode: str = "fast") -> float:
    if mode == "high_precision":
        grid = np.logspace(-6, 0, 40)
    else:
        grid = np.logspace(-6, 0, 20)

    residual_norms: list[float] = []
    smooth_norms: list[float] = []
    for lam in grid:
        gamma, rs = _solve_drt_system(a_w, b_w, d1, float(lam))
        x = np.concatenate([gamma, [rs]])
        residual_norms.append(float(np.linalg.norm(a_w @ x - b_w)))
        smooth_norms.append(float(np.linalg.norm(d1 @ gamma)))

    log_r = np.log10(np.maximum(np.asarray(residual_norms), 1e-30))
    log_s = np.log10(np.maximum(np.asarray(smooth_norms), 1e-30))
    curvature = np.zeros_like(grid)
    for i in range(1, len(grid) - 1):
        x1, y1 = log_r[i - 1], log_s[i - 1]
        x2, y2 = log_r[i], log_s[i]
        x3, y3 = log_r[i + 1], log_s[i + 1]
        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)
        denom = max(a * b * c, 1e-30)
        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        curvature[i] = 2.0 * area2 / denom

    best_idx = int(np.argmax(curvature))
    return float(grid[best_idx])


def find_drt_peaks(tau: np.ndarray, gamma: np.ndarray, min_prominence_pct: float = 0.05) -> list[DrtPeakInfo]:
    if np.max(gamma) <= 0:
        return []

    peak_indices, _ = signal.find_peaks(gamma, prominence=np.max(gamma) * min_prominence_pct)
    if peak_indices.size == 0:
        return []

    results: list[DrtPeakInfo] = []
    valley_indices, _ = signal.find_peaks(-gamma)
    valley_indices = np.concatenate([[0], valley_indices, [len(gamma) - 1]])
    valley_indices.sort()

    log_tau = np.log10(tau)
    for p_idx in peak_indices:
        left_valleys = valley_indices[valley_indices < p_idx]
        right_valleys = valley_indices[valley_indices > p_idx]
        left_idx = int(left_valleys[-1]) if len(left_valleys) > 0 else 0
        right_idx = int(right_valleys[0]) if len(right_valleys) > 0 else len(gamma) - 1

        r_peak = float(np.trapz(gamma[left_idx : right_idx + 1], x=np.log(tau[left_idx : right_idx + 1])))

        try:
            width_results = signal.peak_widths(gamma, [p_idx], rel_height=0.5)
            width_bins = float(width_results[0][0])
            fwhm_decades = width_bins * (log_tau[-1] - log_tau[0]) / max(len(log_tau) - 1, 1)
            n_est = _solve_cpe_n_from_fwhm(max(fwhm_decades, 1e-3))
        except Exception:
            n_est = 0.8

        results.append({"tau": float(tau[p_idx]), "R": max(r_peak, 0.0), "n": float(np.clip(n_est, 0.4, 1.0))})

    results.sort(key=lambda x: x["tau"])
    return results


def _cole_cole_fwhm_decades(n_value: float) -> float:
    angle = np.pi * n_value / 2.0
    numerator = 1.0 + np.sin(angle)
    denominator = max(np.cos(angle), 1e-12)
    return 2.0 * np.log(numerator / denominator) / np.log(10.0)


def _solve_cpe_n_from_fwhm(fwhm_decades: float) -> float:
    fwhm_decades = float(np.clip(fwhm_decades, _cole_cole_fwhm_decades(0.4), _cole_cole_fwhm_decades(1.0)))
    try:
        return float(optimize.brentq(lambda n: _cole_cole_fwhm_decades(float(n)) - fwhm_decades, 0.4, 1.0))
    except Exception:
        candidates = np.linspace(0.4, 1.0, 200)
        widths = np.asarray([_cole_cole_fwhm_decades(float(n)) for n in candidates])
        idx = int(np.argmin(np.abs(widths - fwhm_decades)))
        return float(candidates[idx])
