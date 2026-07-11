from __future__ import annotations

import numpy as np


def adaptive_weight_floor(z_exp: np.ndarray) -> float:
    return max(float(np.median(np.abs(z_exp))) * 1e-4, 1e-6)


def estimate_cpe_n(freq: np.ndarray, z_imag_neg: np.ndarray) -> float:
    if freq.size < 4 or z_imag_neg.size < 4:
        return 0.85
    keep = np.isfinite(freq) & np.isfinite(z_imag_neg) & (freq > 0) & (z_imag_neg > 0)
    if int(keep.sum()) < 4:
        return 0.85
    log_f = np.log10(freq[keep].astype(float))
    log_z = np.log10(z_imag_neg[keep].astype(float))
    peak = int(np.argmax(z_imag_neg[keep]))
    half_window = max(min(log_f.size // 5, 6), 2)
    start = max(0, peak - half_window)
    stop = min(log_f.size, peak + half_window + 1)
    if stop - start < 4:
        start = max(0, min(peak, log_f.size - 4))
        stop = min(log_f.size, start + 4)
    try:
        slope, _ = np.polyfit(log_f[start:stop], log_z[start:stop], 1)
    except Exception:
        return 0.85
    return float(np.clip(abs(slope), 0.4, 1.0))


def local_noise_estimate(z_exp: np.ndarray) -> np.ndarray:
    if z_exp.size < 5:
        return np.full(z_exp.size, adaptive_weight_floor(z_exp), dtype=float)
    real = np.pad(z_exp.real.astype(float), (2, 2), mode="edge")
    imag = np.pad(z_exp.imag.astype(float), (2, 2), mode="edge")
    real_win = np.lib.stride_tricks.sliding_window_view(real, 5)
    imag_win = np.lib.stride_tricks.sliding_window_view(imag, 5)
    real_med = np.median(real_win, axis=1)
    imag_med = np.median(imag_win, axis=1)
    local = np.hypot(real_win - real_med[:, None], imag_win - imag_med[:, None])
    local_med = np.median(local, axis=1)
    sigma = 1.4826 * np.median(np.abs(local - local_med[:, None]), axis=1)
    return np.maximum(sigma, adaptive_weight_floor(z_exp))


def zview_warburg_open(omega: np.ndarray, wo_r: float, wo_t: float, wo_p: float) -> np.ndarray:
    x = (1j * omega * wo_t) ** wo_p
    abs_x = np.abs(x)
    tanh_x = np.ones_like(x, dtype=complex)
    finite_mask = abs_x <= 50.0
    tanh_x[finite_mask] = np.tanh(x[finite_mask])
    small = abs_x < 1e-8
    denom = x * tanh_x
    if np.any(small):
        denom = np.where(small, x * x * (1.0 - (x * x) / 3.0), denom)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0.0j, denom)
    return wo_r / denom
