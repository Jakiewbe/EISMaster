from typing import Dict
import typing






import math


import numpy as np
from scipy import optimize, signal

from eismaster.models import SpectrumData


class DrtPeakInfo(TypedDict):
    tau: float
    R: float
    n: float


def compute_drt(spectrum: SpectrumData, lambda_reg: float = 1e-3, tau_points_per_decade: int = 15) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the Distribution of Relaxation Times (DRT) natively via Ridge Regression (NNLS).
    Useful to cleanly separate time constants without generating an explicit Equivalent Circuit.
    """
    freq_hz = np.asarray(spectrum.freq_hz, dtype=float)
    z_real = np.asarray(spectrum.z_real_ohm, dtype=float)
    z_imag = np.asarray(spectrum.z_imag_ohm, dtype=float)  # exact raw imaginary

    omega = 2.0 * np.pi * freq_hz
    
    max_omega = np.max(omega)
    min_omega = np.min(omega)
    min_tau = max((1.0 / max_omega) * 0.1, 1e-7)
    max_tau = min((1.0 / min_omega) * 10.0, 1e7)
    decades = math.log10(max_tau / min_tau)
    num_tau = max(int(decades * tau_points_per_decade), 20)
    tau = np.logspace(math.log10(min_tau), math.log10(max_tau), num=num_tau)

    N = len(omega)
    M = len(tau)

    A_real = np.zeros((N, M))
    A_imag = np.zeros((N, M))

    for i, w in enumerate(omega):
        wt = w * tau
        denom = 1.0 + wt**2
        A_real[i, :] = 1.0 / denom
        A_imag[i, :] = -wt / denom

    A = np.vstack([
        np.column_stack([A_real, np.ones(N)]),
        np.column_stack([A_imag, np.zeros(N)])
    ])
    
    b = np.concatenate([z_real, z_imag])
    
    modulus_sq = z_real**2 + z_imag**2
    w_vec = 1.0 / np.maximum(modulus_sq, 1e-8)
    W_diag = np.concatenate([w_vec, w_vec])
    sqrt_W = np.sqrt(W_diag)[:, None]
    
    A_w = A * sqrt_W
    b_w = b * sqrt_W[:, 0]
    
    D1 = np.diag(-np.ones(M)) + np.diag(np.ones(M-1), 1)
    D1 = D1[:-1, :] 
    
    L_aug = np.zeros((M-1, M+1))
    L_aug[:, :M] = math.sqrt(lambda_reg) * D1
    
    A_aug = np.vstack([A_w, L_aug])
    b_aug = np.concatenate([b_w, np.zeros(M-1)])
    
    x, _ = optimize.nnls(A_aug, b_aug)
    
    gamma = x[:M]
    Rs = x[M]
    
    return tau, gamma, Rs


def find_drt_peaks(tau: np.ndarray, gamma: np.ndarray, min_prominence_pct: float = 0.05) -> list[DrtPeakInfo]:
    """
    Find significant peaks in the DRT spectrum and estimate their associated R and n values.
    """
    if np.max(gamma) <= 0:
        return []
        
    peak_indices, _ = signal.find_peaks(gamma, prominence=np.max(gamma)*min_prominence_pct)
    
    results: list[DrtPeakInfo] = []
    log_tau = np.log10(tau)
    
    valley_indices, _ = signal.find_peaks(-gamma)
    valley_indices = np.concatenate([[0], valley_indices, [len(gamma)-1]])
    valley_indices.sort()
    
    d_log_tau = (log_tau[-1] - log_tau[0]) / (len(log_tau) - 1)
    
    for p_idx in peak_indices:
        left_valleys = valley_indices[valley_indices < p_idx]
        right_valleys = valley_indices[valley_indices > p_idx]
        
        left_idx = int(left_valleys[-1]) if len(left_valleys) > 0 else 0
        right_idx = int(right_valleys[0]) if len(right_valleys) > 0 else len(gamma)-1
        
        R_peak = float(np.sum(gamma[left_idx:right_idx+1]) * (math.log(10) * d_log_tau))
        
        try:
            width_results = signal.peak_widths(gamma, [p_idx], rel_height=0.5)
            width_bins = float(width_results[0][0])
            fwhm_decades = width_bins * d_log_tau
            n_est = min(1.0, 1.14 / max(fwhm_decades, 0.1))
        except Exception:
            n_est = 0.8
            
        n_est = max(0.4, n_est)
        
        results.append({
            "tau": float(tau[p_idx]),
            "R": float(R_peak),
            "n": float(n_est)
        })
        
    # Sort peaks by tau (highest frequency / lowest tau first)
    results.sort(key=lambda x: x["tau"])
    return results