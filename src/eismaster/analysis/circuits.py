from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CircuitTemplate:
    key: str
    label: str
    parameter_names: tuple[str, ...]
    model: Callable[[np.ndarray, np.ndarray], np.ndarray]
    primary_exports: tuple[str, ...]


def _parallel(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    return 1.0 / ((1.0 / z1) + (1.0 / z2))


def _cpe(q: float, n: float, omega: np.ndarray) -> np.ndarray:
    return 1.0 / (q * (1j * omega) ** n)


def _warburg(sigma: float, omega: np.ndarray) -> np.ndarray:
    return sigma / np.sqrt(1j * omega)


def model_l_rs_rq_w(params: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    l_h, rs, rct, q_dl, n_dl, sigma = params
    omega = 2.0 * np.pi * freq_hz
    return 1j * omega * l_h + rs + _parallel(np.full(freq_hz.shape, rct, dtype=complex), _cpe(q_dl, n_dl, omega)) + _warburg(sigma, omega)


def model_l_rs_rq_rq_w(params: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    l_h, rs, r_sei, q_sei, n_sei, r_ct, q_dl, n_dl, sigma = params
    omega = 2.0 * np.pi * freq_hz
    z_sei = _parallel(np.full(freq_hz.shape, r_sei, dtype=complex), _cpe(q_sei, n_sei, omega))
    z_ct = _parallel(np.full(freq_hz.shape, r_ct, dtype=complex), _cpe(q_dl, n_dl, omega))
    return 1j * omega * l_h + rs + z_sei + z_ct + _warburg(sigma, omega)


def model_l_rs_rq_rq(params: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    l_h, rs, r1, q1, n1, r2, q2, n2 = params
    omega = 2.0 * np.pi * freq_hz
    z_1 = _parallel(np.full(freq_hz.shape, r1, dtype=complex), _cpe(q1, n1, omega))
    z_2 = _parallel(np.full(freq_hz.shape, r2, dtype=complex), _cpe(q2, n2, omega))
    return 1j * omega * l_h + rs + z_1 + z_2


def model_rs_q_rw(params: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    rs, q_dl, n_dl, rct, sigma = params
    omega = 2.0 * np.pi * freq_hz
    z_branch = np.full(freq_hz.shape, rct, dtype=complex) + _warburg(sigma, omega)
    return rs + _parallel(_cpe(q_dl, n_dl, omega), z_branch)


def model_placeholder(params: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    return np.zeros_like(freq_hz, dtype=complex)


TEMPLATES: dict[str, CircuitTemplate] = {
    "zview_segmented_rq_rwo": CircuitTemplate(
        key="zview_segmented_rq_rwo",
        label="Single-arc R(QRWo)",
        parameter_names=("Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P", "split_freq_hz"),
        model=model_placeholder,
        primary_exports=("Rs", "Rct"),
    ),
    "zview_double_rq_qrwo": CircuitTemplate(
        key="zview_double_rq_qrwo",
        label="Double-arc R(QR)(Q(RWo))",
        parameter_names=("Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P", "split1_freq_hz", "split2_freq_hz"),
        model=model_placeholder,
        primary_exports=("Rs", "Rsei", "Rct"),
    ),
}


def get_circuit_templates() -> dict[str, CircuitTemplate]:
    return TEMPLATES.copy()
