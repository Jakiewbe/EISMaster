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
