from __future__ import annotations

import numpy as np

from eismaster.analysis.fitting_math import (
    adaptive_weight_floor,
    estimate_cpe_n,
    local_noise_estimate,
    zview_warburg_open,
)


def test_adaptive_weight_floor_scales_with_impedance() -> None:
    assert adaptive_weight_floor(np.array([1.0 + 0j, 3.0 + 0j])) == 0.0002


def test_estimate_cpe_n_stays_in_physical_range() -> None:
    freq = np.geomspace(1e5, 1e-2, 20)
    z_imag_neg = 10.0 / (1.0 + (freq / 100.0) ** 0.8)
    assert 0.4 <= estimate_cpe_n(freq, z_imag_neg) <= 1.0


def test_local_noise_estimate_is_finite_and_positive() -> None:
    values = np.array([1.0 + 0.1j, 1.1 + 0.2j, 0.9 + 0.0j, 1.0 + 0.1j, 1.05 + 0.2j])
    noise = local_noise_estimate(values)
    assert noise.shape == values.shape
    assert np.all(np.isfinite(noise))
    assert np.all(noise > 0.0)


def test_zview_warburg_open_is_finite_near_zero() -> None:
    omega = np.array([0.0, 1e-12, 1.0])
    result = zview_warburg_open(omega, 1.0, 2.0, 0.5)
    assert np.all(np.isfinite(result))
