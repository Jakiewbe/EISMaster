from __future__ import annotations
from typing import Optional







"""Structured failure diagnosis for EIS impedance fitting."""


from dataclasses import dataclass

import numpy as np
from eismaster.analysis.segmentation import SegmentDetection
from eismaster.models import FitOutcome, SpectrumData


@dataclass(frozen=True)
class FitDiagnosis:
    failure_type: str
    severity: str
    explanation: str
    suggestions: list[str]


def diagnose_fit_failure(
    spectrum: SpectrumData,
    fit: FitOutcome,
    template_key: str,
    segment_hint: Optional[SegmentDetection] = None,
) -> FitDiagnosis:
    """Analyse why a fit failed or raised warnings."""

    del template_key

    if fit.status not in ("failed", "warn"):
        return FitDiagnosis("none", "info", "Fit completed without warnings.", [])

    if "Too few" in fit.message or spectrum.n_points < 8:
        return FitDiagnosis(
            "data_quality",
            "error",
            "The spectrum has too few valid points for a stable fit.",
            ["Import the full frequency range or reduce masking.", "Mask problematic points manually before fitting again."],
        )

    if segment_hint is not None and ("split" in fit.message.lower() or "segment" in fit.message.lower()):
        split_text = ", ".join(str(i) for i in segment_hint.split_indices) or "none"
        return FitDiagnosis(
            "segment_error",
            "error",
            "The selected segmentation does not match the Nyquist features well enough.",
            ["Adjust the segment boundaries manually.", f"Current split indices: {split_text}.", "Try the alternate single-arc or double-arc template."],
        )

    stats = fit.statistics
    if "bound" in fit.message.lower():
        return FitDiagnosis(
            "boundary",
            "warning",
            "One or more fitted parameters converged to a boundary value.",
            ["Relax the parameter bounds or improve the initial guess.", "Try a simpler equivalent-circuit template.", "If high-frequency inductive points dominate, mask them manually and retry."],
        )

    cond = stats.get("jtj_cond")
    if cond is not None and (not np.isfinite(cond) or float(cond) > 1e12):
        return FitDiagnosis(
            "ill_conditioned",
            "warning",
            f"The Jacobian is ill-conditioned (cond={float(cond):.2e}).",
            ["Reduce the number of free parameters.", "Inspect low-frequency tail points and mask suspicious points manually if needed."],
        )

    high_err_params: list[str] = []
    for key in ("Rs_stderr_pct", "Rsei_stderr_pct", "Rct_stderr_pct"):
        val = stats.get(key)
        if val is not None and (not np.isfinite(val) or float(val) > 20.0):
            high_err_params.append(key.replace("_stderr_pct", ""))
    if high_err_params:
        return FitDiagnosis(
            "convergence",
            "warning",
            f"Primary parameters have high uncertainty: {', '.join(high_err_params)}.",
            ["Check whether the selected model is too complex for this spectrum.", "Review masked points and frequency coverage.", "Compare the fit against the alternate template."],
        )

    if fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
        residual_real = spectrum.z_real_ohm - fit.predicted_real_ohm
        residual_imag = spectrum.z_imag_ohm - fit.predicted_imag_ohm
        z_mod = np.maximum(np.abs(spectrum.impedance), 1e-12)
        rel_err = np.sqrt(residual_real**2 + residual_imag**2) / z_mod
        n = len(rel_err)
        if n >= 10:
            low_seg = rel_err[n * 3 // 4 :]
            high_seg = rel_err[: n // 4]
            if np.median(low_seg) > 0.15 and np.median(high_seg) < 0.05:
                return FitDiagnosis(
                    "model_mismatch",
                    "warning",
                    "The model misses systematic error in the low-frequency region.",
                    ["Consider a diffusion element or a more suitable low-frequency branch.", "Mask obviously unstable tail points and refit."],
                )
            if np.median(high_seg) > 0.15 and np.median(low_seg) < 0.05:
                return FitDiagnosis(
                    "model_mismatch",
                    "warning",
                    "The model misses systematic error in the high-frequency region.",
                    ["Inspect the inductive or contact-resistance part of the spectrum.", "Mask unstable high-frequency points and refit."],
                )

    if fit.status == "failed":
        return FitDiagnosis(
            "convergence",
            "error",
            "The nonlinear optimizer did not converge to a stable solution.",
            ["Try the alternate template.", "Check segment boundaries and masked points.", "Inspect whether the data quality is sufficient for fitting."],
        )

    return FitDiagnosis(
        "convergence",
        "warning",
        f"Fit completed with warnings: {fit.message.split(';')[0]}",
        ["Review the fit curve and primary parameter uncertainties."],
    )
