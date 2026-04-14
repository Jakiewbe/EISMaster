from __future__ import annotations
from typing import Optional








import concurrent.futures
import math
import multiprocessing
import sys
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from eismaster.analysis.fitting import fit_spectrum
from eismaster.analysis.quality import QualityReport, assess_spectrum_quality
from eismaster.analysis.segmentation import SegmentDetection, detect_segments
from eismaster.models import BatchItemResult, BatchSummary, FitOutcome, SpectrumData

SINGLE_MODEL_KEY = "zview_segmented_rq_rwo"
DOUBLE_MODEL_KEY = "zview_double_rq_qrwo"
DOUBLE_SWITCH_COUNT = 2
SINGLE_SWITCH_COUNT = 2


def _worker_fixed(spectrum: SpectrumData, mode: str, model_key: str, batch_fast: bool = True) -> BatchItemResult:
    quality = assess_spectrum_quality(spectrum, run_kk=False)
    hint = detect_segments(spectrum, mode=mode)
    fit = _fit_single_safe(spectrum, model_key, hint, batch_fast=batch_fast)
    fit = _prepend_reason(fit, f"Fixed mode selected: {_mode_label(mode)}")
    return BatchItemResult(spectrum=spectrum, quality=quality, fit=fit)


def _worker_auto(spectrum: SpectrumData, batch_fast: bool = True) -> tuple[FitOutcome, FitOutcome, SegmentDetection, QualityReport]:
    quality = assess_spectrum_quality(spectrum, run_kk=False)
    auto_hint = detect_segments(spectrum, mode="auto")
    single_hint = detect_segments(spectrum, mode="single")
    single_fit = _fit_single_safe(spectrum, SINGLE_MODEL_KEY, single_hint, batch_fast=batch_fast)
    if _should_run_double_fit(single_fit, auto_hint):
        double_hint = detect_segments(spectrum, mode="double")
        double_fit = _fit_single_safe(spectrum, DOUBLE_MODEL_KEY, double_hint, warm_start=None, batch_fast=batch_fast)
    else:
        double_fit = FitOutcome(
            model_key=DOUBLE_MODEL_KEY,
            model_label="Double-arc R(QR)(Q(RWo))",
            status="unavailable",
            message="Skipped double-arc evaluation; single-arc result is already stable.",
        )
    return single_fit, double_fit, auto_hint, quality


def analyze_batch(
    spectra: list[SpectrumData],
    model_key: str,
    *,
    batch_fast: bool = True,
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
) -> BatchSummary:
    if not spectra:
        return BatchSummary(model_key=model_key, items=[])

    mode = "double" if model_key == DOUBLE_MODEL_KEY else "single"
    if _should_use_parallel_processes() and len(spectra) >= 2:
        items = _analyze_batch_parallel_fixed(spectra, mode, model_key, progress_callback, batch_fast=batch_fast)
    else:
        items = _analyze_batch_fixed_sequential(spectra, mode, model_key, progress_callback, batch_fast=batch_fast)
    return BatchSummary(model_key=model_key, items=items)


def analyze_batch_auto(
    spectra: list[SpectrumData],
    *,
    batch_fast: bool = True,
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
) -> BatchSummary:
    if not spectra:
        return BatchSummary(model_key="auto", items=[])

    if _should_use_parallel_processes() and len(spectra) >= 2:
        items = _analyze_batch_parallel_auto(spectra, progress_callback, batch_fast=batch_fast)
    else:
        items = _analyze_batch_auto_sequential(spectra, progress_callback, batch_fast=batch_fast)
    return BatchSummary(model_key="auto", items=items)


def _analyze_batch_parallel_fixed(
    spectra: list[SpectrumData],
    mode: str,
    model_key: str,
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
    *,
    batch_fast: bool = True,
) -> list[BatchItemResult]:
    items: list[Optional[BatchItemResult]] = [None] * len(spectra)
    total = len(spectra)
    results: dict[int, BatchItemResult] = {}
    current_idx = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), max(len(spectra), 1))
    ) as executor:
        future_to_idx = {
            executor.submit(_worker_fixed, spec, mode, model_key, batch_fast): i
            for i, spec in enumerate(spectra)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failed_fit = FitOutcome(model_key=model_key, model_label="?", status="failed", message=f"Worker execution failed: {exc}")
                results[idx] = BatchItemResult(
                    spectrum=spectra[idx],
                    quality=assess_spectrum_quality(spectra[idx], run_kk=False),
                    fit=failed_fit,
                )

            while current_idx in results:
                item = results.pop(current_idx)
                items[current_idx] = item
                current_idx += 1
                if progress_callback is not None:
                    progress_callback(current_idx, total, item)

    return [item for item in items if item is not None]


def _analyze_batch_parallel_auto(
    spectra: list[SpectrumData],
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
    *,
    batch_fast: bool = True,
) -> list[BatchItemResult]:
    items: list[BatchItemResult] = []
    active_mode = "single"
    pending_mode: Optional[str] = None
    pending_count = 0
    total = len(spectra)
    results: dict[int, tuple[FitOutcome, FitOutcome, SegmentDetection, QualityReport]] = {}
    current_idx = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(multiprocessing.cpu_count(), max(len(spectra), 1))
    ) as executor:
        future_to_idx = {executor.submit(_worker_auto, spec, batch_fast): i for i, spec in enumerate(spectra)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failed_fit = FitOutcome(model_key="auto", model_label="?", status="failed", message=f"Worker execution failed: {exc}")
                results[idx] = (
                    failed_fit,
                    failed_fit,
                    SegmentDetection("auto", "single", tuple(), tuple()),
                    assess_spectrum_quality(spectra[idx], run_kk=False),
                )

            while current_idx in results:
                single_fit, double_fit, auto_hint, quality = results.pop(current_idx)
                item, active_mode, pending_mode, pending_count = _choose_auto_item(
                    spectra[current_idx],
                    quality,
                    single_fit,
                    double_fit,
                    auto_hint,
                    active_mode,
                    pending_mode,
                    pending_count,
                )
                items.append(item)
                current_idx += 1
                if progress_callback is not None:
                    progress_callback(current_idx, total, item)

    return items


def _analyze_batch_fixed_sequential(
    spectra: list[SpectrumData],
    mode: str,
    model_key: str,
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
    *,
    batch_fast: bool = True,
) -> list[BatchItemResult]:
    items: list[BatchItemResult] = []
    total = len(spectra)
    previous_fit: FitOutcome | None = None
    for index, spectrum in enumerate(spectra, start=1):
        quality = assess_spectrum_quality(spectrum, run_kk=False)
        hint = detect_segments(spectrum, mode=mode)
        fit = _fit_single_safe(spectrum, model_key, hint, warm_start=previous_fit, batch_fast=batch_fast)
        fit = _prepend_reason(fit, f"Fixed mode selected: {_mode_label(mode)}")
        item = BatchItemResult(spectrum=spectrum, quality=quality, fit=fit)
        items.append(item)
        previous_fit = fit
        if progress_callback is not None:
            progress_callback(index, total, item)
    return items


def _analyze_batch_auto_sequential(
    spectra: list[SpectrumData],
    progress_callback: Callable[[int, int, BatchItemResult], None] | None = None,
    *,
    batch_fast: bool = True,
) -> list[BatchItemResult]:
    items: list[BatchItemResult] = []
    active_mode = "single"
    pending_mode: Optional[str] = None
    pending_count = 0
    total = len(spectra)
    previous_single_fit: FitOutcome | None = None
    previous_double_fit: FitOutcome | None = None

    for index, spectrum in enumerate(spectra, start=1):
        quality = assess_spectrum_quality(spectrum, run_kk=False)
        auto_hint = detect_segments(spectrum, mode="auto")
        single_hint = detect_segments(spectrum, mode="single")
        single_fit = _fit_single_safe(spectrum, SINGLE_MODEL_KEY, single_hint, warm_start=previous_single_fit, batch_fast=batch_fast)
        if _should_run_double_fit(single_fit, auto_hint):
            double_hint = detect_segments(spectrum, mode="double")
            double_fit = _fit_single_safe(spectrum, DOUBLE_MODEL_KEY, double_hint, warm_start=previous_double_fit, batch_fast=batch_fast)
            previous_double_fit = double_fit
        else:
            double_fit = FitOutcome(
                model_key=DOUBLE_MODEL_KEY,
                model_label="Double-arc R(QR)(Q(RWo))",
                status="unavailable",
                message="Skipped double-arc evaluation; single-arc result is already stable.",
            )
        previous_single_fit = single_fit
        item, active_mode, pending_mode, pending_count = _choose_auto_item(
            spectrum,
            quality,
            single_fit,
            double_fit,
            auto_hint,
            active_mode,
            pending_mode,
            pending_count,
        )
        items.append(item)
        if progress_callback is not None:
            progress_callback(index, total, item)

    return items


def _choose_auto_item(
    spectrum: SpectrumData,
    quality: QualityReport,
    single_fit: FitOutcome,
    double_fit: FitOutcome,
    auto_hint: SegmentDetection,
    active_mode: str,
    pending_mode: Optional[str],
    pending_count: int,
) -> tuple[BatchItemResult, str, Optional[str], int]:
    preferred_mode, detail = _preferred_mode(auto_hint, single_fit, double_fit)
    active_mode, pending_mode, pending_count, headline = _apply_hysteresis(active_mode, preferred_mode, pending_mode, pending_count)
    chosen_fit = single_fit if active_mode == "single" else double_fit
    chosen_fit = _prepend_reason(chosen_fit, f"{headline}; {detail}")
    return BatchItemResult(spectrum=spectrum, quality=quality, fit=chosen_fit), active_mode, pending_mode, pending_count


def _fit_single_safe(
    spectrum: SpectrumData,
    model_key: str,
    hint: SegmentDetection,
    warm_start: FitOutcome | None = None,
    batch_fast: bool = True,
) -> FitOutcome:
    try:
        fit = fit_spectrum(
            spectrum,
            model_key,
            segment_hint=hint,
            warm_start=warm_start,
            allow_fallback=True,
            auto_preprocess=False,
            use_drt_guided_guess=False,
            batch_fast=batch_fast,
        )
        if _needs_expensive_retry(fit):
            retry = fit_spectrum(
                spectrum,
                model_key,
                segment_hint=hint,
                warm_start=fit if fit.status != "failed" else warm_start,
                allow_fallback=True,
                auto_preprocess=False,
                use_drt_guided_guess=True,
                batch_fast=batch_fast,
            )
            if _fit_is_usable(retry):
                return retry
        return fit
    except Exception as exc:
        return FitOutcome(model_key=model_key, model_label="?", status="failed", message=f"Fit execution failed: {exc}")


def _preferred_mode(
    auto_hint: SegmentDetection,
    single_fit: FitOutcome,
    double_fit: FitOutcome,
) -> tuple[str, str]:
    single_high_error = _has_primary_high_error(single_fit)
    double_high_error = _has_primary_high_error(double_fit)

    if _fit_is_usable(single_fit) and not single_high_error and (not _fit_is_usable(double_fit) or double_high_error):
        if auto_hint.resolved_mode == "double":
            return "single", "Double-arc detection was suggested, but the single-arc fit is more stable."
        return "single", "Single-arc fit is stable and preferred."

    single_score = _candidate_score(single_fit, "single", auto_hint)
    double_score = _candidate_score(double_fit, "double", auto_hint)

    if not np.isfinite(double_score):
        return "single", "Double-arc candidate is unavailable."
    if not np.isfinite(single_score):
        return "double", "Single-arc candidate is unavailable."

    margin = 7.5 if auto_hint.resolved_mode == "single" else 3.5
    if double_score + margin < single_score and _double_is_reliable(double_fit):
        return "double", "Double-arc candidate has the better score and remains reliable."
    if auto_hint.resolved_mode == "double" and double_score + 1.5 < single_score:
        return "double", "Automatic segmentation still favors the double-arc model."
    return "single", "Single-arc candidate remains the conservative default."


def _candidate_score(fit: FitOutcome, mode: str, auto_hint: SegmentDetection) -> float:
    if fit.status in {"failed", "unavailable"}:
        return float("inf")

    score = _fit_scientific_score(fit)

    if mode == "single":
        if auto_hint.resolved_mode == "double":
            score += 4.0
        rct = fit.parameters.get("Rct")
        if rct is None or not np.isfinite(float(rct)) or float(rct) <= 0:
            score += 80.0
    else:
        if auto_hint.resolved_mode == "single":
            score += 8.0
        rsei = fit.parameters.get("Rsei")
        rct = fit.parameters.get("Rct")
        if rsei is None or rct is None:
            score += 100.0
        else:
            rsei = float(rsei)
            rct = float(rct)
            if not np.isfinite(rsei) or not np.isfinite(rct) or rsei <= 0 or rct <= 0:
                score += 100.0
            elif rsei < max(1e-3, rct * 0.05):
                score += 12.0
        if auto_hint.resolved_mode != "double" and len(auto_hint.peak_indices) < 2:
            score += 10.0

    return score


def _fit_scientific_score(fit: FitOutcome) -> float:
    stats = fit.statistics
    score = _finite_or(stats.get("aicc"), _finite_or(stats.get("bic"), 1e9))

    if fit.status == "warn":
        score += 10.0

    chi2 = _finite_or(stats.get("chi2_reduced"), _finite_or(stats.get("pseudo_chi2"), 10.0))
    score += min(chi2 * 4.0, 40.0)

    cond = stats.get("jtj_cond")
    if cond is not None:
        cond = float(cond)
        if not np.isfinite(cond):
            score += 60.0
        elif cond > 1e12:
            score += min(20.0, math.log10(cond) - 12.0 + 8.0)
        elif cond > 1e8:
            score += min(8.0, math.log10(cond) - 8.0 + 2.0)

    corr = stats.get("correlation_matrix_max")
    if corr is not None:
        corr = float(corr)
        if not np.isfinite(corr):
            score += 20.0
        elif corr > 0.995:
            score += 16.0
        elif corr > 0.98:
            score += 8.0
        elif corr > 0.95:
            score += 3.0

    for error_key in ("Rs_stderr_pct", "Rsei_stderr_pct", "Rct_stderr_pct"):
        err = stats.get(error_key)
        if err is None:
            continue
        err = float(err)
        if not np.isfinite(err):
            score += 15.0
        elif err > 50.0:
            score += 16.0
        elif err > 20.0:
            score += 12.0
        elif err > 10.0:
            score += 4.0

    diagnosis_type = (fit.diagnosis_type or "").lower()
    diagnosis_severity = (fit.diagnosis_severity or "").lower()
    diagnosis_explanation = (fit.diagnosis_explanation or "").lower()
    message = (fit.message or "").lower()

    if diagnosis_severity == "error":
        score += 12.0
    elif diagnosis_severity == "warning":
        score += 4.0

    if diagnosis_type == "boundary":
        score += 8.0
    elif diagnosis_type == "ill_conditioned":
        score += 10.0
    elif diagnosis_type == "model_mismatch":
        score += 12.0
    elif diagnosis_type == "segment_error":
        score += 18.0
    elif diagnosis_type == "data_quality":
        score += 14.0
    elif diagnosis_type == "convergence":
        score += 6.0

    if "parameters near bounds" in message:
        score += 8.0
    if "strong_parameter_correlation" in message:
        score += 6.0
    if "direct global fallback" in message:
        score += 8.0
    if "model misses systematic error" in diagnosis_explanation:
        score += 4.0

    return score


def _double_is_reliable(fit: FitOutcome) -> bool:
    if fit.status in {"failed", "unavailable"}:
        return False
    if _has_primary_high_error(fit):
        return False
    rsei = fit.parameters.get("Rsei")
    rct = fit.parameters.get("Rct")
    if rsei is None or rct is None:
        return False
    rsei = float(rsei)
    rct = float(rct)
    if not np.isfinite(rsei) or not np.isfinite(rct) or rsei <= 0 or rct <= 0:
        return False
    if rsei < max(1e-3, rct * 0.05):
        return False
    for key in ("Rsei_stderr_pct", "Rct_stderr_pct"):
        err = fit.statistics.get(key)
        if err is None:
            continue
        err = float(err)
        if not np.isfinite(err) or err > 35.0:
            return False
    return True


def _fit_is_usable(fit: FitOutcome) -> bool:
    return fit.status not in {"failed", "unavailable"}


def _needs_expensive_retry(fit: FitOutcome) -> bool:
    if fit.status in {"failed", "unavailable"}:
        return True
    if _has_primary_high_error(fit):
        return True
    if fit.diagnosis_type in {"convergence", "ill_conditioned", "segment_error", "model_mismatch"}:
        return True
    if fit.fallback_from:
        return True
    corr = fit.statistics.get("correlation_matrix_max")
    if corr is not None and np.isfinite(float(corr)) and float(corr) > 0.995:
        return True
    return False


def _should_run_double_fit(single_fit: FitOutcome, auto_hint: SegmentDetection) -> bool:
    if auto_hint.resolved_mode == "double":
        return True
    if not _fit_is_usable(single_fit):
        return True
    if _has_primary_high_error(single_fit):
        return True
    if single_fit.status == "warn" and single_fit.diagnosis_type in {"segment_error", "model_mismatch", "convergence", "ill_conditioned"}:
        return True
    if single_fit.fallback_from:
        return True
    return False


def _has_primary_high_error(fit: FitOutcome, threshold: float = 20.0) -> bool:
    keys = ("Rs_stderr_pct", "Rsei_stderr_pct", "Rct_stderr_pct") if fit.model_key == DOUBLE_MODEL_KEY else ("Rs_stderr_pct", "Rct_stderr_pct")
    found = False
    for key in keys:
        value = fit.statistics.get(key)
        if value is None:
            continue
        found = True
        value = float(value)
        if not np.isfinite(value) or value > threshold:
            return True
    return False if found else fit.status == "failed"


def _apply_hysteresis(
    active_mode: str,
    preferred_mode: str,
    pending_mode: Optional[str],
    pending_count: int,
) -> tuple[str, Optional[str], int, str]:
    if preferred_mode == active_mode:
        return active_mode, None, 0, f"Auto mode keeps {_mode_label(active_mode)}"

    if pending_mode == preferred_mode:
        pending_count += 1
    else:
        pending_mode = preferred_mode
        pending_count = 1

    required = DOUBLE_SWITCH_COUNT if preferred_mode == "double" else SINGLE_SWITCH_COUNT
    if pending_count >= required:
        active_mode = preferred_mode
        return active_mode, None, 0, f"Auto mode switched to {_mode_label(active_mode)}"

    return active_mode, pending_mode, pending_count, f"Auto mode is evaluating {_mode_label(preferred_mode)} but keeps {_mode_label(active_mode)} for now"


def _prepend_reason(fit: FitOutcome, headline: str) -> FitOutcome:
    message = headline if not fit.message else f"{headline}; {fit.message}"
    return replace(fit, message=message)


def _should_use_parallel_processes() -> bool:
    return not getattr(sys, "frozen", False)


def _mode_label(mode: str) -> str:
    return "double-arc" if mode == "double" else "single-arc"


def _finite_or(value: Optional[float], fallback: float) -> float:
    if value is None:
        return fallback
    value = float(value)
    return value if np.isfinite(value) else fallback
