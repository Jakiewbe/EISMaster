from __future__ import annotations
from typing import Optional
import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from eismaster.analysis.circuits import CircuitTemplate, TEMPLATES, get_circuit_templates as _get_circuit_templates
from eismaster.analysis.diagnostics import FitDiagnosis, diagnose_fit_failure
from eismaster.analysis.preprocessing import PreprocessResult, preprocess_for_fitting
from eismaster.analysis.segmentation import ArcRange, SegmentDetection, detect_segments
from eismaster.models import FitOutcome, SpectrumData

try:
    from pyimpspec import DataSet, fit_circuit, parse_cdc
except Exception:  # pragma: no cover
    DataSet = None
    fit_circuit = None
    parse_cdc = None

try:
    from scipy.optimize import OptimizeResult, least_squares
except Exception:  # pragma: no cover
    OptimizeResult = None  # type: ignore[misc,assignment]
    least_squares = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuessPack:
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class FitConfig:
    primary_error_warn_pct: float = 20.0
    corr_warn_threshold: float = 0.90
    corr_error_threshold: float = 0.95
    pyimpspec_max_nfev: int = 4000
    zview_direct_max_nfev: int = 40000
    zview_single_fast_max_nfev: int = 12000
    zview_single_full_max_nfev: int = 8000
    zview_single_retry_max_nfev: int = 15000
    zview_double_fast_max_nfev: int = 10000
    zview_double_full_max_nfev: int = 10000
    zview_double_retry_max_nfev: int = 15000
    pyimpspec_good_score: float = -1.2
    pyimpspec_factor_probe_count: int = 3


FIT_CONFIG = FitConfig()


@dataclass(frozen=True)
class DrtGuide:
    rs: float
    peaks: tuple[dict[str, float], ...]


FIT_METHODS = ("least_squares", "lbfgsb", "powell")
FIT_WEIGHTS = ("boukamp", "proportional", "modulus")
SEED_FACTORS = (1.0, 0.55, 1.8, 0.3, 3.0, 0.1, 5.0)
ZVIEW_CNLS_WEIGHTS = ("calc-modulus", "calc-proportional", "calc-unit", "data-special", "calc-modulus-zview")


def _adaptive_weight_floor(z_exp: np.ndarray) -> float:
    return max(float(np.median(np.abs(z_exp))) * 1e-4, 1e-6)


def _estimate_cpe_n(freq: np.ndarray, z_imag_neg: np.ndarray) -> float:
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


def _local_noise_estimate(z_exp: np.ndarray) -> np.ndarray:
    if z_exp.size < 5:
        return np.full(z_exp.size, _adaptive_weight_floor(z_exp), dtype=float)
    real = np.pad(z_exp.real.astype(float), (2, 2), mode="edge")
    imag = np.pad(z_exp.imag.astype(float), (2, 2), mode="edge")
    real_win = np.lib.stride_tricks.sliding_window_view(real, 5)
    imag_win = np.lib.stride_tricks.sliding_window_view(imag, 5)
    real_med = np.median(real_win, axis=1)
    imag_med = np.median(imag_win, axis=1)
    local = np.hypot(real_win - real_med[:, None], imag_win - imag_med[:, None])
    local_med = np.median(local, axis=1)
    sigma = 1.4826 * np.median(np.abs(local - local_med[:, None]), axis=1)
    return np.maximum(sigma, _adaptive_weight_floor(z_exp))


def _from_arc_ranges(
    spectrum: SpectrumData,
    arc_ranges: list[ArcRange],
) -> tuple[SegmentDetection, np.ndarray]:
    """Build a SegmentDetection and point_mask from user-defined arc ranges.

    For single arc (1 range):
      - split_index = end of the range
      - peak_index = argmax(-Z'') within [start, end]
      - mask keeps the full spectrum; points after ``end`` are the tail

    For dual arc (2 ranges):
      - split1 = end of arc1, split2 = end of arc2
      - peak1/peak2 = argmax(-Z'') within each arc range
      - mask keeps the full spectrum; points after ``end2`` are the tail
    """
    n = spectrum.n_points
    minus_z = spectrum.minus_z_imag_ohm

    if len(arc_ranges) == 1:
        arc = arc_ranges[0]
        start = max(0, min(arc.start, n - 1))
        end = max(start + 1, min(arc.end, n - 1))
        peak_slice = slice(start, end + 1)
        peak_in_slice = int(np.argmax(minus_z[peak_slice]))
        peak = start + peak_in_slice
        mask = np.ones(n, dtype=bool)
        return SegmentDetection(
            requested_mode="single",
            resolved_mode="single",
            peak_indices=(peak,),
            split_indices=(end,),
        ), mask

    # dual arc
    a1, a2 = arc_ranges[0], arc_ranges[1]
    start1 = max(0, min(a1.start, n - 1))
    end1 = max(start1 + 1, min(a1.end, n - 2))
    start2 = max(end1 + 1, min(a2.start, n - 1))
    end2 = max(start2 + 1, min(a2.end, n - 1))
    peak1_slice = slice(start1, end1 + 1)
    peak2_slice = slice(start2, end2 + 1)
    peak1 = start1 + int(np.argmax(minus_z[peak1_slice]))
    peak2 = start2 + int(np.argmax(minus_z[peak2_slice]))
    mask = np.ones(n, dtype=bool)
    return SegmentDetection(
        requested_mode="double",
        resolved_mode="double",
        peak_indices=(peak1, peak2),
        split_indices=(end1, end2),
    ), mask


def fit_spectrum(
    spectrum: SpectrumData,
    template_key: str,
    point_mask: Optional[np.ndarray] = None,
    segment_hint: Optional[SegmentDetection] = None,
    arc_ranges: Optional[list[ArcRange]] = None,
    warm_start: Optional[FitOutcome] = None,
    *,
    allow_fallback: bool = False,
    auto_preprocess: bool = False,
    use_drt_guided_guess: bool = True,
    batch_fast: bool = False,
    drt_guide: Optional[DrtGuide] = None,
) -> FitOutcome:
    template = TEMPLATES[template_key]
    freq = spectrum.freq_hz

    # --- Derive segment_hint and point_mask from arc_ranges if provided ------
    if arc_ranges and not segment_hint:
        segment_hint, point_mask = _from_arc_ranges(spectrum, arc_ranges)

    if point_mask is None:
        point_mask = np.ones_like(freq, dtype=bool)

    # --- auto preprocessing ------------------------------------------------
    pp_actions: list[str] = []
    if auto_preprocess:
        pp = preprocess_for_fitting(spectrum, point_mask)
        if pp.actions:
            point_mask = pp.mask
            pp_actions = list(pp.actions)

    fit_freq = freq[point_mask]
    fit_z = spectrum.impedance[point_mask]
    if fit_freq.size < max(8, len(template.parameter_names) + 2):
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Too few unmasked points for the selected circuit.",
            masked_points=int((~point_mask).sum()),
            preprocess_actions=pp_actions,
        )

    if template_key == "zview_segmented_rq_rwo":
        result = _fit_zview_global(
            spectrum,
            point_mask,
            segment_hint,
            warm_start=warm_start,
            use_drt_guided_guess=use_drt_guided_guess,
            batch_fast=batch_fast,
            drt_guide=drt_guide,
        )
        result = replace(result, preprocess_actions=pp_actions)
        return _attach_diagnosis(result, spectrum, template_key, segment_hint)
    if template_key == "zview_double_rq_qrwo":
        result = _fit_zview_double_global(
            spectrum,
            point_mask,
            segment_hint,
            warm_start=warm_start,
            use_drt_guided_guess=use_drt_guided_guess,
            batch_fast=batch_fast,
            drt_guide=drt_guide,
        )
        result = replace(result, preprocess_actions=pp_actions)
        return _attach_diagnosis(result, spectrum, template_key, segment_hint)

    if DataSet is None or fit_circuit is None or parse_cdc is None:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="unavailable",
            message="pyimpspec is not installed; fitting backend unavailable.",
            masked_points=int((~point_mask).sum()),
            preprocess_actions=pp_actions,
        )

    data = DataSet(
        frequencies=fit_freq.astype(float),
        impedances=fit_z.astype(complex),
        path=str(spectrum.metadata.file_path),
        label=spectrum.display_name,
    )
    guesses = _build_initial_guesses(
        template,
        spectrum,
        point_mask,
        warm_start=warm_start,
        use_drt_guided_guess=use_drt_guided_guess,
        drt_guide=drt_guide,
    )

    best_fit = None
    best_score = float("inf")
    best_meta: tuple[str, str] | None = None
    last_error = None
    for guess in guesses:
        guess_best = float("inf")
        success_count = 0
        seed_factors = SEED_FACTORS[:3] if batch_fast else SEED_FACTORS
        methods = ("least_squares",) if batch_fast else FIT_METHODS
        weights = ("modulus", "boukamp") if batch_fast else FIT_WEIGHTS
        for factor_index, factor in enumerate(seed_factors, start=1):
            for method in methods:
                for weight in weights:
                    try:
                        circuit = _build_pyimpspec_circuit(template_key, guess, factor)
                        result = fit_circuit(
                            circuit,
                            data,
                            method=method,
                            weight=weight,
                            max_nfev=FIT_CONFIG.pyimpspec_max_nfev,
                        )
                    except Exception as exc:
                        last_error = exc
                        continue
                    score = _fit_score(result)
                    success_count += 1
                    guess_best = min(guess_best, score)
                    if score < best_score:
                        best_fit = result
                        best_score = score
                        best_meta = (method, weight)
                    if guess_best < FIT_CONFIG.pyimpspec_good_score:
                        break
                if guess_best < FIT_CONFIG.pyimpspec_good_score:
                    break
            if guess_best < FIT_CONFIG.pyimpspec_good_score:
                break
            if not batch_fast and factor_index >= FIT_CONFIG.pyimpspec_factor_probe_count and success_count > 0 and guess_best < -0.5:
                break

    if best_fit is None:
        outcome = FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message=f"Numerical optimizer did not converge: {last_error}",
            masked_points=int((~point_mask).sum()),
            preprocess_actions=pp_actions,
        )
        return _attach_diagnosis(outcome, spectrum, template_key, segment_hint)

    full_predicted = best_fit.circuit.get_impedances(spectrum.freq_hz.astype(float))
    statistics = _statistics_from_fit(best_fit)
    parameters = _extract_parameters(template_key, best_fit.circuit)
    parameters.update(_primary_parameter_aliases(template_key, parameters))
    boundary_hits = _boundary_hits(best_fit.circuit)
    parameter_warnings = _parameter_warnings(best_fit)
    status = "ok"
    if boundary_hits or parameter_warnings:
        status = "warn"
    if float(statistics.get("pseudo_chi2", float("inf"))) > 1.0:
        status = "warn"

    notes = [f"method={best_meta[0]}", f"weight={best_meta[1]}"] if best_meta is not None else []
    if boundary_hits:
        notes.append("parameters near bounds: " + ", ".join(boundary_hits))
    if parameter_warnings:
        notes.append("uncertain parameters: " + ", ".join(parameter_warnings[:4]))

    outcome = FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message="; ".join(notes) if notes else "fit completed",
        parameters=parameters,
        statistics=statistics,
        predicted_real_ohm=full_predicted.real,
        predicted_imag_ohm=full_predicted.imag,
        masked_points=int((~point_mask).sum()),
        preprocess_actions=pp_actions,
    )
    return _attach_diagnosis(outcome, spectrum, template_key, segment_hint)


def _build_initial_guesses(
    template: CircuitTemplate,
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    *,
    warm_start: Optional[FitOutcome] = None,
    use_drt_guided_guess: bool = True,
    drt_guide: Optional[DrtGuide] = None,
) -> list[GuessPack]:
    """Return one or more initial-guess packs.  The geometric strategy is
    appended when it succeeds; callers loop over all packs."""
    primary = _build_initial_guess(template, spectrum, point_mask)
    packs = [primary]
    warm_pack = _build_initial_guess_from_fit(template, warm_start)
    if warm_pack is not None:
        packs.insert(0, warm_pack)
    geo = _build_initial_guess_geometric(template, spectrum, point_mask)
    if geo is not None:
        packs.append(geo)
    if use_drt_guided_guess:
        drt_pack = _build_initial_guess_drt(template, spectrum, point_mask, drt_guide=drt_guide)
        if drt_pack is not None:
            packs.append(drt_pack)
    return packs


def _build_initial_guess(template: CircuitTemplate, spectrum: SpectrumData, point_mask: np.ndarray) -> GuessPack:
    freq = spectrum.freq_hz[point_mask]
    z_real = spectrum.z_real_ohm[point_mask]
    y = -spectrum.z_imag_ohm[point_mask]
    span = max(float(np.nanmax(z_real) - np.nanmin(z_real)), 1e-3)
    rs = max(float(np.nanmin(z_real)), 1e-6)
    l_h = max(float(max(spectrum.z_imag_ohm[0], 0.0) / (2.0 * math.pi * max(spectrum.freq_hz[0], 1e-9))), 1e-9)
    main_peak_index = int(np.argmax(y))
    main_peak_freq = float(freq[main_peak_index]) if freq.size else 1.0
    n_default = _estimate_cpe_n(freq, y)
    q_main = max(1.0 / (max(span, 1e-6) * (2.0 * math.pi * max(main_peak_freq, 1e-6)) ** n_default), 1e-9)
    sigma = max(float(abs(spectrum.z_imag_ohm[-1]) / math.sqrt(2.0 * math.pi * max(spectrum.freq_hz[-1], 1e-9))), 1e-6)

    raise KeyError(template.key)


def _build_initial_guess_drt(
    template: CircuitTemplate,
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    drt_guide: Optional[DrtGuide] = None,
) -> Optional[GuessPack]:
    guide = drt_guide if drt_guide is not None else build_drt_guide(spectrum)
    if guide is None:
        return None

    rs_drt = guide.rs
    peaks = guide.peaks
    if not peaks:
        return None

    freq = spectrum.freq_hz[point_mask]
    z_real = spectrum.z_real_ohm[point_mask]
    span = max(float(np.nanmax(z_real) - np.nanmin(z_real)), 1e-3)
    rs = max(float(rs_drt), 1e-6)
    peak1 = peaks[0]
    peak2 = peaks[-1]

    return None


def build_drt_guide(spectrum: SpectrumData) -> Optional[DrtGuide]:
    try:
        from eismaster.analysis.native_drt import compute_drt, find_drt_peaks

        tau, gamma, rs_drt = compute_drt(spectrum)
        peaks = find_drt_peaks(tau, gamma)
    except Exception:
        logger.debug("DRT guided initial guess failed", exc_info=True)
        return None
    normalized_peaks = tuple({str(key): float(value) for key, value in peak.items()} for peak in peaks)
    return DrtGuide(rs=float(rs_drt), peaks=normalized_peaks)


def _build_initial_guess_from_fit(template: CircuitTemplate, warm_start: Optional[FitOutcome]) -> Optional[GuessPack]:
    if warm_start is None or warm_start.model_key != template.key:
        return None
    return None


def _build_initial_guess_geometric(
    template: CircuitTemplate, spectrum: SpectrumData, point_mask: np.ndarray
) -> Optional[GuessPack]:
    """Try to derive initial values by fitting a circle to the Nyquist arc.
    Returns *None* when the arc is too flat or noisy for a reliable circle fit."""
    try:
        x = spectrum.z_real_ohm[point_mask].astype(float)
        y_pos = (-spectrum.z_imag_ohm[point_mask]).astype(float)
        if x.size < 6 or np.all(y_pos <= 0):
            return None
        # keep only the portion with y > 0 (capacitive arc)
        keep = y_pos > 0
        if keep.sum() < 6:
            return None
        xk, yk = x[keep], y_pos[keep]
        # algebraic circle fit  (Kasa method)
        A = np.column_stack([xk, yk, np.ones(xk.size)])
        b_vec = xk**2 + yk**2
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        cx, cy = result[0] / 2.0, result[1] / 2.0
        radius = math.sqrt(max(result[2] + cx**2 + cy**2, 1e-12))
        if radius < 1e-6 or not np.isfinite(radius):
            return None
        rs_geo = max(cx - radius, 1e-9)
        rct_geo = max(2.0 * radius, 1e-6)
        freq_masked = spectrum.freq_hz[point_mask][keep]
        peak_idx = int(np.argmax(yk))
        peak_freq_geo = max(float(freq_masked[peak_idx]), 1e-6)
        n_geo = 0.85
        q_geo = max(1.0 / (rct_geo * (2.0 * math.pi * peak_freq_geo) ** n_geo), 1e-12)
        sigma_geo = max(float(abs(spectrum.z_imag_ohm[-1]) / math.sqrt(2.0 * math.pi * max(spectrum.freq_hz[-1], 1e-9))), 1e-6)
        l_h_geo = 1e-7
    except Exception:
        logger.debug("Operation failed", exc_info=True)
        return None

    return None


def _fit_zview_global(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    segment_hint: Optional[SegmentDetection] = None,
    warm_start: Optional[FitOutcome] = None,
    use_drt_guided_guess: bool = True,
    batch_fast: bool = False,
    drt_guide: Optional[DrtGuide] = None,
) -> FitOutcome:
    template = TEMPLATES["zview_segmented_rq_rwo"]
    if least_squares is None:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="unavailable",
            message="scipy is not installed; ZView-style global fitting unavailable.",
            masked_points=int((~point_mask).sum()),
        )

    detection = segment_hint or detect_segments(spectrum, mode="auto")
    if not detection.split_indices:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Unable to detect the semicircle-to-tail split point.",
            masked_points=int((~point_mask).sum()),
        )

    if detection.resolved_mode == "double" and len(detection.split_indices) >= 2:
        split_index = int(detection.split_indices[-1])
    else:
        split_index = int(detection.split_indices[0])

    if detection.peak_indices:
        peak_candidates = [int(index) for index in detection.peak_indices if index <= split_index]
        if peak_candidates:
            peak_index = max(peak_candidates, key=lambda index: float(spectrum.minus_z_imag_ohm[index]))
        else:
            peak_index = int(detection.peak_indices[0])
    else:
        peak_index = int(np.argmax(spectrum.minus_z_imag_ohm))
    masked_indices = np.flatnonzero(point_mask)
    arc_indices = masked_indices[masked_indices <= split_index]
    tail_indices = masked_indices[masked_indices > split_index]
    if arc_indices.size < 6 or tail_indices.size < 6:
        direct_result = _fit_zview_global_direct(spectrum, point_mask, detection, split_index, peak_index, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=("segmented tail too short; direct global fallback; " + (direct_result.message or "")).strip(),
                fallback_from=template.key,
            )
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Too few points in the detected arc or tail region.",
            masked_points=int((~point_mask).sum()),
        )

    if DataSet is None or fit_circuit is None or parse_cdc is None:
        return _fit_zview_global_direct(spectrum, point_mask, detection, split_index, peak_index, warm_start=warm_start)

    arc_data = DataSet(
        frequencies=spectrum.freq_hz[arc_indices].astype(float),
        impedances=spectrum.impedance[arc_indices].astype(complex),
        path=str(spectrum.metadata.file_path),
        label=f"{spectrum.display_name}-arc",
    )
    tail_data = DataSet(
        frequencies=spectrum.freq_hz[tail_indices].astype(float),
        impedances=spectrum.impedance[tail_indices].astype(complex),
        path=str(spectrum.metadata.file_path),
        label=f"{spectrum.display_name}-tail",
    )

    arc_fit = _fit_arc_region(spectrum, arc_data, split_index, peak_index, batch_fast=batch_fast)
    tail_fit = _fit_tail_region(spectrum, tail_data, split_index, batch_fast=batch_fast)
    if arc_fit is None or tail_fit is None:
        direct_result = _fit_zview_global_direct(spectrum, point_mask, detection, split_index, peak_index, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=("segmented init failed; direct global fallback; " + (direct_result.message or "")).strip(),
                fallback_from=template.key,
            )
        if arc_fit is not None:
            return _arc_fallback_outcome(template, spectrum, arc_fit, detection, point_mask)
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Segmented initial fitting failed; adjust the split point and retry.",
            masked_points=int((~point_mask).sum()),
        )

    global_result = _fit_full_zview_cnls(
        spectrum,
        point_mask,
        arc_fit,
        tail_fit,
        warm_start=warm_start,
        use_drt_guided_guess=use_drt_guided_guess,
        batch_fast=batch_fast,
        drt_guide=drt_guide,
    )
    if global_result is None:
        direct_result = _fit_zview_global_direct(spectrum, point_mask, detection, split_index, peak_index, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=("segmented global failed; direct global fallback; " + (direct_result.message or "")).strip(),
                fallback_from=template.key,
            )
        return _arc_fallback_outcome(template, spectrum, arc_fit, detection, point_mask)

    fitted = global_result.x
    predicted = _zview_full_model(spectrum.freq_hz.astype(float), fitted)
    statistics = _zview_statistics(global_result, int(point_mask.sum()))
    statistics["segment_arc_pseudo_chi2"] = float(arc_fit.pseudo_chisqr)
    statistics["segment_tail_pseudo_chi2"] = float(tail_fit.pseudo_chisqr)
    statistics["weight"] = str(getattr(global_result, "weight_tag", "calc-modulus"))
    arc_rs, arc_rs_err, arc_rp, arc_rp_err = _arc_region_resistances(arc_fit)

    parameters = {
        "Rs": float(fitted[0]),
        "CPE_T": float(fitted[1]),
        "CPE_P": float(fitted[2]),
        "Rct": float(arc_rp),
        "Rct_global": float(fitted[3]),
        "Wo_R": float(fitted[4]),
        "Wo_T": float(fitted[5]),
        "Wo_P": float(fitted[6]),
        "split_freq_hz": float(spectrum.freq_hz[split_index]),
        "split_index": float(split_index),
        "Rs_arc": float(arc_rs),
    }
    statistics["Rs_arc_stderr_pct"] = arc_rs_err
    statistics["Rct_stderr_pct"] = arc_rp_err

    warnings: list[str] = []
    boundary_hits = _zview_boundary_hits(global_result.x, global_result.bounds, _SINGLE_PARAM_NAMES)
    cnls_stats, cnls_warnings = _cnls_diagnostics(global_result, int(point_mask.sum()), ["Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P"])
    statistics.update(cnls_stats)
    if "Rs_stderr_pct" in statistics:
        statistics["Rs_global_stderr_pct"] = statistics["Rs_stderr_pct"]
    if "Rct_stderr_pct" in statistics:
        statistics["Rct_global_stderr_pct"] = statistics["Rct_stderr_pct"]
    statistics["Rs_stderr_pct"] = arc_rs_err
    statistics["Rct_stderr_pct"] = arc_rp_err
    status = "ok"
    if float(statistics.get("pseudo_chi2", float("inf"))) > 1.0 or boundary_hits or _is_high_error(arc_rs_err) or _is_high_error(arc_rp_err):
        status = "warn"
    if boundary_hits:
        warnings.append("parameters near bounds")
    if cnls_warnings:
        warnings.append("global reference uncertain: " + ", ".join(cnls_warnings[:4]))

    message = (
        f"mode={detection.resolved_mode}; split@{parameters['split_freq_hz']:.6g} Hz; "
        f"init=R(QR)+RWo; global=R(QRWo); weight={statistics['weight']}"
    )
    if detection.resolved_mode == "double":
        message = "geometry suggests double-semicircle; current result uses single-semicircle approximation; " + message
    if warnings:
        message += "; " + "; ".join(warnings)

    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message=message,
        parameters=parameters,
        statistics=statistics,
        predicted_real_ohm=predicted.real,
        predicted_imag_ohm=predicted.imag,
        masked_points=int((~point_mask).sum()),
        confidence_intervals=_confidence_intervals_from_stats(statistics, ["Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P"]),
        correlation_matrix_max=_corr_max_from_stats(statistics),
    )


def _fit_zview_double_global(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    segment_hint: Optional[SegmentDetection] = None,
    warm_start: Optional[FitOutcome] = None,
    use_drt_guided_guess: bool = True,
    batch_fast: bool = False,
    drt_guide: Optional[DrtGuide] = None,
) -> FitOutcome:
    template = TEMPLATES["zview_double_rq_qrwo"]
    if least_squares is None:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="unavailable",
            message="scipy is not installed; ZView-style global fitting unavailable.",
            masked_points=int((~point_mask).sum()),
        )

    detection = segment_hint or detect_segments(spectrum, mode="double")
    if detection.resolved_mode != "double" or len(detection.split_indices) < 2:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Unable to identify two semicircle regions and the low-frequency tail.",
            masked_points=int((~point_mask).sum()),
        )

    split1, split2 = map(int, detection.split_indices[:2])
    peak1 = int(detection.peak_indices[0]) if len(detection.peak_indices) >= 1 else split1 // 2
    peak2 = int(detection.peak_indices[1]) if len(detection.peak_indices) >= 2 else (split1 + split2) // 2
    masked_indices = np.flatnonzero(point_mask)
    arc1_indices = masked_indices[masked_indices <= split1]
    arc2_indices = masked_indices[(masked_indices > split1) & (masked_indices <= split2)]
    tail_indices = masked_indices[masked_indices > split2]
    if arc1_indices.size < 6 or arc2_indices.size < 6:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Too few points in one of the detected double-semicircle regions.",
            masked_points=int((~point_mask).sum()),
        )
    if tail_indices.size < 6:
        direct_result = _fit_zview_double_global_direct(spectrum, point_mask, detection, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=(
                    "Double-arc segmented tail is too short; fell back to direct global fitting. "
                    + (direct_result.message or "")
                ).strip(),
            )
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Too few low-frequency tail points after the second split; move split2 left or import more low-frequency data.",
            masked_points=int((~point_mask).sum()),
        )

    if DataSet is None or fit_circuit is None or parse_cdc is None:
        return _fit_zview_double_global_direct(spectrum, point_mask, detection, warm_start=warm_start)

    arc1_data = DataSet(
        frequencies=spectrum.freq_hz[arc1_indices].astype(float),
        impedances=spectrum.impedance[arc1_indices].astype(complex),
        path=str(spectrum.metadata.file_path),
        label=f"{spectrum.display_name}-arc1",
    )
    arc2_data = DataSet(
        frequencies=spectrum.freq_hz[arc2_indices].astype(float),
        impedances=spectrum.impedance[arc2_indices].astype(complex),
        path=str(spectrum.metadata.file_path),
        label=f"{spectrum.display_name}-arc2",
    )
    tail_data = DataSet(
        frequencies=spectrum.freq_hz[tail_indices].astype(float),
        impedances=spectrum.impedance[tail_indices].astype(complex),
        path=str(spectrum.metadata.file_path),
        label=f"{spectrum.display_name}-tail",
    )

    arc1_fit = _fit_arc_region(spectrum, arc1_data, split1, peak1, batch_fast=batch_fast)
    arc2_fit = _fit_arc_region_on_window(spectrum, arc2_data, arc2_indices, peak2, batch_fast=batch_fast)
    tail_fit = _fit_tail_region(spectrum, tail_data, split2, batch_fast=batch_fast)
    if arc1_fit is None or arc2_fit is None or tail_fit is None:
        direct_result = _fit_zview_double_global_direct(spectrum, point_mask, detection, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=("segmented init failed; direct global fallback; " + (direct_result.message or "")).strip(),
                fallback_from=template.key,
            )
        if arc1_fit is not None or arc2_fit is not None:
            return _double_arc_fallback_outcome(template, spectrum, arc1_fit, arc2_fit, detection, point_mask)
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Segmented initial fitting failed for the double-semicircle model.",
            masked_points=int((~point_mask).sum()),
        )

    global_result = _fit_full_zview_double_cnls(
        spectrum,
        point_mask,
        arc1_fit,
        arc2_fit,
        tail_fit,
        detection,
        warm_start=warm_start,
        use_drt_guided_guess=use_drt_guided_guess,
        batch_fast=batch_fast,
        drt_guide=drt_guide,
    )
    if global_result is None:
        direct_result = _fit_zview_double_global_direct(spectrum, point_mask, detection, warm_start=warm_start)
        if direct_result.status != "failed":
            return replace(
                direct_result,
                message=("segmented global failed; direct global fallback; " + (direct_result.message or "")).strip(),
                fallback_from=template.key,
            )
        return _double_arc_fallback_outcome(template, spectrum, arc1_fit, arc2_fit, detection, point_mask)

    fitted = global_result.x
    predicted = _zview_double_model(spectrum.freq_hz.astype(float), fitted)
    statistics = _zview_statistics(global_result, int(point_mask.sum()))
    statistics["segment_arc1_pseudo_chi2"] = float(arc1_fit.pseudo_chisqr)
    statistics["segment_arc2_pseudo_chi2"] = float(arc2_fit.pseudo_chisqr)
    statistics["segment_tail_pseudo_chi2"] = float(tail_fit.pseudo_chisqr)
    statistics["weight"] = str(getattr(global_result, "weight_tag", "calc-modulus"))
    arc1_rs, arc1_rs_err, arc1_rp, arc1_rp_err = _arc_region_resistances(arc1_fit)
    arc2_rs, arc2_rs_err, arc2_rp, arc2_rp_err = _arc_region_resistances(arc2_fit)

    parameters = {
        "Rs": float(fitted[0]),
        "Q1": float(fitted[1]),
        "n1": float(fitted[2]),
        "Rsei": float(arc1_rp),
        "Rsei_global": float(fitted[3]),
        "Q2": float(fitted[4]),
        "n2": float(fitted[5]),
        "Rct": float(arc2_rp),
        "Rct_global": float(fitted[6]),
        "Wo_R": float(fitted[7]),
        "Wo_T": float(fitted[8]),
        "Wo_P": float(fitted[9]),
        "split1_freq_hz": float(spectrum.freq_hz[split1]),
        "split2_freq_hz": float(spectrum.freq_hz[split2]),
        "Rs_arc1": float(arc1_rs),
        "Rs_arc2": float(arc2_rs),
    }
    statistics["Rs_arc1_stderr_pct"] = arc1_rs_err
    statistics["Rs_arc2_stderr_pct"] = arc2_rs_err
    statistics["Rsei_stderr_pct"] = arc1_rp_err
    statistics["Rct_stderr_pct"] = arc2_rp_err

    warnings: list[str] = []
    boundary_hits = _zview_boundary_hits(global_result.x, global_result.bounds, _DOUBLE_PARAM_NAMES)
    cnls_stats, cnls_warnings = _cnls_diagnostics(
        global_result,
        int(point_mask.sum()),
        ["Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P"],
    )
    statistics.update(cnls_stats)
    if "Rs_stderr_pct" in statistics:
        statistics["Rs_global_stderr_pct"] = statistics["Rs_stderr_pct"]
    if "Rsei_stderr_pct" in statistics:
        statistics["Rsei_global_stderr_pct"] = statistics["Rsei_stderr_pct"]
    if "Rct_stderr_pct" in statistics:
        statistics["Rct_global_stderr_pct"] = statistics["Rct_stderr_pct"]
    statistics["Rs_stderr_pct"] = arc1_rs_err
    statistics["Rsei_stderr_pct"] = arc1_rp_err
    statistics["Rct_stderr_pct"] = arc2_rp_err
    status = "ok"
    if (
        float(statistics.get("pseudo_chi2", float("inf"))) > 1.0
        or boundary_hits
        or _is_high_error(arc1_rs_err)
        or _is_high_error(arc1_rp_err)
        or _is_high_error(arc2_rp_err)
    ):
        status = "warn"
    if boundary_hits:
        warnings.append("parameters near bounds")
    if cnls_warnings:
        warnings.append("global reference uncertain: " + ", ".join(cnls_warnings[:4]))

    message = (
        f"mode=double; split1@{parameters['split1_freq_hz']:.6g} Hz; "
        f"split2@{parameters['split2_freq_hz']:.6g} Hz; "
        f"init=R(QR)+R(QR)+RWo; global=R(QR)(Q(RWo)); weight={statistics['weight']}"
    )
    if warnings:
        message += "; " + "; ".join(warnings)

    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message=message,
        parameters=parameters,
        statistics=statistics,
        predicted_real_ohm=predicted.real,
        predicted_imag_ohm=predicted.imag,
        masked_points=int((~point_mask).sum()),
        confidence_intervals=_confidence_intervals_from_stats(statistics, ["Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P"]),
        correlation_matrix_max=_corr_max_from_stats(statistics),
    )


def _fit_zview_global_direct(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    detection: SegmentDetection,
    split_index: int,
    peak_index: int,
    warm_start: Optional[FitOutcome] = None,
) -> FitOutcome:
    template = TEMPLATES["zview_segmented_rq_rwo"]
    freq = spectrum.freq_hz[point_mask].astype(float)
    z_exp = spectrum.impedance[point_mask].astype(complex)
    split_freq = max(float(spectrum.freq_hz[split_index]), 1e-6)
    peak_freq = max(float(spectrum.freq_hz[peak_index]), 1e-6)
    rs_guess = max(float(np.min(spectrum.z_real_ohm[point_mask])), 1e-6)
    split_real = float(spectrum.z_real_ohm[split_index])
    tail_real = float(spectrum.z_real_ohm[np.flatnonzero(point_mask)[-1]])
    rct_guess = max(split_real - rs_guess, 1e-4)
    wo_r_guess = max(tail_real - split_real, max(rct_guess * 0.15, 1e-4))
    cpe_p_guess = 0.9
    cpe_t_guess = max(1.0 / (rct_guess * (2.0 * math.pi * peak_freq) ** cpe_p_guess), 1e-12)
    wo_t_guess = max(1.0 / (2.0 * math.pi * split_freq), 1e-6)
    wo_p_guess = 0.5
    x0 = np.asarray([rs_guess, cpe_t_guess, cpe_p_guess, rct_guess, wo_r_guess, wo_t_guess, wo_p_guess], dtype=float)
    lower = np.asarray([1e-9, 1e-12, 0.2, 1e-9, 1e-9, 1e-9, 0.2], dtype=float)
    upper = np.asarray([1e6, 1e0, 1.0, 1e8, 1e8, 1e6, 1.0], dtype=float)
    seeds = _warm_start_zview_single_seeds(warm_start) + [
        x0,
        x0 * np.asarray([1.0, 1.3, 1.0, 0.8, 1.1, 1.0, 1.0]),
        x0 * np.asarray([1.0, 0.7, 1.0, 1.2, 0.9, 1.2, 1.0]),
    ]
    result = _solve_zview_cnls(freq, z_exp, _zview_full_model, lower, upper, seeds, param_count=x0.size)
    if result is None:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Direct ZView-style CNLS did not converge.",
            masked_points=int((~point_mask).sum()),
        )

    predicted = _zview_full_model(spectrum.freq_hz.astype(float), result.x)
    statistics = _zview_statistics(result, int(point_mask.sum()))
    statistics["weight"] = str(getattr(result, "weight_tag", "calc-modulus"))
    cnls_stats, cnls_warnings = _cnls_diagnostics(result, int(point_mask.sum()), ["Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P"])
    statistics.update(cnls_stats)
    boundary_hits = _zview_boundary_hits(result.x, result.bounds, _SINGLE_PARAM_NAMES)
    status = "warn" if boundary_hits or cnls_warnings or float(statistics.get("pseudo_chi2", float("inf"))) > 1.0 else "ok"
    message = (
        f"mode={detection.resolved_mode}; split@{float(spectrum.freq_hz[split_index]):.6g} Hz; "
        f"path=direct-global; model=R(QRWo); weight={statistics['weight']}"
    )
    if boundary_hits:
        message += "; parameters near bounds"
    if cnls_warnings:
        message += "; uncertain parameters: " + ", ".join(cnls_warnings[:4])
    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message=message,
        parameters={
            "Rs": float(result.x[0]),
            "CPE_T": float(result.x[1]),
            "CPE_P": float(result.x[2]),
            "Rct": float(result.x[3]),
            "Wo_R": float(result.x[4]),
            "Wo_T": float(result.x[5]),
            "Wo_P": float(result.x[6]),
            "split_freq_hz": float(spectrum.freq_hz[split_index]),
            "split_index": float(split_index),
        },
        statistics=statistics,
        predicted_real_ohm=predicted.real,
        predicted_imag_ohm=predicted.imag,
        masked_points=int((~point_mask).sum()),
        confidence_intervals=_confidence_intervals_from_stats(statistics, ["Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P"]),
        correlation_matrix_max=_corr_max_from_stats(statistics),
    )


def _fit_zview_double_global_direct(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    detection: SegmentDetection,
    warm_start: Optional[FitOutcome] = None,
) -> FitOutcome:
    template = TEMPLATES["zview_double_rq_qrwo"]
    freq = spectrum.freq_hz[point_mask].astype(float)
    z_exp = spectrum.impedance[point_mask].astype(complex)
    split1, split2 = map(int, detection.split_indices[:2])
    peak1 = int(detection.peak_indices[0])
    peak2 = int(detection.peak_indices[1])
    rs_guess = max(float(np.min(spectrum.z_real_ohm[point_mask])), 1e-6)
    split1_real = float(spectrum.z_real_ohm[split1])
    split2_real = float(spectrum.z_real_ohm[split2])
    tail_real = float(spectrum.z_real_ohm[np.flatnonzero(point_mask)[-1]])
    rsei_guess = max(split1_real - rs_guess, 1e-4)
    rct_guess = max(split2_real - split1_real, 1e-4)
    n1_guess = 0.9
    n2_guess = 0.85
    peak1_freq = max(float(spectrum.freq_hz[peak1]), 1e-6)
    peak2_freq = max(float(spectrum.freq_hz[peak2]), 1e-6)
    q1_guess = max(1.0 / (rsei_guess * (2.0 * math.pi * peak1_freq) ** n1_guess), 1e-12)
    q2_guess = max(1.0 / (rct_guess * (2.0 * math.pi * peak2_freq) ** n2_guess), 1e-12)
    wo_r_guess = max(tail_real - split2_real, max(rct_guess * 0.15, 1e-4))
    wo_t_guess = max(1.0 / (2.0 * math.pi * max(float(spectrum.freq_hz[split2]), 1e-6)), 1e-6)
    wo_p_guess = 0.5
    x0 = np.asarray([rs_guess, q1_guess, n1_guess, rsei_guess, q2_guess, n2_guess, rct_guess, wo_r_guess, wo_t_guess, wo_p_guess], dtype=float)
    lower = np.asarray([1e-9, 1e-12, 0.2, 1e-9, 1e-12, 0.2, 1e-9, 1e-9, 1e-9, 0.2], dtype=float)
    upper = np.asarray([1e6, 1e0, 1.0, 1e8, 1e0, 1.0, 1e8, 1e8, 1e6, 1.0], dtype=float)
    seeds = _warm_start_zview_double_seeds(warm_start) + [
        x0,
        x0 * np.asarray([1.0, 1.2, 1.0, 0.8, 1.2, 1.0, 1.1, 1.1, 1.0, 1.0]),
        x0 * np.asarray([1.0, 0.8, 1.0, 1.2, 0.8, 1.0, 0.9, 0.9, 1.2, 1.0]),
    ]
    result = _solve_zview_cnls(freq, z_exp, _zview_double_model, lower, upper, seeds, param_count=x0.size)
    if result is None:
        return FitOutcome(
            model_key=template.key,
            model_label=template.label,
            status="failed",
            message="Direct double-arc ZView-style CNLS did not converge.",
            masked_points=int((~point_mask).sum()),
        )

    predicted = _zview_double_model(spectrum.freq_hz.astype(float), result.x)
    statistics = _zview_statistics(result, int(point_mask.sum()))
    statistics["weight"] = str(getattr(result, "weight_tag", "calc-modulus"))
    cnls_stats, cnls_warnings = _cnls_diagnostics(result, int(point_mask.sum()), ["Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P"])
    statistics.update(cnls_stats)
    if "Rsei_stderr_pct" in statistics:
        statistics["Rsei_global_stderr_pct"] = statistics["Rsei_stderr_pct"]
    if "Rct_stderr_pct" in statistics:
        statistics["Rct_global_stderr_pct"] = statistics["Rct_stderr_pct"]
    boundary_hits = _zview_boundary_hits(result.x, result.bounds, _DOUBLE_PARAM_NAMES)
    status = "warn" if boundary_hits or cnls_warnings or float(statistics.get("pseudo_chi2", float("inf"))) > 1.0 else "ok"
    message = (
        f"mode={detection.resolved_mode}; split1@{float(spectrum.freq_hz[split1]):.6g} Hz; "
        f"split2@{float(spectrum.freq_hz[split2]):.6g} Hz; path=direct-global; model=R(QR)(Q(RWo)); "
        f"weight={statistics['weight']}"
    )
    if boundary_hits:
        message += "; parameters near bounds"
    if cnls_warnings:
        message += "; uncertain parameters: " + ", ".join(cnls_warnings[:4])
    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message=message,
        parameters={
            "Rs": float(result.x[0]),
            "Q1": float(result.x[1]),
            "n1": float(result.x[2]),
            "Rsei": float(result.x[3]),
            "Rsei_global": float(result.x[3]),
            "Q2": float(result.x[4]),
            "n2": float(result.x[5]),
            "Rct": float(result.x[6]),
            "Rct_global": float(result.x[6]),
            "Wo_R": float(result.x[7]),
            "Wo_T": float(result.x[8]),
            "Wo_P": float(result.x[9]),
            "split1_freq_hz": float(spectrum.freq_hz[split1]),
            "split2_freq_hz": float(spectrum.freq_hz[split2]),
        },
        statistics=statistics,
        predicted_real_ohm=predicted.real,
        predicted_imag_ohm=predicted.imag,
        masked_points=int((~point_mask).sum()),
        confidence_intervals=_confidence_intervals_from_stats(statistics, ["Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P"]),
        correlation_matrix_max=_corr_max_from_stats(statistics),
    )


def _solve_zview_cnls(
    freq: np.ndarray,
    z_exp: np.ndarray,
    model_fn,
    lower: np.ndarray,
    upper: np.ndarray,
    seeds: list[np.ndarray],
    *,
    param_count: int,
    max_nfev: int = 40000,
    weight_tags: Optional[tuple[str, ...]] = None,
    seed_limit: Optional[int] = None,
) -> Optional[OptimizeResult]:
    best = None
    best_score = float("inf")
    tags = weight_tags or ZVIEW_CNLS_WEIGHTS
    if seed_limit is not None:
        seeds = seeds[:seed_limit]

    # Two-stage sniffing: W×S → S+W calls
    # Stage 1: all seeds with fastest weight → find best seed
    # Stage 2: all weights on best seed → pick best weight
    probe_tag = tags[0]  # "calc-modulus" — fastest weight
    best_seed = None
    best_seed_score = float("inf")
    for seed in seeds:
        s = np.clip(np.asarray(seed, dtype=float), lower * 1.001, upper * 0.999)
        try:
            result = least_squares(
                lambda p, t=probe_tag: _zview_residual(freq, z_exp, p, model_fn, t),
                s,
                bounds=(lower, upper),
                method="trf",
                max_nfev=max_nfev,
                x_scale="jac",
                ftol=1e-6,
                xtol=1e-6,
            )
        except Exception:
            continue
        score = _cnls_selection_score(result, param_count)
        if score < best_seed_score:
            best_seed_score = score
            best_seed = np.asarray(result.x, dtype=float)

    if best_seed is None:
        return None

    # Stage 2: all weights on the best seed
    for weight_tag in tags:
        s = np.clip(best_seed, lower * 1.001, upper * 0.999)
        try:
            result = least_squares(
                lambda p, t=weight_tag: _zview_residual(freq, z_exp, p, model_fn, t),
                s,
                bounds=(lower, upper),
                method="trf",
                max_nfev=max_nfev,
                x_scale="jac",
                ftol=1e-6,
                xtol=1e-6,
            )
        except Exception:
            continue
        result.bounds = (lower, upper)  # type: ignore[attr-defined]
        result.weight_tag = weight_tag  # type: ignore[attr-defined]
        score = _cnls_selection_score(result, param_count)
        if score < best_score:
            best = result
            best_score = score
    return best


def _fit_arc_region(spectrum: SpectrumData, arc_data: DataSet, split_index: int, peak_index: int, *, batch_fast: bool = False):
    base_circuit = parse_cdc("R(QR)")
    elements = base_circuit.get_elements()
    arc_indices = np.arange(split_index + 1, dtype=int)
    return _fit_arc_region_on_window(spectrum, arc_data, arc_indices, peak_index, batch_fast=batch_fast)


def _fit_arc_region_on_window(spectrum: SpectrumData, arc_data: DataSet, arc_indices: np.ndarray, peak_index: int, *, batch_fast: bool = False):
    base_circuit = parse_cdc("R(QR)")
    elements = base_circuit.get_elements()
    z_real = spectrum.z_real_ohm[arc_indices]
    span = max(float(np.nanmax(z_real) - np.nanmin(z_real)), 1e-3)
    rs = max(float(np.nanmin(z_real)), 1e-6)
    peak_freq = float(spectrum.freq_hz[peak_index])
    n_default = _estimate_cpe_n(
        spectrum.freq_hz[arc_indices].astype(float),
        spectrum.minus_z_imag_ohm[arc_indices].astype(float),
    )
    q_main = max(1.0 / (max(span, 1e-6) * (2.0 * math.pi * max(peak_freq, 1e-6)) ** n_default), 1e-9)
    elements[0].set_values(R=rs)
    elements[1].set_values(Y=q_main, n=n_default)
    elements[2].set_values(R=span)
    best_fit = None
    best_score = float("inf")
    best_hits = float("inf")
    methods = ("least_squares",) if batch_fast else ("least_squares", "lbfgsb")
    weights = ("modulus", "boukamp") if batch_fast else ("modulus", "boukamp", "proportional")
    for method in methods:
        for weight in weights:
            try:
                circuit = parse_cdc("R(QR)")
                circuit.get_elements()[0].set_values(R=rs)
                circuit.get_elements()[1].set_values(Y=q_main, n=n_default)
                circuit.get_elements()[2].set_values(R=span)
                result = fit_circuit(circuit, arc_data, method=method, weight=weight, max_nfev=FIT_CONFIG.pyimpspec_max_nfev)
                score = float(getattr(result, "pseudo_chisqr", float("inf")))
                hits = float(len(_boundary_hits(result.circuit)))
                if score < best_score or (math.isclose(score, best_score) and hits < best_hits):
                    best_fit = result
                    best_score = score
                    best_hits = hits
            except Exception:
                logger.debug("Arc window fit candidate failed", exc_info=True)
    return best_fit


def _fit_tail_region(spectrum: SpectrumData, tail_data: DataSet, split_index: int, *, batch_fast: bool = False):
    circuit = parse_cdc("RWo")
    elements = circuit.get_elements()
    rs_guess = max(float(spectrum.z_real_ohm[split_index]), 1e-6)
    f_ref = max(float(spectrum.freq_hz[split_index]), 1e-6)
    elements[0].set_values(R=rs_guess)
    elements[1].set_values(Y=5e-5, B=3.0 / (2.0 * math.pi * f_ref), n=0.5).set_fixed(n=True)
    try:
        coarse = fit_circuit(circuit, tail_data, method="least_squares", weight="modulus", max_nfev=FIT_CONFIG.pyimpspec_max_nfev)
    except Exception:
        logger.debug("Operation failed", exc_info=True)
        return None
    if not batch_fast:
        try:
            refined = coarse.circuit.copy()
            refined_elements = refined.get_elements()
            refined_elements[1].set_fixed(n=False)
            refined_elements[1].set_lower_limits(
                Y=max(float(refined_elements[1].get_values()["Y"]) * 1e-3, 1e-12),
                B=max(float(refined_elements[1].get_values()["B"]) * 1e-2, 1e-9),
                n=0.2,
            ).set_upper_limits(
                Y=max(float(refined_elements[1].get_values()["Y"]) * 1e3, 1e-9),
                B=max(float(refined_elements[1].get_values()["B"]) * 1e2, 1e-6),
                n=1.0,
            )
            fine = fit_circuit(refined, tail_data, method="least_squares", weight="modulus", max_nfev=FIT_CONFIG.pyimpspec_max_nfev)
            if float(getattr(fine, "pseudo_chisqr", float("inf"))) <= float(getattr(coarse, "pseudo_chisqr", float("inf"))):
                return fine
        except Exception:
            logger.debug("Tail refinement failed", exc_info=True)
    return coarse


def _arc_region_resistances(fit_result) -> tuple[float, float, float, float]:
    elements = fit_result.circuit.get_elements()
    params_df = fit_result.to_parameters_dataframe()
    rs = float(elements[0].get_values()["R"])
    rp = float(elements[2].get_values()["R"])
    rs_err = _stderr_pct_from_dataframe(params_df, 0)
    rp_err = _stderr_pct_from_dataframe(params_df, 3)
    return rs, rs_err, rp, rp_err


def _stderr_pct_from_dataframe(params_df, row_index: int) -> float:
    try:
        value = float(params_df.iloc[row_index]["Std. err. (%)"])
    except Exception:
        return float("nan")
    return value


def _is_high_error(value: float, threshold: float = FIT_CONFIG.primary_error_warn_pct) -> bool:
    return not np.isfinite(value) or float(value) > threshold


def _zview_wo_r_from_pyimpspec(y_value: float, t_value: float, p_value: float) -> float:
    if y_value <= 0 or t_value <= 0 or p_value <= 0:
        return float("nan")
    return float((t_value / y_value) ** p_value)


def _warm_start_zview_single_seeds(warm_start: Optional[FitOutcome]) -> list[np.ndarray]:
    if warm_start is None or warm_start.model_key != "zview_segmented_rq_rwo":
        return []
    params = warm_start.parameters
    required = ("Rs", "CPE_T", "CPE_P", "Rct", "Wo_R", "Wo_T", "Wo_P")
    try:
        base = np.asarray([float(params[name]) for name in required], dtype=float)
    except Exception:
        return []
    return [
        base,
        base * np.asarray([1.0, 1.05, 1.0, 1.05, 0.95, 1.0, 1.0]),
        base * np.asarray([1.0, 0.95, 1.0, 0.95, 1.05, 1.0, 1.0]),
    ]


def _warm_start_zview_double_seeds(warm_start: Optional[FitOutcome]) -> list[np.ndarray]:
    if warm_start is None or warm_start.model_key != "zview_double_rq_qrwo":
        return []
    params = warm_start.parameters
    required = ("Rs", "Q1", "n1", "Rsei", "Q2", "n2", "Rct", "Wo_R", "Wo_T", "Wo_P")
    try:
        base = np.asarray([float(params[name]) for name in required], dtype=float)
    except Exception:
        return []
    return [
        base,
        base * np.asarray([1.0, 1.05, 1.0, 1.05, 1.05, 1.0, 0.95, 0.95, 1.0, 1.0]),
        base * np.asarray([1.0, 0.95, 1.0, 0.95, 0.95, 1.0, 1.05, 1.05, 1.0, 1.0]),
    ]


def _fit_full_zview_cnls(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    arc_fit,
    tail_fit,
    warm_start: Optional[FitOutcome] = None,
    use_drt_guided_guess: bool = True,
    batch_fast: bool = False,
    drt_guide: Optional[DrtGuide] = None,
):
    arc_elements = arc_fit.circuit.get_elements()
    tail_elements = tail_fit.circuit.get_elements()
    x0 = np.asarray(
        [
            float(arc_elements[0].get_values()["R"]),
            float(arc_elements[1].get_values()["Y"]),
            float(arc_elements[1].get_values()["n"]),
            float(tail_elements[0].get_values()["R"]),
            _zview_wo_r_from_pyimpspec(
                float(tail_elements[1].get_values()["Y"]),
                float(tail_elements[1].get_values()["B"]),
                float(tail_elements[1].get_values()["n"]),
            ),
            float(tail_elements[1].get_values()["B"]),
            float(tail_elements[1].get_values()["n"]),
        ],
        dtype=float,
    )
    lower = np.asarray([1e-9, 1e-10, 0.2, 1e-9, 1e-9, 1e-9, 0.2], dtype=float)
    upper = np.asarray([1e3, 1e0, 1.0, 1e5, 1e5, 1e3, 1.0], dtype=float)
    freq = spectrum.freq_hz[point_mask].astype(float)
    z_exp = spectrum.impedance[point_mask].astype(complex)
    seeds = _warm_start_zview_single_seeds(warm_start) + [
        x0,
        x0 * np.asarray([1.0, 1.0, 1.0, 0.85, 1.15, 1.0, 1.0]),
        x0 * np.asarray([1.0, 1.0, 1.0, 1.15, 0.85, 1.0, 1.0]),
    ]
    result = _solve_zview_cnls(
        freq,
        z_exp,
        _zview_full_model,
        lower,
        upper,
        seeds,
        param_count=len(x0),
        max_nfev=FIT_CONFIG.zview_single_fast_max_nfev if batch_fast else FIT_CONFIG.zview_single_full_max_nfev,
        weight_tags=("calc-modulus", "calc-unit") if batch_fast else None,
        seed_limit=3 if batch_fast else None,
    )
    if result is not None or not use_drt_guided_guess:
        return result
    guide = drt_guide if drt_guide is not None else build_drt_guide(spectrum)
    if guide is None:
        return result
    peaks = guide.peaks
    if len(peaks) >= 1:
        p1 = max(peaks, key=lambda p: p["R"])
        q1 = max(1.0 / (max(p1["R"], 1e-4) * (2 * math.pi / max(p1["tau"], 1e-6)) ** p1["n"]), 1e-12)
        drt_x0 = x0.copy()
        drt_x0[0] = max(guide.rs, 1e-6)
        drt_x0[1] = q1
        drt_x0[2] = p1["n"]
        drt_x0[3] = max(p1["R"], 1e-4)
        seeds = [drt_x0] + seeds[:2]
        return _solve_zview_cnls(
            freq,
            z_exp,
            _zview_full_model,
            lower,
            upper,
            seeds,
            param_count=len(x0),
            max_nfev=18000 if batch_fast else FIT_CONFIG.zview_single_retry_max_nfev,
            weight_tags=("calc-modulus", "calc-unit") if batch_fast else None,
            seed_limit=2 if batch_fast else None,
        )
    return result


def _fit_full_zview_double_cnls(
    spectrum: SpectrumData,
    point_mask: np.ndarray,
    arc1_fit,
    arc2_fit,
    tail_fit,
    detection: Optional[SegmentDetection] = None,
    warm_start: Optional[FitOutcome] = None,
    use_drt_guided_guess: bool = True,
    batch_fast: bool = False,
    drt_guide: Optional[DrtGuide] = None,
):
    arc1_elements = arc1_fit.circuit.get_elements()
    arc2_elements = arc2_fit.circuit.get_elements()
    tail_elements = tail_fit.circuit.get_elements()
    x0 = np.asarray(
        [
            float(arc1_elements[0].get_values()["R"]),
            float(arc1_elements[1].get_values()["Y"]),
            float(arc1_elements[1].get_values()["n"]),
            float(arc1_elements[2].get_values()["R"]),
            float(arc2_elements[1].get_values()["Y"]),
            float(arc2_elements[1].get_values()["n"]),
            float(arc2_elements[2].get_values()["R"]),
            _zview_wo_r_from_pyimpspec(
                float(tail_elements[1].get_values()["Y"]),
                float(tail_elements[1].get_values()["B"]),
                float(tail_elements[1].get_values()["n"]),
            ),
            float(tail_elements[1].get_values()["B"]),
            float(tail_elements[1].get_values()["n"]),
        ],
        dtype=float,
    )
    lower = np.asarray([1e-9, 1e-10, 0.2, 1e-9, 1e-10, 0.2, 1e-9, 1e-9, 1e-9, 0.2], dtype=float)
    upper = np.asarray([1e4, 1e0, 1.0, 1e6, 1e0, 1.0, 1e6, 1e6, 1e4, 1.0], dtype=float)
    freq = spectrum.freq_hz[point_mask].astype(float)
    z_exp = spectrum.impedance[point_mask].astype(complex)
    seeds = _warm_start_zview_double_seeds(warm_start) + [
        x0,
        x0 * np.asarray([1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 1.1, 1.0, 1.0]),
        x0 * np.asarray([1.0, 1.0, 1.0, 1.2, 1.0, 1.0, 1.2, 0.9, 1.0, 1.0]),
    ]
    if detection is not None and len(detection.peak_indices) >= 2:
        peak1 = int(detection.peak_indices[0])
        peak2 = int(detection.peak_indices[1])
        split1 = int(detection.split_indices[0]) if len(detection.split_indices) >= 1 else peak1 + 1
        split2 = int(detection.split_indices[1]) if len(detection.split_indices) >= 2 else peak2 + 1
        rs_guess = max(float(spectrum.z_real_ohm[0]), 1e-6)
        rsei_guess = max(float(spectrum.z_real_ohm[split1] - spectrum.z_real_ohm[0]), 1e-6)
        rct_guess = max(float(spectrum.z_real_ohm[split2] - spectrum.z_real_ohm[split1]), 1e-6)
        n_guess = 0.9
        peak1_freq = max(float(spectrum.freq_hz[peak1]), 1e-6)
        peak2_freq = max(float(spectrum.freq_hz[peak2]), 1e-6)
        q1_guess = max(1.0 / (rsei_guess * (2.0 * math.pi * peak1_freq) ** n_guess), 1e-12)
        q2_guess = max(1.0 / (rct_guess * (2.0 * math.pi * peak2_freq) ** n_guess), 1e-12)
        guided_seed = x0.copy()
        guided_seed[0] = rs_guess
        guided_seed[1] = q1_guess
        guided_seed[2] = n_guess
        guided_seed[3] = rsei_guess
        guided_seed[4] = q2_guess
        guided_seed[5] = n_guess
        guided_seed[6] = rct_guess
        seeds.append(guided_seed)
    result = _solve_zview_cnls(
        freq,
        z_exp,
        _zview_double_model,
        lower,
        upper,
        seeds,
        param_count=len(x0),
        max_nfev=FIT_CONFIG.zview_double_fast_max_nfev if batch_fast else FIT_CONFIG.zview_double_full_max_nfev,
        weight_tags=("calc-modulus", "calc-unit") if batch_fast else None,
        seed_limit=3 if batch_fast else None,
    )
    if result is not None or not use_drt_guided_guess:
        return result
    guide = drt_guide if drt_guide is not None else build_drt_guide(spectrum)
    if guide is None:
        return result
    peaks = guide.peaks
    if len(peaks) >= 2:
        p1 = peaks[0]
        p2 = peaks[-1]
        q1 = max(1.0 / (max(p1["R"], 1e-4) * (2 * math.pi / max(p1["tau"], 1e-6)) ** p1["n"]), 1e-12)
        q2 = max(1.0 / (max(p2["R"], 1e-4) * (2 * math.pi / max(p2["tau"], 1e-6)) ** p2["n"]), 1e-12)
        drt_x0 = x0.copy()
        drt_x0[0] = max(guide.rs, 1e-6)
        drt_x0[1] = q1
        drt_x0[2] = p1["n"]
        drt_x0[3] = max(p1["R"], 1e-4)
        drt_x0[4] = q2
        drt_x0[5] = p2["n"]
        drt_x0[6] = max(p2["R"], 1e-4)
        seeds = [drt_x0] + seeds[:2]
        return _solve_zview_cnls(
            freq,
            z_exp,
            _zview_double_model,
            lower,
            upper,
            seeds,
            param_count=len(x0),
            max_nfev=22000 if batch_fast else FIT_CONFIG.zview_double_retry_max_nfev,
            weight_tags=("calc-modulus", "calc-unit") if batch_fast else None,
            seed_limit=2 if batch_fast else None,
        )
    if len(peaks) == 1:
        p1 = peaks[0]
        q1 = max(1.0 / (max(p1["R"], 1e-4) * (2 * math.pi / max(p1["tau"], 1e-6)) ** p1["n"]), 1e-12)
        drt_x0 = x0.copy()
        drt_x0[0] = max(guide.rs, 1e-6)
        drt_x0[1] = q1
        drt_x0[2] = p1["n"]
        drt_x0[3] = max(p1["R"] * 0.1, 1e-4)
        drt_x0[4] = q1
        drt_x0[5] = p1["n"]
        drt_x0[6] = max(p1["R"] * 0.9, 1e-4)
        seeds = [drt_x0] + seeds[:2]
        return _solve_zview_cnls(
            freq,
            z_exp,
            _zview_double_model,
            lower,
            upper,
            seeds,
            param_count=len(x0),
            max_nfev=22000 if batch_fast else FIT_CONFIG.zview_double_retry_max_nfev,
            weight_tags=("calc-modulus", "calc-unit") if batch_fast else None,
            seed_limit=2 if batch_fast else None,
        )
    return result


def _zview_full_model(freq_hz: np.ndarray, params: np.ndarray) -> np.ndarray:
    r1, cpe_t, cpe_p, r2, wo_r, wo_t, wo_p = params
    omega = 2.0 * np.pi * freq_hz
    z_cpe = 1.0 / (cpe_t * (1j * omega) ** cpe_p)
    z_wo = _zview_warburg_open(omega, wo_r, wo_t, wo_p)
    z_branch = np.full(freq_hz.shape, r2, dtype=complex) + z_wo
    return r1 + 1.0 / (1.0 / z_cpe + 1.0 / z_branch)


def _zview_double_model(freq_hz: np.ndarray, params: np.ndarray) -> np.ndarray:
    rs, q1, n1, r2, q2, n2, r3, wo_r, wo_t, wo_p = params
    omega = 2.0 * np.pi * freq_hz
    z_cpe1 = 1.0 / (q1 * (1j * omega) ** n1)
    z_cpe2 = 1.0 / (q2 * (1j * omega) ** n2)
    z_arc1 = 1.0 / (1.0 / np.full(freq_hz.shape, r2, dtype=complex) + 1.0 / z_cpe1)
    z_wo = _zview_warburg_open(omega, wo_r, wo_t, wo_p)
    z_arc2 = 1.0 / (1.0 / z_cpe2 + 1.0 / (np.full(freq_hz.shape, r3, dtype=complex) + z_wo))
    return rs + z_arc1 + z_arc2


def _zview_warburg_open(omega: np.ndarray, wo_r: float, wo_t: float, wo_p: float) -> np.ndarray:
    x = (1j * omega * wo_t) ** wo_p
    abs_x = np.abs(x)
    tanh_x = np.ones_like(x, dtype=complex)
    finite_mask = abs_x <= 50.0
    tanh_x[finite_mask] = np.tanh(x[finite_mask])
    small = abs_x < 1e-8
    denom = x * tanh_x
    if np.any(small):
        # tanh(x) = x - x^3/3 + O(x^5), so x*tanh(x) = x^2 - x^4/3 + O(x^6)
        denom = np.where(small, x * x * (1.0 - (x * x) / 3.0), denom)
    denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0.0j, denom)
    return wo_r / denom


def _zview_statistics(result, n_points: int) -> dict[str, float]:
    residual = result.fun
    rss = float(np.sum(np.square(residual)))
    n_obs = max(2 * int(n_points), 1)
    n_params = int(result.x.size)
    dof = max(n_obs - n_params, 1)
    aic = n_obs * math.log(max(rss / n_obs, 1e-300)) + 2 * n_params
    aicc = aic + 2 * n_params * (n_params + 1) / max(n_obs - n_params - 1, 1)
    return {
        "rss": rss,
        "chi2_reduced": rss / dof,
        "aic": aic,
        "aicc": aicc,
        "bic": n_obs * math.log(max(rss / n_obs, 1e-300)) + n_params * math.log(n_obs),
        "pseudo_chi2": rss / dof,
        "n_obs": float(n_obs),
        "dof": float(dof),
        "method": "least_squares",
    }


def _zview_residual(
    freq: np.ndarray,
    z_exp: np.ndarray,
    params: np.ndarray,
    model_fn,
    weight_tag: str,
) -> np.ndarray:
    z_model = model_fn(freq, params)
    diff = z_model - z_exp
    floor = _adaptive_weight_floor(z_exp)
    if weight_tag == "calc-unit":
        return np.concatenate([diff.real, diff.imag])
    if weight_tag == "calc-proportional":
        w_real = np.maximum(np.abs(z_exp.real), floor)
        w_imag = np.maximum(np.abs(z_exp.imag), floor)
        return np.concatenate([diff.real / w_real, diff.imag / w_imag])
    if weight_tag == "data-special":
        sigma = _local_noise_estimate(z_exp)
        return np.concatenate([diff.real / sigma, diff.imag / sigma])
    if weight_tag == "calc-modulus-zview":
        weight = np.maximum(np.abs(z_model), floor)
        return np.concatenate([diff.real / weight, diff.imag / weight])
    weight = np.maximum(np.abs(z_exp), floor)
    return np.concatenate([diff.real / weight, diff.imag / weight])


def _cnls_selection_score(result: OptimizeResult, n_params: int) -> float:
    residual = result.fun
    n_obs = max(int(residual.size), 1)
    rss = float(np.sum(np.square(residual)))
    aic = n_obs * math.log(max(rss / n_obs, 1e-300)) + 2 * n_params
    score = aic + 2 * n_params * (n_params + 1) / max(n_obs - n_params - 1, 1)
    signs = np.sign(residual)
    nonzero = signs[signs != 0]
    if nonzero.size >= 4:
        runs = 1 + int(np.sum(nonzero[1:] != nonzero[:-1]))
        pos = int(np.sum(nonzero > 0))
        neg = int(np.sum(nonzero < 0))
        expected = 1.0 + 2.0 * pos * neg / max(nonzero.size, 1)
        if expected > 0 and runs < expected * 0.6:
            score += 5.0
    jac = getattr(result, "jac", None)
    if jac is not None:
        try:
            cond = float(np.linalg.cond(jac.T @ jac))
        except Exception:
            cond = float("inf")
        if np.isfinite(cond):
            score += 0.02 * math.log10(max(cond, 1.0))
        else:
            score += 1e6
    return score


def _cnls_diagnostics(result: OptimizeResult, n_points: int, names: list[str]) -> tuple[dict[str, float], list[str]]:
    jac = getattr(result, "jac", None)
    if jac is None:
        return {}, []
    try:
        u, s, vt = np.linalg.svd(jac, full_matrices=False)
        cond = float(np.inf if s.size == 0 or s[-1] <= 0 else s[0] / max(s[-1], 1e-30))
        rss = float(np.sum(np.square(result.fun)))
        n_obs = max(2 * int(n_points), 1)
        dof = max(n_obs - len(names), 1)
        sigma2 = rss / dof
        inv_sq = np.where(s > s[0] * 1e-12, 1.0 / (s * s), 0.0)
        cov = sigma2 * (vt.T * inv_sq) @ vt
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        return {}, []
    stats = {"jtj_cond": cond}
    warnings: list[str] = []
    if not np.isfinite(cond) or cond > 1e12:
        warnings.append("ill_conditioned")
    corr_max = 0.0
    try:
        denom = np.outer(stderr, stderr)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        if corr.size:
            off_diag = corr - np.diag(np.diag(corr))
            corr_max = float(np.max(np.abs(off_diag)))
            stats["correlation_matrix_max"] = corr_max
            if corr_max > FIT_CONFIG.corr_warn_threshold:
                warnings.append("strong_parameter_correlation")
    except Exception:
        logger.debug("Correlation diagnostics failed", exc_info=True)
    for name, value, err in zip(names, result.x, stderr):
        rel = float(abs(err) / max(abs(value), 1e-30) * 100.0)
        stats[f"{name}_stderr_pct"] = rel
        stats[f"{name}_ci95_low"] = float(value - 1.96 * err)
        stats[f"{name}_ci95_high"] = float(value + 1.96 * err)
        if not np.isfinite(rel) or rel > 50.0:
            warnings.append(name)
    return stats, warnings


def _zview_boundary_hits(
    values: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], names: list[str]
) -> list[str]:
    lower, upper = bounds
    hits: list[str] = []
    for name, value, lo, hi in zip(names, values, lower, upper):
        if abs(value - lo) <= max(abs(lo), 1.0) * 1e-5:
            hits.append(f"{name}=lower")
        if abs(value - hi) <= max(abs(hi), 1.0) * 1e-5:
            hits.append(f"{name}=upper")
    return hits


_SINGLE_PARAM_NAMES = ["R1", "CPE_T", "CPE_P", "R2", "Wo_R", "Wo_T", "Wo_P"]
_DOUBLE_PARAM_NAMES = ["Rs", "Q1", "n1", "R2", "Q2", "n2", "R3", "Wo_R", "Wo_T", "Wo_P"]


def _confidence_intervals_from_stats(stats: dict[str, float], names: list[str]) -> dict[str, tuple[float, float]]:
    intervals: dict[str, tuple[float, float]] = {}
    for name in names:
        low = stats.get(f"{name}_ci95_low")
        high = stats.get(f"{name}_ci95_high")
        if low is None or high is None:
            continue
        if np.isfinite(float(low)) and np.isfinite(float(high)):
            intervals[name] = (float(low), float(high))
    return intervals


def _corr_max_from_stats(stats: dict[str, float]) -> float:
    value = stats.get("correlation_matrix_max", 0.0)
    return float(value) if np.isfinite(float(value)) else 0.0


def _fit_score(result: object) -> float:
    score = math.log10(max(float(result.pseudo_chisqr), 1e-300))
    score += 0.35 * len(_boundary_hits(result.circuit))
    score += 0.1 * len(_parameter_warnings(result))
    return score


def _statistics_from_fit(result: object) -> dict[str, float]:
    stats_df = result.to_statistics_dataframe()
    stats = dict(zip(stats_df["Label"], stats_df["Value"]))
    return {
        "rss": float(result.minimizer_result.chisqr),
        "chi2_reduced": float(result.minimizer_result.redchi),
        "aic": float(result.minimizer_result.aic),
        "bic": float(result.minimizer_result.bic),
        "pseudo_chi2": float(result.pseudo_chisqr),
        "n_obs": float(result.minimizer_result.ndata),
        "dof": float(result.minimizer_result.nfree),
        "method": str(stats.get("Method", "")),
        "weight": str(stats.get("Weight", "")),
    }


def _extract_parameters(template_key: str, circuit: object) -> dict[str, float]:
    return {}


def _boundary_hits(circuit) -> list[str]:
    hits: list[str] = []
    for idx, element in enumerate(circuit.get_elements(), start=1):
        values = element.get_values()
        lowers = element.get_lower_limits()
        uppers = element.get_upper_limits()
        for name, value in values.items():
            lower = lowers.get(name, -np.inf)
            upper = uppers.get(name, np.inf)
            if np.isfinite(lower) and abs(value - lower) <= max(abs(lower), 1.0) * 1e-4:
                hits.append(f"{type(element).__name__}{idx}.{name}=lower")
            if np.isfinite(upper) and abs(value - upper) <= max(abs(upper), 1.0) * 1e-4:
                hits.append(f"{type(element).__name__}{idx}.{name}=upper")
    return hits


def _parameter_warnings(result) -> list[str]:
    warnings: list[str] = []
    params_df = result.to_parameters_dataframe()
    for _, row in params_df.iterrows():
        fixed = str(row["Fixed"]).lower() == "yes"
        std_err = row["Std. err. (%)"]
        if not fixed and (not np.isfinite(std_err) or float(std_err) > 50.0):
            warnings.append(f"{row['Element']}.{row['Parameter']}")
    return warnings


def _peak_frequencies(freq: np.ndarray, y: np.ndarray, target_count: int) -> list[float]:
    if y.size == 0:
        return [1.0] * target_count
    idx = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if idx.size == 0:
        peak = float(freq[int(np.argmax(y))])
        return [peak] * target_count
    ranked = idx[np.argsort(y[idx])[::-1]]
    selected = [float(freq[i]) for i in ranked[:target_count]]
    while len(selected) < target_count:
        selected.append(selected[-1])
    return sorted(selected, reverse=True)


def _primary_parameter_aliases(template_key: str, params: dict[str, float]) -> dict[str, float]:
    if template_key == "zview_double_rq_qrwo":
        return {"Rs": params["Rs"], "R2": params["R2"], "R3": params["R3"]}
    return {}


def get_circuit_templates() -> dict[str, CircuitTemplate]:
    return _get_circuit_templates()


# ---------------------------------------------------------------------------
#  Fallback helpers – produce a degraded FitOutcome from intermediate arc fits
# ---------------------------------------------------------------------------

def _arc_fallback_outcome(
    template: CircuitTemplate,
    spectrum: SpectrumData,
    arc_fit,
    detection: SegmentDetection,
    point_mask: np.ndarray,
) -> FitOutcome:
    """Build a *warn*-level outcome using the R(QR) arc fit when the global
    R(QRWo) CNLS did not converge.  The predicted curve only covers the arc
    region, so the low-frequency tail will not be drawn."""
    arc_rs, arc_rs_err, arc_rp, arc_rp_err = _arc_region_resistances(arc_fit)
    arc_elements = arc_fit.circuit.get_elements()
    predicted = arc_fit.circuit.get_impedances(spectrum.freq_hz.astype(float))

    parameters = {
        "Rs": float(arc_rs),
        "Rct": float(arc_rp),
        "CPE_T": float(arc_elements[1].get_values()["Y"]),
        "CPE_P": float(arc_elements[1].get_values()["n"]),
        "split_freq_hz": float(spectrum.freq_hz[detection.split_indices[0]]) if detection.split_indices else float("nan"),
    }
    statistics = {
        "pseudo_chi2": float(arc_fit.pseudo_chisqr),
        "Rs_stderr_pct": arc_rs_err,
        "Rct_stderr_pct": arc_rp_err,
    }
    status = "warn"
    if _is_high_error(arc_rs_err) or _is_high_error(arc_rp_err):
        status = "warn"

    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status=status,
        message=f"全局 R(QRWo) 未收敛，降级为半圆 R(QR) 结果；Rs={arc_rs:.6g}, Rct={arc_rp:.6g}",
        parameters=parameters,
        statistics=statistics,
        predicted_real_ohm=predicted.real,
        predicted_imag_ohm=predicted.imag,
        masked_points=int((~point_mask).sum()),
        fallback_from="zview_segmented_rq_rwo",
    )


def _double_arc_fallback_outcome(
    template: CircuitTemplate,
    spectrum: SpectrumData,
    arc1_fit,
    arc2_fit,
    detection: SegmentDetection,
    point_mask: np.ndarray,
) -> FitOutcome:
    """Build a degraded outcome from whatever arc fits succeeded."""
    parameters: dict[str, float] = {}
    statistics: dict[str, float] = {}
    parts: list[str] = []

    if arc1_fit is not None:
        rs1, rs1_err, rp1, rp1_err = _arc_region_resistances(arc1_fit)
        parameters["Rs"] = float(rs1)
        parameters["Rsei"] = float(rp1)
        statistics["Rs_stderr_pct"] = rs1_err
        statistics["Rsei_stderr_pct"] = rp1_err
        statistics["segment_arc1_pseudo_chi2"] = float(arc1_fit.pseudo_chisqr)
        parts.append(f"Rs={rs1:.6g}, Rsei={rp1:.6g}")

    if arc2_fit is not None:
        rs2, rs2_err, rp2, rp2_err = _arc_region_resistances(arc2_fit)
        parameters["Rct"] = float(rp2)
        statistics["Rct_stderr_pct"] = rp2_err
        statistics["segment_arc2_pseudo_chi2"] = float(arc2_fit.pseudo_chisqr)
        parts.append(f"Rct={rp2:.6g}")

    msg = "全局 R(QR)(Q(RWo)) 未收敛，降级为各半圆 R(QR) 结果"
    if parts:
        msg += "；" + ", ".join(parts)

    # Try to build predicted curve from available arc fits
    predicted_real = None
    predicted_imag = None
    best_arc = arc1_fit or arc2_fit
    if best_arc is not None:
        try:
            predicted = best_arc.circuit.get_impedances(spectrum.freq_hz.astype(float))
            predicted_real = predicted.real
            predicted_imag = predicted.imag
        except Exception:
            logger.debug("Predicted curve generation failed", exc_info=True)

    return FitOutcome(
        model_key=template.key,
        model_label=template.label,
        status="warn",
        message=msg,
        parameters=parameters,
        statistics=statistics,
        predicted_real_ohm=predicted_real,
        predicted_imag_ohm=predicted_imag,
        masked_points=int((~point_mask).sum()),
        fallback_from="zview_double_rq_qrwo",
    )


def _attach_diagnosis(
    outcome: FitOutcome,
    spectrum: SpectrumData,
    template_key: str,
    segment_hint: Optional[SegmentDetection],
) -> FitOutcome:
    """Attach a structured diagnosis when the outcome is not *ok*."""
    if outcome.status == "ok":
        return outcome
    diag = diagnose_fit_failure(spectrum, outcome, template_key, segment_hint)
    # Append diagnosis to message
    diag_text = f"[诊断: {diag.explanation}]"
    if diag.suggestions:
        diag_text += " 建议: " + "; ".join(diag.suggestions[:3])
    new_message = outcome.message + " " + diag_text if outcome.message else diag_text
    return replace(
        outcome,
        message=new_message,
        diagnosis_type=diag.failure_type,
        diagnosis_severity=diag.severity,
        diagnosis_explanation=diag.explanation,
        diagnosis_suggestions=list(diag.suggestions),
    )
