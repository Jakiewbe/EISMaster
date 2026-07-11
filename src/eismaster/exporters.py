from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from eismaster.models import BatchSummary, FitOutcome, QualityReport, SpectrumData

TXT_ENCODING = "utf-8"
DEFAULT_DRT_LOGTAU_BREAKS = (-3.0, 0.0)
DRT_SHEET_ORDER = (
    "drt_line_plot",
    "drt_cloud_density",
    "drt_quant_area",
)


def export_spectrum_bundle(
    output_path: str | Path,
    spectrum: SpectrumData,
    *,
    fit: FitOutcome | None = None,
    quality: QualityReport | None = None,
    fmt: str = "txt",
    drt_source_dir: str | Path | None = None,
) -> dict[str, Path]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spectra = [spectrum]
    fits = [fit]

    raw_matrix = build_raw_plot_matrix(spectra)
    rs_rct_table = build_rs_rct_table([(spectrum, fit)])
    fit_matrix = build_fit_overlay_matrix(spectra, fits)
    drt_matrices = build_drt_matrices(spectra, drt_source_dir)
    fit_report = build_fit_report_table(spectrum, quality, fit) if fit is not None and quality is not None else None

    return _write_export_bundle(target, raw_matrix, rs_rct_table, fit_matrix, drt_matrices, fit_report, fmt)


def export_batch_summary(
    output_path: str | Path,
    summary: BatchSummary,
    *,
    fmt: str = "txt",
    drt_source_dir: str | Path | None = None,
) -> dict[str, Path]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spectra = [item.spectrum for item in summary.items]
    fits = [item.fit for item in summary.items]

    raw_matrix = build_raw_plot_matrix(spectra)
    rs_rct_table = build_rs_rct_table([(item.spectrum, item.fit) for item in summary.items])
    fit_matrix = build_fit_overlay_matrix(spectra, fits)
    drt_matrices = build_drt_matrices(spectra, drt_source_dir)

    return _write_export_bundle(target, raw_matrix, rs_rct_table, fit_matrix, drt_matrices, None, fmt)


def _write_export_bundle(
    target: Path,
    raw_matrix: pd.DataFrame,
    rs_rct_table: pd.DataFrame,
    fit_matrix: pd.DataFrame,
    drt_matrices: dict[str, pd.DataFrame],
    fit_report: pd.DataFrame | None,
    fmt: str,
) -> dict[str, Path]:
    if fmt == "xlsx":
        workbook = target if target.suffix.lower() == ".xlsx" else target.with_suffix(".xlsx")
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            raw_matrix.to_excel(writer, sheet_name="raw_plot", index=False, header=False)
            rs_rct_table.to_excel(writer, sheet_name="rs_rct", index=False)
            fit_matrix.to_excel(writer, sheet_name="fit_overlay", index=False, header=False)
            for sheet_name in DRT_SHEET_ORDER:
                if sheet_name not in drt_matrices:
                    continue
                drt_matrix = drt_matrices[sheet_name]
                drt_matrix.to_excel(writer, sheet_name=sheet_name, index=False, header=(sheet_name == "drt_quant_area"))
            if fit_report is not None:
                fit_report.to_excel(writer, sheet_name="fit_report", index=False)
        return {"workbook": workbook}

    suffix = ".csv" if fmt == "csv" else ".txt"
    base = target.with_suffix("")
    paths = {
        "raw_plot": _write_matrix(raw_matrix, base.with_name(base.name + "_raw_plot" + suffix), fmt),
        "rs_rct": _write_frame(rs_rct_table, base.with_name(base.name + "_rs_rct" + suffix), fmt),
        "fit_overlay": _write_matrix(fit_matrix, base.with_name(base.name + "_fit_overlay" + suffix), fmt),
    }
    for key in DRT_SHEET_ORDER:
        if key not in drt_matrices:
            continue
        drt_matrix = drt_matrices[key]
        paths[key] = _write_matrix(drt_matrix, base.with_name(base.name + "_" + key + suffix), fmt)
    if fit_report is not None:
        paths["fit_report"] = _write_frame(fit_report, base.with_name(base.name + "_fit_report" + suffix), fmt)
    return paths


def export_fit_results(output_dir: str | Path, spectrum: SpectrumData, quality: QualityReport, fit: FitOutcome) -> dict[str, Path]:
    return export_spectrum_bundle(output_dir, spectrum, fit=fit, quality=quality, fmt="txt")


def write_drt_only_export(
    output_path: str | Path,
    spectra: list[SpectrumData],
    drt_source_dir: str | Path,
    *,
    fmt: str = "xlsx",
    logtau_breaks: tuple[float, ...] | None = None,
    line_x_axis: str = "logtau",
) -> dict[str, Path]:
    target = Path(output_path)
    matrices = build_drt_matrices(spectra, drt_source_dir, logtau_breaks=logtau_breaks, line_x_axis=line_x_axis)
    if not matrices:
        return {}
    if fmt == "xlsx":
        workbook = target if target.suffix.lower() == ".xlsx" else target.with_suffix(".xlsx")
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for sheet_name in DRT_SHEET_ORDER:
                matrix = matrices[sheet_name]
                matrix.to_excel(writer, sheet_name=sheet_name, index=False, header=(sheet_name == "drt_quant_area"))
        return {"drt": workbook}
    suffix = ".csv" if fmt == "csv" else ".txt"
    base = target.with_suffix("")
    return {
        key: _write_matrix(matrices[key], base.with_name(base.name + "_" + key).with_suffix(suffix), fmt)
        for key in DRT_SHEET_ORDER
    }


def build_raw_plot_matrix(spectra: list[SpectrumData]) -> pd.DataFrame:
    max_len = max((s.n_points for s in spectra), default=0)
    rows: list[list[object]] = []
    row1: list[object] = []
    row2: list[object] = []
    for spectrum in spectra:
        mark = _export_label_for_spectrum(spectrum)
        row1.extend([mark, ""])
        row2.extend(["z_real", "imag_pos"])
    rows.append(row1)
    rows.append(row2)
    for i in range(max_len):
        row: list[object] = []
        for spectrum in spectra:
            row.append(_safe_get(spectrum.z_real_ohm, i))
            row.append(_safe_get(spectrum.minus_z_imag_ohm, i))
        rows.append(row)
    return pd.DataFrame(rows)


def build_fit_overlay_matrix(spectra: list[SpectrumData], fits: list[FitOutcome | None]) -> pd.DataFrame:
    max_len = max((s.n_points for s in spectra), default=0)
    rows: list[list[object]] = []
    row1: list[object] = []
    row2: list[object] = []
    for spectrum in spectra:
        mark = _export_label_for_spectrum(spectrum)
        row1.extend([mark, "", "", ""])
        row2.extend(["z_real_exp", "imag_exp_pos", "z_real_fit", "imag_fit_pos"])
    rows.append(row1)
    rows.append(row2)
    for i in range(max_len):
        row: list[object] = []
        for spectrum, fit in zip(spectra, fits, strict=True):
            row.append(_safe_get(spectrum.z_real_ohm, i))
            row.append(_safe_get(spectrum.minus_z_imag_ohm, i))
            if fit is not None and fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
                row.append(_safe_get(fit.predicted_real_ohm, i))
                row.append(_safe_get(-fit.predicted_imag_ohm, i))
            else:
                row.extend([np.nan, np.nan])
        rows.append(row)
    return pd.DataFrame(rows)


def build_rs_rct_table(pairs: list[tuple[SpectrumData, FitOutcome | None]]) -> pd.DataFrame:
    rows = []
    for spectrum, fit in pairs:
        rows.append(
            {
                "label": _export_label_for_spectrum(spectrum),
                "file": spectrum.metadata.file_path.stem,
                "Rs": np.nan if fit is None else fit.parameters.get("Rs", np.nan),
                "Rct": np.nan if fit is None else fit.parameters.get("Rct", np.nan),
                "Rsei": np.nan if fit is None else fit.parameters.get("Rsei", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_fit_report_table(spectrum: SpectrumData, quality: QualityReport, fit: FitOutcome) -> pd.DataFrame:
    rows = [
        {"section": "meta", "key": "generated", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"section": "meta", "key": "file", "value": str(spectrum.metadata.file_path)},
        {"section": "meta", "key": "technique", "value": spectrum.metadata.technique},
        {"section": "meta", "key": "instrument", "value": spectrum.metadata.instrument_model},
        {"section": "fit", "key": "model", "value": fit.model_label},
        {"section": "fit", "key": "status", "value": fit.status},
        {"section": "fit", "key": "message", "value": fit.message},
    ]
    if fit.diagnosis_type:
        rows.extend(
            [
                {"section": "fit", "key": "diagnosis_type", "value": fit.diagnosis_type},
                {"section": "fit", "key": "diagnosis_severity", "value": fit.diagnosis_severity},
                {"section": "fit", "key": "diagnosis_explanation", "value": fit.diagnosis_explanation},
            ]
        )
        rows.extend(
            {"section": "fit", "key": f"diagnosis_suggestion_{i+1}", "value": text}
            for i, text in enumerate(fit.diagnosis_suggestions[:3])
        )
    rows.extend({"section": "quality", "key": f"issue_{i+1}", "value": line} for i, line in enumerate(quality.summary_lines()))
    rows.extend({"section": "parameter", "key": key, "value": value} for key, value in fit.parameters.items())
    rows.extend({"section": "stat", "key": key, "value": value} for key, value in fit.statistics.items())
    return pd.DataFrame(rows)


def build_drt_matrix(spectra: list[SpectrumData], drt_source_dir: str | Path | None) -> pd.DataFrame | None:
    return build_drt_matrices(spectra, drt_source_dir).get("drt_cloud_density")


def build_drt_matrices(
    spectra: list[SpectrumData],
    drt_source_dir: str | Path | None,
    *,
    logtau_breaks: tuple[float, ...] | None = None,
    line_x_axis: str = "logtau",
) -> dict[str, pd.DataFrame]:
    if line_x_axis not in {"logtau", "tau"}:
        raise ValueError(f"Unsupported line_x_axis: {line_x_axis!r}; expected 'logtau' or 'tau'.")
    if drt_source_dir is None:
        return {}
    source = Path(drt_source_dir)
    if not source.exists():
        return {}

    parsed: list[tuple[str, list[float], list[float], list[float]]] = []
    common_logtau: list[float] | None = None

    for spectrum in spectra:
        file_path = source / f"{spectrum.metadata.file_path.stem}_DRT.txt"
        if not file_path.exists():
            continue
        tau, gamma = _parse_drt_file(file_path)
        logtau = [float(np.log10(t)) if t > 0 else np.nan for t in tau]
        logtau, gamma, area = _drt_values_by_logtau(logtau, gamma)
        if common_logtau is None:
            common_logtau = logtau
        elif logtau != common_logtau:
            gamma = _interpolate_to_logtau(logtau, gamma, common_logtau)
            area = _interpolate_to_logtau(logtau, area, common_logtau)
            logtau = common_logtau
        parsed.append((_export_label_for_spectrum(spectrum), logtau, gamma, area))

    if not parsed or common_logtau is None:
        return {}

    return {
        "drt_line_plot": _build_drt_line_plot_matrix(parsed, line_x_axis),
        "drt_cloud_density": _build_drt_cloud_matrix(parsed, common_logtau, "gamma_tau", value_index=2),
        "drt_quant_area": _build_drt_quant_area_trend_matrix(parsed, logtau_breaks),
    }


def _build_drt_line_plot_matrix(parsed: list[tuple[str, list[float], list[float], list[float]]], line_x_axis: str) -> pd.DataFrame:
    first_logtau = parsed[0][1] if parsed else []
    if line_x_axis == "tau":
        first_x = [10.0 ** value if np.isfinite(value) else np.nan for value in first_logtau]
        x_label = "tau/s"
    else:
        first_x = first_logtau
        x_label = "logtau"
    max_len = len(first_logtau)
    rows: list[list[object]] = []
    row1: list[object] = [""]
    row2: list[object] = [x_label]
    for mark, _, _, _ in parsed:
        row1.append(mark)
        row2.append("gamma_tau")

    rows.append(row1)
    rows.append(row2)

    for i in range(max_len):
        row: list[object] = [first_x[i]]
        for _, _logtau, gamma, _ in parsed:
            row.append(gamma[i] if i < len(gamma) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_drt_cloud_matrix(
    parsed: list[tuple[str, list[float], list[float], list[float]]],
    common_logtau: list[float],
    value_label: str,
    *,
    value_index: int,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    row1: list[object] = [""]
    row2: list[object] = ["logtau"]
    for mark, _, _, _ in parsed:
        row1.append(mark)
        row2.append(value_label)

    rows.append(row1)
    rows.append(row2)

    for i, logtau_value in enumerate(common_logtau):
        row: list[object] = []
        row.append(logtau_value)
        for item in parsed:
            values = item[value_index]
            row.append(values[i] if i < len(values) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_drt_quant_area_trend_matrix(
    parsed: list[tuple[str, list[float], list[float], list[float]]],
    logtau_breaks: tuple[float, ...] | None,
) -> pd.DataFrame:
    regions = _drt_logtau_regions(logtau_breaks)
    rows: list[dict[str, object]] = []
    for mark, logtau, _, area in parsed:
        x = np.asarray(logtau, dtype=float)
        y = np.asarray(area, dtype=float)
        row: dict[str, object] = {
            "sample": mark,
            "time_min": _time_minutes_from_label(mark),
            "total_area": float(np.nansum(y)),
        }
        for label, lo, hi in regions:
            mask = (x >= lo) & (x < hi)
            row[label] = float(np.nansum(y[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def _drt_logtau_regions(logtau_breaks: tuple[float, ...] | None) -> list[tuple[str, float, float]]:
    breaks = tuple(sorted(set(DEFAULT_DRT_LOGTAU_BREAKS if logtau_breaks is None else logtau_breaks)))
    if not breaks:
        return [("logtau_all", -np.inf, np.inf)]

    regions: list[tuple[str, float, float]] = []
    regions.append((f"logtau_lt_{_format_logtau_bound(breaks[0])}", -np.inf, breaks[0]))
    for lo, hi in zip(breaks, breaks[1:], strict=False):
        regions.append((f"logtau_{_format_logtau_bound(lo)}_to_{_format_logtau_bound(hi)}", lo, hi))
    regions.append((f"logtau_ge_{_format_logtau_bound(breaks[-1])}", breaks[-1], np.inf))
    return regions


def _format_logtau_bound(value: float) -> str:
    text = f"{value:g}".replace("-", "neg").replace(".", "p")
    return text


def _export_label_for_spectrum(spectrum: SpectrumData) -> str:
    return _export_label_from_stem(spectrum.metadata.file_path.stem)


def _export_label_from_stem(stem: str) -> str:
    parts = [part for part in re.split(r"[_\-\s]+", stem) if part]
    if not parts:
        return stem

    preferred = [
        part
        for part in parts
        if re.fullmatch(r"(?i)(ocv|t\d+[smhd]?|e\d+|c\d+|soc\d+|rest|charge|discharge|before|after|init|mid|end)", part)
    ]
    if preferred:
        return preferred[-1]

    informative = [
        part
        for part in parts
        if re.search(r"[A-Za-z\u4e00-\u9fff]", part) and not re.fullmatch(r"\d+(?:\.\d+)?", part) and len(part) <= 16
    ]
    if informative:
        return informative[-1]

    return parts[-1]


def _time_minutes_from_label(label: str) -> float:
    match = re.search(r"(?i)t(\d+(?:\.\d+)?)([smhd]?)", label)
    if match is None:
        return np.nan
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "s":
        return value / 60.0
    if unit == "h":
        return value * 60.0
    if unit == "d":
        return value * 1440.0
    return value


def _parse_drt_file(path: Path) -> tuple[list[float], list[float]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = None
    use_freq = True
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower.startswith("freq"):
            start_idx = i + 1
            use_freq = True
            break
        if lower.startswith("tau"):
            start_idx = i + 1
            use_freq = False
            break
    if start_idx is None:
        return _parse_peak_drt_file(lines)
    xs: list[float] = []
    ys: list[float] = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    tau = [1.0 / (2.0 * np.pi * f) if f > 0 else np.nan for f in xs] if use_freq else xs
    return tau, ys


def _parse_peak_drt_file(lines: list[str]) -> tuple[list[float], list[float]]:
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("peak number"):
            start_idx = i + 1
            break
    if start_idx is None:
        return [], []

    peaks: list[tuple[float, float, float]] = []
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            height = float(parts[1])
            mu = float(parts[2])
            sigma = abs(float(parts[3]))
        except ValueError:
            continue
        if sigma > 0:
            peaks.append((height, mu, sigma))
    if not peaks:
        return [], []

    start = min(mu - 4.0 * sigma for _, mu, sigma in peaks)
    stop = max(mu + 4.0 * sigma for _, mu, sigma in peaks)
    logtau_ln = np.linspace(start, stop, 160)
    gamma = np.zeros_like(logtau_ln)
    for height, mu, sigma in peaks:
        gamma += height * np.exp(-0.5 * ((logtau_ln - mu) / sigma) ** 2)
    tau = np.exp(logtau_ln)
    return tau.tolist(), gamma.tolist()


def _drt_values_by_logtau(logtau: list[float], gamma: list[float]) -> tuple[list[float], list[float], list[float]]:
    x = np.asarray(logtau, dtype=float)
    y = np.asarray(gamma, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return [], [], []

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size == 1:
        return x.tolist(), y.tolist(), y.tolist()

    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = (x[:-1] + x[1:]) / 2.0
    edges[0] = x[0] - (x[1] - x[0]) / 2.0
    edges[-1] = x[-1] + (x[-1] - x[-2]) / 2.0
    dln_tau = np.diff(edges) * np.log(10.0)
    return x.tolist(), y.tolist(), (y * dln_tau).tolist()


def _interpolate_to_logtau(logtau: list[float], values: list[float], target: list[float]) -> list[float]:
    x = np.asarray(logtau, dtype=float)
    y = np.asarray(values, dtype=float)
    target_x = np.asarray(target, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return [np.nan for _ in target]
    return np.interp(target_x, x[finite], y[finite], left=np.nan, right=np.nan).tolist()


def _safe_get(values: np.ndarray, index: int) -> float:
    if index >= len(values):
        return np.nan
    return values[index]


def _write_matrix(frame: pd.DataFrame, path: Path, fmt: str) -> Path:
    if fmt == "csv":
        frame.to_csv(path, index=False, header=False, encoding=TXT_ENCODING)
    else:
        frame.to_csv(path, index=False, header=False, sep="\t", encoding=TXT_ENCODING)
    return path


def _write_frame(frame: pd.DataFrame, path: Path, fmt: str) -> Path:
    if fmt == "csv":
        frame.to_csv(path, index=False, encoding=TXT_ENCODING)
    else:
        frame.to_csv(path, index=False, sep="\t", encoding=TXT_ENCODING)
    return path
