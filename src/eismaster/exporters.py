from __future__ import annotations
from typing import Optional
import typing







from datetime import datetime
from pathlib import Path
import re


import numpy as np
import pandas as pd

from eismaster.models import BatchSummary, FitOutcome, QualityReport, SpectrumData

TXT_ENCODING = "utf-8"


def export_spectrum_bundle(
    output_path: str | Path,
    spectrum: SpectrumData,
    *,
    fit: Optional[FitOutcome] = None,
    quality: Optional[QualityReport] = None,
    fmt: str = "txt",
    drt_source_dir: str | Optional[Path] = None,
) -> dict[str, Path]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spectra = [spectrum]
    fits = [fit]

    raw_matrix = build_raw_plot_matrix(spectra)
    rs_rct_table = build_rs_rct_table([(spectrum, fit)])
    fit_matrix = build_fit_overlay_matrix(spectra, fits)
    drt_matrix = build_drt_matrix(spectra, drt_source_dir)

    if fmt == "xlsx":
        workbook = target if target.suffix.lower() == ".xlsx" else target.with_suffix(".xlsx")
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            raw_matrix.to_excel(writer, sheet_name="raw_plot", index=False, header=False)
            rs_rct_table.to_excel(writer, sheet_name="rs_rct", index=False)
            fit_matrix.to_excel(writer, sheet_name="fit_overlay", index=False, header=False)
            if drt_matrix is not None:
                drt_matrix.to_excel(writer, sheet_name="drt", index=False, header=False)
            if fit is not None and quality is not None:
                build_fit_report_table(spectrum, quality, fit).to_excel(writer, sheet_name="fit_report", index=False)
        return {"workbook": workbook}

    suffix = ".csv" if fmt == "csv" else ".txt"
    base = target.with_suffix("")
    paths = {
        "raw_plot": _write_matrix(raw_matrix, base.with_name(base.name + "_raw_plot" + suffix), fmt),
        "rs_rct": _write_frame(rs_rct_table, base.with_name(base.name + "_rs_rct" + suffix), fmt),
        "fit_overlay": _write_matrix(fit_matrix, base.with_name(base.name + "_fit_overlay" + suffix), fmt),
    }
    if drt_matrix is not None:
        paths["drt"] = _write_matrix(drt_matrix, base.with_name(base.name + "_drt" + suffix), fmt)
    if fit is not None and quality is not None:
        report = build_fit_report_table(spectrum, quality, fit)
        paths["fit_report"] = _write_frame(report, base.with_name(base.name + "_fit_report" + suffix), fmt)
    return paths


def export_batch_summary(
    output_path: str | Path,
    summary: BatchSummary,
    *,
    fmt: str = "txt",
    drt_source_dir: str | Optional[Path] = None,
) -> dict[str, Path]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    spectra = [item.spectrum for item in summary.items]
    fits = [item.fit for item in summary.items]

    raw_matrix = build_raw_plot_matrix(spectra)
    rs_rct_table = build_rs_rct_table([(item.spectrum, item.fit) for item in summary.items])
    fit_matrix = build_fit_overlay_matrix(spectra, fits)
    drt_matrix = build_drt_matrix(spectra, drt_source_dir)

    if fmt == "xlsx":
        workbook = target if target.suffix.lower() == ".xlsx" else target.with_suffix(".xlsx")
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            raw_matrix.to_excel(writer, sheet_name="raw_plot", index=False, header=False)
            rs_rct_table.to_excel(writer, sheet_name="rs_rct", index=False)
            fit_matrix.to_excel(writer, sheet_name="fit_overlay", index=False, header=False)
            if drt_matrix is not None:
                drt_matrix.to_excel(writer, sheet_name="drt", index=False, header=False)
        return {"workbook": workbook}

    suffix = ".csv" if fmt == "csv" else ".txt"
    base = target.with_suffix("")
    paths = {
        "raw_plot": _write_matrix(raw_matrix, base.with_name(base.name + "_raw_plot" + suffix), fmt),
        "rs_rct": _write_frame(rs_rct_table, base.with_name(base.name + "_rs_rct" + suffix), fmt),
        "fit_overlay": _write_matrix(fit_matrix, base.with_name(base.name + "_fit_overlay" + suffix), fmt),
    }
    if drt_matrix is not None:
        paths["drt"] = _write_matrix(drt_matrix, base.with_name(base.name + "_drt" + suffix), fmt)
    return paths


def export_fit_results(output_dir: str | Path, spectrum: SpectrumData, quality: QualityReport, fit: FitOutcome) -> dict[str, Path]:
    return export_spectrum_bundle(output_dir, spectrum, fit=fit, quality=quality, fmt="txt")


def write_drt_only_export(output_path: str | Path, spectra: list[SpectrumData], drt_source_dir: str | Path, *, fmt: str = "csv") -> dict[str, Path]:
    target = Path(output_path)
    matrix = build_drt_matrix(spectra, drt_source_dir)
    if matrix is None:
        return {}
    if fmt == "xlsx":
        workbook = target if target.suffix.lower() == ".xlsx" else target.with_suffix(".xlsx")
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            matrix.to_excel(writer, sheet_name="drt", index=False, header=False)
        return {"drt": workbook}
    suffix = ".csv" if fmt == "csv" else ".txt"
    return {"drt": _write_matrix(matrix, target.with_suffix(suffix), fmt)}


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


def build_fit_overlay_matrix(spectra: list[SpectrumData], fits: list[Optional[FitOutcome]]) -> pd.DataFrame:
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
        for spectrum, fit in zip(spectra, fits):
            row.append(_safe_get(spectrum.z_real_ohm, i))
            row.append(_safe_get(spectrum.minus_z_imag_ohm, i))
            if fit is not None and fit.predicted_real_ohm is not None and fit.predicted_imag_ohm is not None:
                row.append(_safe_get(fit.predicted_real_ohm, i))
                row.append(_safe_get(-fit.predicted_imag_ohm, i))
            else:
                row.extend([np.nan, np.nan])
        rows.append(row)
    return pd.DataFrame(rows)


def build_rs_rct_table(pairs: list[tuple[SpectrumData, Optional[FitOutcome]]]) -> pd.DataFrame:
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
    rows.extend({"section": "quality", "key": f"issue_{i+1}", "value": line} for i, line in enumerate(quality.summary_lines()))
    rows.extend({"section": "parameter", "key": key, "value": value} for key, value in fit.parameters.items())
    rows.extend({"section": "stat", "key": key, "value": value} for key, value in fit.statistics.items())
    return pd.DataFrame(rows)


def build_drt_matrix(spectra: list[SpectrumData], drt_source_dir: str | Optional[Path]) -> Optional[pd.DataFrame]:
    if drt_source_dir is None:
        return None
    source = Path(drt_source_dir)
    if not source.exists():
        return None

    parsed: list[tuple[str, list[float], list[float]]] = []
    max_len = 0
    first_logtau: list[float] | None = None
    
    for spectrum in spectra:
        file_path = source / f"{spectrum.metadata.file_path.stem}_DRT.txt"
        if not file_path.exists():
            continue
        tau, gamma = _parse_drt_file(file_path)
        logtau = [float(np.log10(t)) if t > 0 else np.nan for t in tau]
        parsed.append((_export_label_for_spectrum(spectrum), logtau, gamma))
        if first_logtau is None:
            first_logtau = logtau
        max_len = max(max_len, len(logtau))
        
    if not parsed or first_logtau is None:
        return None

    rows: list[list[object]] = []
    row1: list[object] = [""]
    row2: list[object] = ["logtau"]
    for mark, _, _ in parsed:
        row1.append(mark)
        row2.append("gamma_tau")
        
    rows.append(row1)
    rows.append(row2)
    
    for i in range(max_len):
        row: list[object] = []
        row.append(first_logtau[i] if i < len(first_logtau) else np.nan)
        for _, _, gamma in parsed:
            row.append(gamma[i] if i < len(gamma) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


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
        return [], []
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
    if use_freq:
        tau = [1.0 / (2.0 * np.pi * f) if f > 0 else np.nan for f in xs]
    else:
        tau = xs
    return tau, ys


def _safe_get(values: np.ndarray, index: int):
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
