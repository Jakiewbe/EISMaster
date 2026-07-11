from __future__ import annotations

import numpy as np

from eismaster.models import BatchSummary, FitOutcome


def format_stat(value: object) -> str:
    if value is None:
        return ""
    number = float(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.2f}"


def _primary_error_pairs(fit: FitOutcome) -> list[tuple[str, str]]:
    if fit.model_key == "zview_double_rq_qrwo":
        return [("Rs_stderr_pct", "Rs"), ("Rsei_stderr_pct", "Rsei"), ("Rct_stderr_pct", "Rct")]
    return [("Rs_stderr_pct", "Rs"), ("Rct_stderr_pct", "Rct")]


def fit_error_summary(fit: FitOutcome | None) -> str:
    if fit is None:
        return ""
    values: list[tuple[str, float]] = []
    for key, label in _primary_error_pairs(fit):
        raw = fit.statistics.get(key)
        if raw is None:
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append((label, value))
    if not values:
        return "未计算误差"
    high = [f"{label} {value:.1f}%" for label, value in values if value > 20.0]
    if high:
        return "超误差:" + "，".join(high)
    return "，".join(f"{label} {value:.1f}%" for label, value in values)


def fit_status_text(
    fit: FitOutcome | None,
    batch_summary: BatchSummary | None,
    *,
    fit_busy: bool,
    batch_busy: bool,
    drt_busy: bool,
) -> str:
    if drt_busy:
        return "DRT 运行中"
    if batch_busy:
        return "批量拟合进行中"
    if fit_busy:
        return "正在拟合..."
    if fit is not None:
        labels = {
            "ok": "拟合正常",
            "warn": "拟合警告",
            "failed": "拟合失败",
            "unavailable": "拟合不可用",
        }
        return labels.get(fit.status, fit.status)
    if batch_summary is not None and batch_summary.items:
        total = len(batch_summary.items)
        done = sum(item.fit is not None for item in batch_summary.items)
        failed = any(item.fit is not None and item.fit.status == "failed" for item in batch_summary.items)
        warned = any(item.fit is not None and item.fit.status == "warn" for item in batch_summary.items)
        suffix = "，含失败项目" if failed else "，含警告项目" if warned else ""
        return f"批量完成 {done}/{total}{suffix}"
    return "等待中"
