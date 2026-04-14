from __future__ import annotations

import numpy as np

from eismaster.analysis.preprocessing import detect_outliers_common
from eismaster.models import QualityIssue, QualityReport, SpectrumData


def assess_spectrum_quality(spectrum: SpectrumData, run_kk: bool = False) -> QualityReport:
    issues: list[QualityIssue] = []
    freq = spectrum.freq_hz
    z_real = spectrum.z_real_ohm
    z_imag = spectrum.z_imag_ohm

    if spectrum.n_points < 8:
        issues.append(QualityIssue("error", "数据点过少，无法稳定拟合。"))
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(z_real)) or not np.all(np.isfinite(z_imag)):
        issues.append(QualityIssue("error", "谱图中存在非有限值。"))
    if np.any(freq <= 0):
        issues.append(QualityIssue("error", "频率必须为正值。"))
    if np.any(np.diff(freq) >= 0):
        issues.append(QualityIssue("warning", "频率序列不是严格降序。"))
    if np.unique(np.round(freq, 12)).size != freq.size:
        issues.append(QualityIssue("warning", "检测到重复频点。"))
    if np.any(z_real < 0):
        issues.append(QualityIssue("warning", "检测到负实部阻抗。"))
    if z_imag.size > 0 and z_imag[0] > 0:
        issues.append(QualityIssue("info", "高频端虚部起点高于零。"))

    outlier_flags = _detect_outliers(spectrum)
    outlier_count = int(outlier_flags.sum())
    if outlier_count > 0:
        issues.append(QualityIssue("warning", f"检测到 {outlier_count} 个可能异常点。"))

    kk_status = "not_run"
    kk_message = "KK/Z-HIT 未执行。"
    if run_kk:
        kk_status, kk_message = _run_kramers_kronig_check(spectrum, issues)

    status = "pass"
    if any(issue.severity == "error" for issue in issues):
        status = "fail"
    elif any(issue.severity == "warning" for issue in issues):
        status = "warn"

    return QualityReport(status=status, issues=issues, kk_status=kk_status, kk_message=kk_message)


def _run_kramers_kronig_check(spectrum: SpectrumData, issues: list[QualityIssue]) -> tuple[str, str]:
    try:
        import pyimpspec

        data_set = pyimpspec.DataSet(
            frequencies=spectrum.freq_hz.astype(float),
            impedances=spectrum.impedance.astype(complex),
            path=str(spectrum.metadata.file_path),
            label=spectrum.display_name,
        )
        kk_result = pyimpspec.perform_kramers_kronig_test(data_set, timeout=10, num_procs=1)
        stats = kk_result.to_statistics_dataframe()
        pseudo_chi2 = float(stats.loc[stats["Label"] == "Log pseudo chi-squared", "Value"].iloc[0])

        if pseudo_chi2 <= 0.0:
            kk_status = "pass"
        elif pseudo_chi2 <= 0.5:
            kk_status = "warn"
            issues.append(QualityIssue("warning", f"KK 检查临界：log pseudo chi-squared = {pseudo_chi2:.3f}。"))
        else:
            kk_status = "fail"
            issues.append(QualityIssue("warning", f"KK 检查失败：log pseudo chi-squared = {pseudo_chi2:.3f}。"))

        kk_message = f"log pseudo chi-squared = {pseudo_chi2:.3f}; RC count = {kk_result.get_num_RC()}"
        return kk_status, kk_message
    except Exception as exc:
        return "not_run", f"KK/Z-HIT 不可用：{exc}"


def _detect_outliers(spectrum: SpectrumData) -> np.ndarray:
    mask = np.ones(spectrum.n_points, dtype=bool)
    return detect_outliers_common(
        spectrum,
        mask,
        curvature_scale=6.0,
        slope_scale=6.0,
        gradient_scale=7.0,
        vote_threshold=2,
    )
