from __future__ import annotations

from eismaster.models import FitOutcome
from eismaster.ui.presenters import fit_error_summary, fit_status_text, format_stat


def test_fit_status_text_exposes_running_and_failed_states() -> None:
    assert fit_status_text(None, None, fit_busy=True, batch_busy=False, drt_busy=False) == "正在拟合..."
    fit = FitOutcome(model_key="m", model_label="M", status="failed", message="bad fit")
    assert fit_status_text(fit, None, fit_busy=False, batch_busy=False, drt_busy=False) == "拟合失败"


def test_format_stat_preserves_existing_display_rules() -> None:
    assert format_stat(None) == ""
    assert format_stat(float("nan")) == "nan"
    assert format_stat(1.234) == "1.23"


def test_error_summary_uses_primary_parameter_errors() -> None:
    fit = FitOutcome(
        model_key="zview_segmented_rq_rwo",
        model_label="M",
        status="warn",
        message="warning",
        statistics={"Rs_stderr_pct": 5.0, "Rct_stderr_pct": 25.0},
    )
    assert fit_error_summary(fit) == "超误差:Rct 25.0%"
