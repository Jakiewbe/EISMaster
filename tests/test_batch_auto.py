from __future__ import annotations

import unittest

from eismaster.analysis.batch import (
    _apply_hysteresis,
    _candidate_score,
    _finite_or,
    _fit_scientific_score,
    _mode_label,
    _needs_expensive_retry,
    _should_run_double_fit,
)
from eismaster.analysis.segmentation import SegmentDetection
from eismaster.models import FitOutcome


class BatchAutoTests(unittest.TestCase):
    def test_hysteresis_keeps_current_when_preferred_matches(self) -> None:
        active, pending, count, label = _apply_hysteresis("single", "single", None, 0)
        self.assertEqual(active, "single")
        self.assertIsNone(pending)
        self.assertEqual(count, 0)
        self.assertIn("keeps", label)

    def test_hysteresis_starts_pending_counter(self) -> None:
        active, pending, count, label = _apply_hysteresis("single", "double", None, 0)
        self.assertEqual(active, "single")
        self.assertEqual(pending, "double")
        self.assertEqual(count, 1)
        self.assertIn("evaluating", label)

    def test_hysteresis_switches_after_required_count(self) -> None:
        # Start with single, prefer double (DOUBLE_SWITCH_COUNT = 2)
        active, pending, count, _ = _apply_hysteresis("single", "double", None, 0)
        self.assertEqual(active, "single")
        self.assertEqual(count, 1)
        # Continue preferring double - should switch after 2 votes
        active, pending, count, label = _apply_hysteresis(active, "double", pending, count)
        self.assertEqual(active, "double")
        self.assertIsNone(pending)
        self.assertEqual(count, 0)
        self.assertIn("switched", label)

    def test_hysteresis_pending_resets_on_flip(self) -> None:
        active, pending, count, _ = _apply_hysteresis("single", "double", None, 0)
        self.assertEqual(count, 1)
        # Now prefer single again (matches active)
        active, pending, count, label = _apply_hysteresis(active, "single", pending, count)
        self.assertEqual(active, "single")
        self.assertIsNone(pending)
        self.assertEqual(count, 0)

    def test_finite_or_with_none(self) -> None:
        self.assertEqual(_finite_or(None, 42.0), 42.0)

    def test_finite_or_with_valid_value(self) -> None:
        self.assertEqual(_finite_or(3.14, 42.0), 3.14)

    def test_finite_or_with_nan(self) -> None:
        import math
        self.assertEqual(_finite_or(float("nan"), 42.0), 42.0)

    def test_finite_or_with_inf(self) -> None:
        self.assertEqual(_finite_or(float("inf"), 42.0), 42.0)

    def test_mode_label(self) -> None:
        self.assertEqual(_mode_label("double"), "double-arc")
        self.assertEqual(_mode_label("single"), "single-arc")

    def test_fit_scientific_score_prefers_clean_result(self) -> None:
        clean = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="ok",
            message="mode=single",
            parameters={"Rct": 10.0},
            statistics={"aicc": -120.0, "chi2_reduced": 0.2, "Rs_stderr_pct": 3.0, "Rct_stderr_pct": 4.0},
        )
        risky = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="warn",
            message="mode=single; parameters near bounds; strong_parameter_correlation; direct global fallback",
            parameters={"Rct": 10.0},
            statistics={
                "aicc": -120.0,
                "chi2_reduced": 0.2,
                "Rs_stderr_pct": 3.0,
                "Rct_stderr_pct": 4.0,
                "correlation_matrix_max": 0.998,
            },
        )
        self.assertLess(_fit_scientific_score(clean), _fit_scientific_score(risky))

    def test_candidate_score_penalizes_segment_mismatch(self) -> None:
        hint = SegmentDetection(requested_mode="auto", resolved_mode="single", peak_indices=(10,), split_indices=(20,))
        stable = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="ok",
            message="mode=single",
            parameters={"Rct": 12.0},
            statistics={"aicc": -100.0, "chi2_reduced": 0.3, "Rs_stderr_pct": 5.0, "Rct_stderr_pct": 6.0},
        )
        mismatch = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="warn",
            message="mode=single",
            parameters={"Rct": 12.0},
            statistics={"aicc": -110.0, "chi2_reduced": 0.2, "Rs_stderr_pct": 5.0, "Rct_stderr_pct": 6.0},
            diagnosis_type="segment_error",
            diagnosis_severity="error",
            diagnosis_explanation="The selected segmentation does not match the Nyquist features well enough.",
        )
        self.assertLess(_candidate_score(stable, "single", hint), _candidate_score(mismatch, "single", hint))

    def test_should_run_double_fit_skips_stable_single_case(self) -> None:
        hint = SegmentDetection(requested_mode="auto", resolved_mode="single", peak_indices=(10,), split_indices=(20,))
        single_fit = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="ok",
            message="mode=single",
            parameters={"Rct": 10.0},
            statistics={"Rs_stderr_pct": 3.0, "Rct_stderr_pct": 4.0},
        )
        self.assertFalse(_should_run_double_fit(single_fit, hint))

    def test_should_run_double_fit_runs_on_warned_single_case(self) -> None:
        hint = SegmentDetection(requested_mode="auto", resolved_mode="single", peak_indices=(10,), split_indices=(20,))
        single_fit = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="warn",
            message="mode=single",
            parameters={"Rct": 10.0},
            statistics={"Rs_stderr_pct": 3.0, "Rct_stderr_pct": 4.0},
            diagnosis_type="model_mismatch",
            diagnosis_severity="warning",
        )
        self.assertTrue(_should_run_double_fit(single_fit, hint))

    def test_needs_expensive_retry_for_fallback_or_bad_diagnosis(self) -> None:
        fit = FitOutcome(
            model_key="zview_segmented_rq_rwo",
            model_label="single",
            status="warn",
            message="mode=single",
            fallback_from="zview_segmented_rq_rwo",
            diagnosis_type="ill_conditioned",
            diagnosis_severity="warning",
        )
        self.assertTrue(_needs_expensive_retry(fit))


if __name__ == "__main__":
    unittest.main()
