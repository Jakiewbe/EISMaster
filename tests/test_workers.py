from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from eismaster.matlab_drt import MatlabDrtConfig, MatlabDrtResult
from eismaster.models import BatchSummary, FitOutcome
from eismaster.ui.workers import BatchFitWorker, MatlabDrtWorker, SingleFitWorker
from tests.fixture_factory import make_double_arc_spectrum


def test_single_fit_worker_emits_result(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    fit = FitOutcome(model_key="model", model_label="Model", status="ok", message="done")
    emitted: list[tuple[object, object, str]] = []
    worker = SingleFitWorker(spectrum, "model", None)
    worker.finished.connect(lambda result, error, name: emitted.append((result, error, name)))
    with patch("eismaster.ui.workers.fit_spectrum", return_value=fit):
        worker.run()
    assert emitted == [(fit, None, spectrum.display_name)]


def test_single_fit_worker_exposes_original_exception(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    failure = RuntimeError("fit failed")
    emitted: list[tuple[object, object, str]] = []
    worker = SingleFitWorker(spectrum, "model", None)
    worker.finished.connect(lambda result, error, name: emitted.append((result, error, name)))
    with patch("eismaster.ui.workers.fit_spectrum", side_effect=failure):
        worker.run()
    assert emitted == [(None, failure, spectrum.display_name)]


def test_batch_worker_forwards_progress_and_result(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    summary = BatchSummary(model_key="auto", items=[])
    progress: list[tuple[int, int, str]] = []
    finished: list[tuple[object, object]] = []
    worker = BatchFitWorker([spectrum])
    worker.progress.connect(lambda index, total, name: progress.append((index, total, name)))
    worker.finished.connect(lambda result, error: finished.append((result, error)))

    def run_batch(spectra, *, progress_callback):
        del spectra
        item = type("Item", (), {"spectrum": spectrum})()
        progress_callback(1, 1, item)
        return summary

    with patch("eismaster.ui.workers.analyze_batch_auto", side_effect=run_batch):
        worker.run()
    assert progress == [(1, 1, spectrum.display_name)]
    assert finished == [(summary, None)]


def test_matlab_worker_stages_and_emits_result(tmp_path: Path) -> None:
    spectrum = make_double_arc_spectrum(tmp_path / "sample.txt")
    staging = tmp_path / "staging"
    result = MatlabDrtResult(
        command=["matlab"],
        returncode=0,
        stdout="",
        stderr="",
        output_files=[],
        staging_dir=staging,
        output_dir=tmp_path / "results",
    )
    finished: list[tuple[object, object]] = []
    worker = MatlabDrtWorker(MatlabDrtConfig(), [spectrum], tmp_path)
    worker.finished.connect(lambda value, error: finished.append((value, error)))
    with patch("eismaster.ui.workers.stage_matlab_drt_inputs", return_value=staging), patch(
        "eismaster.ui.workers.run_matlab_drt", return_value=result
    ):
        worker.run()
    assert finished == [(result, None)]
