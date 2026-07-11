from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from eismaster.analysis.batch import analyze_batch_auto
from eismaster.analysis.fitting import fit_spectrum
from eismaster.analysis.segmentation import ArcRange
from eismaster.matlab_drt import MatlabDrtConfig, run_matlab_drt, stage_matlab_drt_inputs
from eismaster.models import SpectrumData


class MatlabDrtWorker(QObject):
    finished = Signal(object, object)

    def __init__(self, config: MatlabDrtConfig, spectra: list[SpectrumData], export_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.spectra = spectra
        self.export_dir = export_dir

    def run(self) -> None:
        try:
            staging_dir = stage_matlab_drt_inputs(self.spectra, self.export_dir)
            result = run_matlab_drt(self.config, staging_dir, self.export_dir / "results")
            self.finished.emit(result, None)
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, exc)


class BatchFitWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, object)

    def __init__(self, spectra: list[SpectrumData]) -> None:
        super().__init__()
        self.spectra = spectra

    def run(self) -> None:
        try:
            summary = analyze_batch_auto(self.spectra, progress_callback=self._emit_progress)
            self.finished.emit(summary, None)
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, exc)

    def _emit_progress(self, index: int, total: int, item) -> None:
        self.progress.emit(index, total, item.spectrum.display_name)


class SingleFitWorker(QObject):
    finished = Signal(object, object, str)

    def __init__(self, spectrum: SpectrumData, template_key: str, arc_ranges: list[ArcRange] | None) -> None:
        super().__init__()
        self.spectrum = spectrum
        self.template_key = template_key
        self.arc_ranges = arc_ranges
        self.display_name = spectrum.display_name

    def run(self) -> None:
        try:
            fit = fit_spectrum(self.spectrum, self.template_key, arc_ranges=self.arc_ranges)
            self.finished.emit(fit, None, self.display_name)
        except Exception as exc:  # pragma: no cover
            self.finished.emit(None, exc, self.display_name)
