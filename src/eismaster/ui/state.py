from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eismaster.analysis.segmentation import SegmentDetection
from eismaster.models import BatchSummary, FitOutcome, QualityReport, SpectrumData


@dataclass
class AppState:
    spectra: list[SpectrumData] = field(default_factory=list)
    qualities: dict[str, QualityReport] = field(default_factory=dict)
    fits: dict[tuple[str, str], FitOutcome] = field(default_factory=dict)
    segment_hints: dict[str, SegmentDetection] = field(default_factory=dict)
    point_masks: dict[str, np.ndarray] = field(default_factory=dict)
    batch_summary: BatchSummary | None = None
    current_index: int = -1
    drt_busy: bool = False
    batch_busy: bool = False
    fit_busy: bool = False

    def invalidate_batch_outputs(self) -> None:
        self.batch_summary = None

    def clear_all(self) -> None:
        self.spectra.clear()
        self.qualities.clear()
        self.fits.clear()
        self.segment_hints.clear()
        self.point_masks.clear()
        self.batch_summary = None
        self.current_index = -1
