from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class SpectrumMetadata:
    file_path: Path
    technique: str = ""
    instrument_model: str = ""
    acquired_at: Optional[datetime] = None
    note: str = ""
    header: dict[str, str] = field(default_factory=dict)
    source_format: str = ""


@dataclass
class SpectrumData:
    metadata: SpectrumMetadata
    freq_hz: np.ndarray
    z_real_ohm: np.ndarray
    z_imag_ohm: np.ndarray
    z_mod_ohm: np.ndarray
    phase_deg: np.ndarray

    @property
    def impedance(self) -> np.ndarray:
        return self.z_real_ohm + 1j * self.z_imag_ohm

    @property
    def minus_z_imag_ohm(self) -> np.ndarray:
        return -self.z_imag_ohm

    @property
    def n_points(self) -> int:
        return int(self.freq_hz.size)

    @property
    def display_name(self) -> str:
        return self.metadata.file_path.name

    @property
    def acquired_label(self) -> str:
        if self.metadata.acquired_at is None:
            return "未知"
        return self.metadata.acquired_at.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class QualityIssue:
    severity: str
    message: str


@dataclass
class QualityReport:
    status: str
    issues: list[QualityIssue] = field(default_factory=list)
    kk_status: str = "not_run"
    kk_message: str = "KK/Z-HIT 未执行。"

    def summary_lines(self) -> list[str]:
        lines = [f"状态: {self.status}", f"KK/Z-HIT: {self.kk_status} - {self.kk_message}"]
        if not self.issues:
            lines.append("未发现明显质量问题。")
            return lines
        for issue in self.issues:
            lines.append(f"[{issue.severity}] {issue.message}")
        return lines


@dataclass
class FitOutcome:
    model_key: str
    model_label: str
    status: str
    message: str
    parameters: dict[str, float] = field(default_factory=dict)
    statistics: dict[str, float] = field(default_factory=dict)
    predicted_real_ohm: Optional[np.ndarray] = None
    predicted_imag_ohm: Optional[np.ndarray] = None
    masked_points: int = 0
    preprocess_actions: list[str] = field(default_factory=list)
    fallback_from: Optional[str] = None

    @property
    def predicted_impedance(self) -> Optional[np.ndarray]:
        if self.predicted_real_ohm is None or self.predicted_imag_ohm is None:
            return None
        return self.predicted_real_ohm + 1j * self.predicted_imag_ohm


@dataclass
class BatchItemResult:
    spectrum: SpectrumData
    quality: QualityReport
    fit: Optional[FitOutcome] = None


@dataclass
class BatchSummary:
    model_key: str
    items: list[BatchItemResult]


def sort_key_for_spectrum(spectrum: SpectrumData) -> Tuple[Any, str]:
    stamp = spectrum.metadata.acquired_at or datetime.min
    return (stamp, spectrum.metadata.file_path.name.lower())


IMPFT_BASELINE = 700.0


class SamplingMode(str, Enum):
    FIXED = "fixed"
    SEGMENTED = "segmented"
    MANUAL = "manual"


class FixedSamplingStrategy(str, Enum):
    INTERVAL = "interval"
    COUNT = "count"


class EisTechnique(str, Enum):
    PEIS = "PEIS"
    IMPFT = "IMPFT"


@dataclass
class SamplingSegment:
    duration_min: float = 120.0
    point_count: int = 3


@dataclass
class FixedSamplingPlan:
    strategy: FixedSamplingStrategy = FixedSamplingStrategy.COUNT
    interval_min: float = 30.0
    point_count: int = 10


@dataclass
class SegmentedSamplingPlan:
    segments: list[SamplingSegment] = field(default_factory=list)


@dataclass
class ManualSamplingPlan:
    raw_text: str = ""


@dataclass
class EisTechniqueConfig:
    technique: EisTechnique = EisTechnique.IMPFT
    start_freq_hz: float = 100_000.0
    stop_freq_hz: float = 0.01
    amplitude_mv: float = 5.0
    baseline_duration_s: float = IMPFT_BASELINE
    initial_e_mode: str = "Load E"
    bias_current_a: Optional[float] = None


@dataclass
class ProtocolStep:
    name: str = "Discharge"
    duration_min: float = 360.0
    current_a: float = 0.1
    sampling_mode: SamplingMode = SamplingMode.FIXED
    fixed_sampling: FixedSamplingPlan = field(default_factory=FixedSamplingPlan)
    segmented_sampling: SegmentedSamplingPlan = field(
        default_factory=lambda: SegmentedSamplingPlan(
            segments=[
                SamplingSegment(duration_min=90.0, point_count=2),
                SamplingSegment(duration_min=150.0, point_count=5),
                SamplingSegment(duration_min=120.0, point_count=3),
            ]
        )
    )
    manual_sampling: ManualSamplingPlan = field(default_factory=ManualSamplingPlan)
    eis: EisTechniqueConfig = field(default_factory=EisTechniqueConfig)
    note: str = ""
    enabled: bool = True


def default_protocol_step() -> ProtocolStep:
    return ProtocolStep()


@dataclass
class ExperimentPlan:
    name: str = "Operando EIS Plan"
    capacity_ah: float = 0.6
    initial_soc_percent: float = 100.0
    bias_current_a: float = 0.1
    steps: list[ProtocolStep] = field(default_factory=lambda: [default_protocol_step()])


@dataclass
class CurrentProfilePoint:
    time_min: float
    current_a: float


@dataclass
class SamplingEvent:
    step_index: int
    step_name: str
    absolute_time_min: float
    local_time_min: float
    event_end_time_min: float
    current_a: float
    soc_before_percent: float
    soc_after_percent: float
    is_risky: bool


@dataclass
class SocSimulationPoint:
    time_min: float
    soc_percent: float


@dataclass
class SocSimulationResult:
    soc_curve: list[SocSimulationPoint]
    current_profile: list[CurrentProfilePoint]
    sampling_events: list[SamplingEvent]
    total_duration_min: float
    total_discharge_ah: float
    total_eis_ah: float

    @property
    def risky_events(self) -> list[SamplingEvent]:
        return [event for event in self.sampling_events if event.is_risky]
