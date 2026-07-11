from __future__ import annotations

from pathlib import Path

import numpy as np

from eismaster.models import BatchSummary, SpectrumData, SpectrumMetadata
from eismaster.ui.state import AppState


def _spectrum(path: Path) -> SpectrumData:
    return SpectrumData(
        metadata=SpectrumMetadata(file_path=path),
        freq_hz=np.array([1.0]),
        z_real_ohm=np.array([1.0]),
        z_imag_ohm=np.array([-1.0]),
        z_mod_ohm=np.array([2**0.5]),
        phase_deg=np.array([-45.0]),
    )


def test_clear_all_removes_derived_state(tmp_path: Path) -> None:
    spectrum = _spectrum(tmp_path / "sample.txt")
    state = AppState(spectra=[spectrum], current_index=0)
    state.point_masks[spectrum.display_name] = np.array([True])
    state.batch_summary = BatchSummary(model_key="demo", items=[])
    state.clear_all()
    assert state.spectra == []
    assert state.current_index == -1
    assert state.point_masks == {}
    assert state.batch_summary is None


def test_invalidate_batch_outputs_preserves_loaded_spectra(tmp_path: Path) -> None:
    spectrum = _spectrum(tmp_path / "sample.txt")
    state = AppState(spectra=[spectrum], current_index=0)
    state.batch_summary = BatchSummary(model_key="demo", items=[])
    state.invalidate_batch_outputs()
    assert state.spectra == [spectrum]
    assert state.current_index == 0
    assert state.batch_summary is None
