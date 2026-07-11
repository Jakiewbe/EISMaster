from __future__ import annotations

from pathlib import Path


def test_pyinstaller_spec_bundles_matlab_resources() -> None:
    text = Path("EISMaster.spec").read_text(encoding="utf-8")
    assert '"matlab_bridge"' in text
    assert '"matlab-DRTtools-local"' in text
    assert "datas.append" in text
    assert "raise FileNotFoundError" in text
