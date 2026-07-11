from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from eismaster.models import SpectrumData, SpectrumMetadata

_SINGLE_FREQ = np.geomspace(100000.0, 0.01, 85)
_SINGLE_OMEGA = 2.0 * np.pi * _SINGLE_FREQ
_SINGLE_Z = 5.0 + 42.0 / (1.0 + (1j * _SINGLE_OMEGA * 2e-3) ** 0.88)
SINGLE_ROWS = tuple(
    (float(freq), float(value.real), float(value.imag))
    for freq, value in zip(_SINGLE_FREQ, _SINGLE_Z, strict=True)
)


def write_single_arc_txt(path: Path) -> Path:
    rows = [
        "Synthetic fixture",
        "A.C. Impedance",
        "Instrument Model: CHI660F",
        "Note: synthetic regression fixture",
        "Freq/Hz, Z'/ohm, Z''/ohm",
        "Hz,ohm,ohm",
    ]
    rows.extend(f"{freq:.12g},{real:.12g},{imag:.12g}" for freq, real, imag in SINGLE_ROWS)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_single_arc_bin(path: Path) -> Path:
    header = bytearray(0x280)
    header[:26] = b"IMP A.C. Impedance CHI660F"
    struct.pack_into("<H", header, 0x25E, len(SINGLE_ROWS))
    records = b"".join(struct.pack("<4f", freq, freq, real, imag) for freq, real, imag in SINGLE_ROWS)
    path.write_bytes(bytes(header) + records)
    return path


def make_double_arc_spectrum(path: Path) -> SpectrumData:
    freq = np.geomspace(100000.0, 0.1, 61)
    omega = 2.0 * np.pi * freq
    z = 4.0 + 18.0 / (1.0 + (1j * omega * 2e-4) ** 0.9)
    z += 55.0 / (1.0 + (1j * omega * 2e-2) ** 0.82)
    return SpectrumData(
        metadata=SpectrumMetadata(file_path=path, source_format="synthetic"),
        freq_hz=freq,
        z_real_ohm=z.real,
        z_imag_ohm=z.imag,
        z_mod_ohm=np.abs(z),
        phase_deg=np.degrees(np.angle(z)),
    )
