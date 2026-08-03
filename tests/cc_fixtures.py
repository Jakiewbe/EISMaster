from __future__ import annotations

import struct
from pathlib import Path


def write_cc_bin(path: Path, voltages: list[float], current_raw: int) -> Path:
    raw = bytearray(0x696 + len(voltages) * 4)
    raw[8:10] = b"CP"
    marker = b"Chronopotentiometry"
    raw[14 : 14 + len(marker)] = marker
    struct.pack_into("<H", raw, 0x262, len(voltages))
    struct.pack_into("<f", raw, 0x492, 0.01)
    struct.pack_into("<h", raw, 0x4C4, current_raw)
    for index, voltage in enumerate(voltages):
        struct.pack_into("<f", raw, 0x696 + index * 4, voltage)
    path.write_bytes(raw)
    return path


def write_eis_marker(path: Path) -> Path:
    raw = bytearray(0x696)
    raw[8:11] = b"IMP"
    path.write_bytes(raw)
    return path
