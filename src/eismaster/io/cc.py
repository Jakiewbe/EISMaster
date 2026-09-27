from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

CC_COUNT_OFFSET = 0x262
CC_INTERVAL_OFFSET = 0x45E
CC_CURRENT_OFFSET = 0x4C4
CC_DATA_HEADER_END = 0x696
CC_TECHNIQUE_MARKER = b"Chronopotentiometry"
CC_FILE_PATTERN = re.compile(r"^(?P<prefix>.+?)_CC_(?P<target>\d+p\d+)V$", re.IGNORECASE)
ISTEP_DATA_OFFSET = 0x699
ISTEP_TECHNIQUE_MARKER = b"Multi-Current Steps"
ISTEP_FILE_PATTERN = re.compile(r"^(?P<prefix>.+?)_CC_T(?P<time>\d+(?:\.\d+)?)M$", re.IGNORECASE)


@dataclass(frozen=True)
class CCSegment:
    source_path: Path
    sample_id: str
    target_voltage_v: float | None
    mode: str
    sample_interval_s: float
    voltage_v: np.ndarray
    current_sign_raw: int
    sequence_key: float | None = None

    @property
    def n_points(self) -> int:
        return int(self.voltage_v.size)

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.n_points - 1) * self.sample_interval_s)


def parse_cc_bin(path: str | Path) -> CCSegment:
    source_path = Path(path)
    raw = source_path.read_bytes()
    if _is_cc_bytes(raw):
        return _parse_chronopotentiometry(source_path, raw)
    if _is_istep_bytes(raw):
        return _parse_istep(source_path, raw)
    raise ValueError(f"Not a supported CHI CC file: {source_path.name}")


def _parse_chronopotentiometry(source_path: Path, raw: bytes) -> CCSegment:

    match = CC_FILE_PATTERN.fullmatch(source_path.stem)
    if match is None:
        raise ValueError(f"Invalid CC filename pattern: {source_path.name}")

    count = struct.unpack_from("<H", raw, CC_COUNT_OFFSET)[0]
    interval_s = struct.unpack_from("<f", raw, CC_INTERVAL_OFFSET)[0]
    current_sign_raw = struct.unpack_from("<h", raw, CC_CURRENT_OFFSET)[0]
    data_start = len(raw) - count * 4
    if count < 2 or data_start < CC_DATA_HEADER_END:
        raise ValueError(f"Invalid CC data length: {source_path.name}")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError(f"Invalid CC sampling interval: {source_path.name}")
    if current_sign_raw == 0:
        raise ValueError(f"Missing CC current direction: {source_path.name}")

    voltage_v = np.asarray(
        [struct.unpack_from("<f", raw, data_start + index * 4)[0] for index in range(count)],
        dtype=float,
    )
    if not np.all(np.isfinite(voltage_v)):
        raise ValueError(f"Non-finite CC voltage data: {source_path.name}")

    target_voltage_v = float(match.group("target").replace("p", "."))
    sample_id = match.group("prefix").rsplit("_", 1)[-1]
    mode = "charge" if current_sign_raw < 0 else "discharge"
    voltage_delta = float(voltage_v[-1] - voltage_v[0])
    if mode == "charge" and voltage_delta <= 0:
        raise ValueError(f"CC voltage direction conflicts with charge: {source_path.name}")
    if mode == "discharge" and voltage_delta >= 0:
        raise ValueError(f"CC voltage direction conflicts with discharge: {source_path.name}")

    return CCSegment(
        source_path=source_path,
        sample_id=sample_id,
        target_voltage_v=target_voltage_v,
        mode=mode,
        sample_interval_s=float(interval_s),
        voltage_v=voltage_v,
        current_sign_raw=current_sign_raw,
    )


def _parse_istep(source_path: Path, raw: bytes) -> CCSegment:
    match = ISTEP_FILE_PATTERN.fullmatch(source_path.stem)
    if match is None:
        raise ValueError(f"Invalid ISTEP CC filename pattern: {source_path.name}")

    data_length = len(raw) - ISTEP_DATA_OFFSET
    count, remainder = divmod(data_length, 4)
    if remainder or count < 2:
        raise ValueError(f"Incomplete ISTEP CC data: {source_path.name}")

    voltage_v = np.asarray(
        [struct.unpack_from("<f", raw, ISTEP_DATA_OFFSET + index * 4)[0] for index in range(count)],
        dtype=float,
    )
    if not np.all(np.isfinite(voltage_v)):
        raise ValueError(f"Non-finite ISTEP CC voltage data: {source_path.name}")

    voltage_delta = float(voltage_v[-1] - voltage_v[0])
    if voltage_delta == 0:
        raise ValueError(f"Cannot determine ISTEP CC direction: {source_path.name}")
    mode = "charge" if voltage_delta > 0 else "discharge"
    return CCSegment(
        source_path=source_path,
        sample_id=match.group("prefix").rsplit("_", 1)[-1],
        target_voltage_v=None,
        mode=mode,
        sample_interval_s=1.0,
        voltage_v=voltage_v,
        current_sign_raw=-1 if mode == "charge" else 1,
        sequence_key=float(match.group("time")),
    )


def load_cc_segments_from_folder(folder: str | Path) -> list[CCSegment]:
    base = Path(folder)
    segments: list[CCSegment] = []
    errors: list[str] = []
    for path in sorted(base.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".bin":
            continue
        raw = path.read_bytes()
        if not (_is_cc_bytes(raw) or _is_istep_bytes(raw)):
            continue
        try:
            segments.append(parse_cc_bin(path))
        except Exception as exc:
            if _is_istep_bytes(raw) and len(raw) <= ISTEP_DATA_OFFSET + 4:
                continue
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise ValueError("CC 文件解析失败:\n" + "\n".join(errors))
    if not segments:
        raise ValueError(f"No valid CC files found in {base}")
    return _normalize_istep_modes(segments)


def _is_cc_bytes(raw: bytes) -> bool:
    return raw[8:10] == b"CP" and CC_TECHNIQUE_MARKER in raw[:96]


def _is_istep_bytes(raw: bytes) -> bool:
    return raw[8:13] == b"ISTEP" and ISTEP_TECHNIQUE_MARKER in raw[:96]


def _normalize_istep_modes(segments: list[CCSegment]) -> list[CCSegment]:
    grouped: dict[str, list[CCSegment]] = {}
    for segment in segments:
        if segment.sequence_key is not None:
            grouped.setdefault(segment.sample_id, []).append(segment)

    normalized = list(segments)
    for sample_id, group in grouped.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda segment: segment.sequence_key or 0.0)
        overall_delta = float(ordered[-1].voltage_v[-1] - ordered[0].voltage_v[0])
        if overall_delta == 0:
            continue
        mode = "charge" if overall_delta > 0 else "discharge"
        sign = -1 if mode == "charge" else 1
        for index, segment in enumerate(normalized):
            if segment.sample_id == sample_id and segment.sequence_key is not None:
                normalized[index] = replace(segment, mode=mode, current_sign_raw=sign)
    return normalized
