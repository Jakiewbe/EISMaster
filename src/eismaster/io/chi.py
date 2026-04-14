from __future__ import annotations
from typing import Optional
import typing
from collections.abc import Iterable







import csv
import math
import re
import struct
from datetime import datetime
from pathlib import Path


import numpy as np

from eismaster.models import SpectrumData, SpectrumMetadata, sort_key_for_spectrum

SUPPORTED_SUFFIXES = {".bin", ".txt", ".csv"}
TXT_DATE_FORMATS = (
    "%b. %d, %Y   %H:%M:%S",
    "%b %d, %Y   %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def load_spectrum(path: str | Path) -> SpectrumData:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".bin":
        return parse_chi_bin(file_path)
    if suffix == ".txt":
        return parse_chi_txt(file_path)
    if suffix == ".csv":
        return parse_delimited_text(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_spectra_from_folder(folder: str | Path) -> list[SpectrumData]:
    base = Path(folder)
    spectra: list[SpectrumData] = []
    errors: list[str] = []
    for path in base.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            spectra.append(load_spectrum(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not spectra:
        detail = "\n".join(errors[:10]) if errors else "No supported files found."
        raise ValueError(f"No valid EIS files could be loaded from {base}\n{detail}")
    spectra.sort(key=sort_key_for_spectrum)
    return spectra


def parse_chi_txt(path: Path) -> SpectrumData:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if _looks_like_plain_numeric_eis(lines):
        rows = _parse_numeric_text_rows(lines)
        return _build_spectrum(
            path=path,
            source_format="txt",
            technique="A.C. Impedance",
            instrument_model="CHI660F",
            acquired_at=None,
            note="",
            header={},
            rows=rows,
        )
    if len(lines) < 5:
        raise ValueError(f"Text file too short: {path}")

    acquired_at = _parse_datetime_line(lines[0].strip())
    technique = lines[1].strip() if len(lines) > 1 else ""
    header: dict[str, str] = {}
    instrument_model = ""
    note = ""
    data_start = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Freq/Hz"):
            data_start = idx + 2
            break
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
            if key.strip() == "Instrument Model":
                instrument_model = value.strip()
            elif key.strip() == "Note":
                note = value.strip()

    if data_start is None:
        raise ValueError(f"Unable to find CHI data header in {path}")

    rows: list[tuple[float, float, float, Optional[float], Optional[float]]] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        freq = float(parts[0])
        z_real = float(parts[1])
        z_imag = float(parts[2])
        z_mod = float(parts[3]) if len(parts) > 3 and parts[3] else None
        phase = float(parts[4]) if len(parts) > 4 and parts[4] else None
        rows.append((freq, z_real, z_imag, z_mod, phase))

    return _build_spectrum(
        path=path,
        source_format="txt",
        technique=technique,
        instrument_model=instrument_model,
        acquired_at=acquired_at,
        note=note,
        header=header,
        rows=rows,
    )


def parse_delimited_text(path: Path) -> SpectrumData:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")

    if any("Freq/Hz" in line for line in lines[:20]):
        return parse_chi_txt(path)

    sample = "\n".join(lines[:5])
    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ","

    first_tokens = [token.strip() for token in lines[0].split(delimiter)]
    numeric_first_row = _is_float(first_tokens[0]) and len(first_tokens) >= 3 and _is_float(first_tokens[1]) and _is_float(first_tokens[2])
    data_lines = lines if numeric_first_row else lines[1:]

    rows: list[tuple[float, float, float, Optional[float], Optional[float]]] = []
    for line in data_lines:
        parts = [part.strip() for part in line.split(delimiter)]
        if len(parts) < 3 or not _is_float(parts[0]) or not _is_float(parts[1]) or not _is_float(parts[2]):
            continue
        freq = float(parts[0])
        z_real = float(parts[1])
        z_imag = float(parts[2])
        rows.append((freq, z_real, z_imag, None, None))

    if not rows:
        raise ValueError(f"No frequency/Z columns found in {path}")

    return _build_spectrum(
        path=path,
        source_format="csv",
        technique="A.C. Impedance",
        instrument_model="Unknown",
        acquired_at=None,
        note="",
        header={},
        rows=rows,
    )


def parse_chi_bin(path: Path) -> SpectrumData:
    raw = path.read_bytes()
    if b"IMP" not in raw[:32]:
        raise ValueError(f"{path.name} does not look like a CHI impedance binary file")

    count = _extract_record_count(raw)
    if count is None:
        raise ValueError(f"Unable to determine record count for {path.name}")
    start = len(raw) - count * 16
    if start < 0:
        raise ValueError(f"Invalid record count for {path.name}")

    rows: list[tuple[float, float, float, Optional[float], Optional[float]]] = []
    prev_freq = math.inf
    for idx in range(count):
        freq_1, freq_2, z_real, z_imag = struct.unpack_from("<4f", raw, start + idx * 16)
        if not all(math.isfinite(value) for value in (freq_1, freq_2, z_real, z_imag)):
            raise ValueError(f"Non-finite binary record at index {idx} in {path.name}")
        if freq_1 <= 0 or abs(freq_1 - freq_2) / max(freq_1, 1.0) > 1e-3:
            raise ValueError(f"Unexpected binary record layout at index {idx} in {path.name}")
        if freq_1 > prev_freq * 1.05:
            raise ValueError(f"Binary frequencies are not monotonic in {path.name}")
        prev_freq = freq_1
        rows.append((freq_1, z_real, z_imag, None, None))

    acquired_at = _extract_bin_datetime(raw)
    header = {}
    if acquired_at is not None:
        header["Acquired At"] = acquired_at.strftime("%Y-%m-%d %H:%M:%S")

    return _build_spectrum(
        path=path,
        source_format="bin",
        technique="A.C. Impedance" if b"A.C. Impedance" in raw[:128] else "Unknown",
        instrument_model="CHI660F" if b"CHI660F" in raw else "Unknown",
        acquired_at=acquired_at,
        note="",
        header=header,
        rows=rows,
    )


def _build_spectrum(
    *,
    path: Path,
    source_format: str,
    technique: str,
    instrument_model: str,
    acquired_at: Optional[datetime],
    note: str,
    header: dict[str, str],
    rows: Iterable[tuple[float, float, float, Optional[float], Optional[float]]],
) -> SpectrumData:
    numeric_rows = list(rows)
    if not numeric_rows:
        raise ValueError(f"No EIS rows parsed from {path.name}")

    freq = np.asarray([row[0] for row in numeric_rows], dtype=float)
    z_real = np.asarray([row[1] for row in numeric_rows], dtype=float)
    z_imag = np.asarray([row[2] for row in numeric_rows], dtype=float)
    z_mod = np.asarray([row[3] if row[3] is not None else math.hypot(row[1], row[2]) for row in numeric_rows], dtype=float)
    phase = np.asarray([row[4] if row[4] is not None else math.degrees(math.atan2(row[2], row[1])) for row in numeric_rows], dtype=float)

    metadata = SpectrumMetadata(
        file_path=path,
        technique=technique,
        instrument_model=instrument_model,
        acquired_at=acquired_at,
        note=note,
        header=header,
        source_format=source_format,
    )
    return SpectrumData(
        metadata=metadata,
        freq_hz=freq,
        z_real_ohm=z_real,
        z_imag_ohm=z_imag,
        z_mod_ohm=z_mod,
        phase_deg=phase,
    )


def _parse_datetime_line(text: str) -> Optional[datetime]:
    for fmt in TXT_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_record_count(raw: bytes) -> Optional[int]:
    candidates = []
    for offset in (0x25E, 0x266):
        if len(raw) >= offset + 2:
            count = struct.unpack_from("<H", raw, offset)[0]
            if 5 <= count <= len(raw) // 16:
                candidates.append(count)
    if candidates and len(set(candidates)) == 1:
        return candidates[0]
    return candidates[0] if candidates else None


def _extract_bin_datetime(raw: bytes) -> Optional[datetime]:
    offsets = (0x26A, 0x26E, 0x272, 0x276, 0x27A, 0x27E)
    if len(raw) < max(offsets) + 2:
        return None
    values = [struct.unpack_from("<H", raw, offset)[0] for offset in offsets]
    try:
        return datetime(*values)
    except ValueError:
        return None


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _split_numeric_line(line: str) -> list[str]:
    return [part for part in re.split(r"[\s,;\t]+", line.strip()) if part]


def _parse_numeric_text_rows(lines: list[str]) -> list[tuple[float, float, float, Optional[float], Optional[float]]]:
    rows: list[tuple[float, float, float, Optional[float], Optional[float]]] = []
    for line in lines:
        parts = _split_numeric_line(line)
        if len(parts) < 3 or not _is_float(parts[0]) or not _is_float(parts[1]) or not _is_float(parts[2]):
            continue
        freq = float(parts[0])
        z_real = float(parts[1])
        z_imag = float(parts[2])
        z_mod = float(parts[3]) if len(parts) > 3 and _is_float(parts[3]) else None
        phase = float(parts[4]) if len(parts) > 4 and _is_float(parts[4]) else None
        rows.append((freq, z_real, z_imag, z_mod, phase))
    return rows


def _looks_like_plain_numeric_eis(lines: list[str]) -> bool:
    checked = 0
    numeric = 0
    for line in lines[:12]:
        parts = _split_numeric_line(line)
        if len(parts) >= 3:
            checked += 1
            if _is_float(parts[0]) and _is_float(parts[1]) and _is_float(parts[2]):
                numeric += 1
    return checked > 0 and numeric == checked
