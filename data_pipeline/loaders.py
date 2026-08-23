"""Strict CSV loading.

The loader does exactly three things: read bytes, parse types, normalise the
timezone to UTC. It does *not* sort, deduplicate, fill, or drop anything —
those are precisely the defects the validators are looking for, and repairing
them here would make the gate blind.

Unparseable numeric cells become NaN (pandas' behaviour) and are then caught by
the missing-values check, rather than raising and hiding the rest of the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_pipeline.schema import CANONICAL_TZ, DATETIME_COL, PRICE_COLS, VOLUME_COL


@dataclass(frozen=True)
class LoadedFrame:
    frame: pd.DataFrame
    sha256: str
    path: Path
    #: Timezone string observed in the file before UTC normalisation, or None
    #: if the parsed datetimes were naive.
    original_tz: str | None
    #: True when the raw column contained a mix of tz-aware and naive values,
    #: or multiple offsets that pandas could not resolve to one tz.
    mixed_timezones: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ohlcv_csv(path: Path) -> LoadedFrame:
    path = Path(path)
    digest = sha256_file(path)
    raw = pd.read_csv(path)

    original_tz: str | None = None
    mixed = False

    if DATETIME_COL in raw.columns:
        parsed = pd.to_datetime(raw[DATETIME_COL], errors="coerce", format="mixed", utc=False)
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            original_tz = str(parsed.dt.tz)
            parsed = parsed.dt.tz_convert(CANONICAL_TZ)
        elif parsed.dtype == object:
            # Mixed offsets or a mix of naive/aware values: pandas keeps object
            # dtype. Record it as a finding rather than forcing a conversion.
            mixed = True
            parsed = pd.to_datetime(raw[DATETIME_COL], errors="coerce", utc=True)
            original_tz = "MIXED"
        else:
            original_tz = None  # naive
        raw[DATETIME_COL] = parsed

    for col in (*PRICE_COLS, VOLUME_COL):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return LoadedFrame(
        frame=raw,
        sha256=digest,
        path=path,
        original_tz=original_tz,
        mixed_timezones=mixed,
    )
