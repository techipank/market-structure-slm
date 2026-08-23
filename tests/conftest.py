"""Synthetic fixtures.

Every corrupt-data test builds its frame by taking a known-good series and
introducing exactly one defect. That keeps each test's failure message
diagnostic: if two checks fire, the defect leaked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.loaders import LoadedFrame
from data_pipeline.schema import (
    CLOSE_COL,
    DATETIME_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)


def make_clean_daily(n: int = 120, start: str = "2024-01-01") -> pd.DataFrame:
    """A gentle, deterministic random walk on business days.

    Deterministic seed: the whole point of the gate is reproducibility, and a
    flaky fixture would undermine the abnormal-gap test in particular.
    """
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, periods=n, tz="UTC")
    steps = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.0008, size=n))
    spread = np.abs(rng.normal(0.004, 0.001, size=n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {
            DATETIME_COL: idx,
            OPEN_COL: open_,
            HIGH_COL: high,
            LOW_COL: low,
            CLOSE_COL: close,
            VOLUME_COL: rng.integers(1_000_000, 5_000_000, size=n).astype(float),
        }
    )


def as_loaded(
    frame: pd.DataFrame,
    original_tz: str | None = "UTC",
    mixed: bool = False,
    path: str = "synthetic.csv",
) -> LoadedFrame:
    from pathlib import Path

    return LoadedFrame(
        frame=frame,
        sha256="0" * 64,
        path=Path(path),
        original_tz=original_tz,
        mixed_timezones=mixed,
    )


@pytest.fixture
def clean_daily() -> pd.DataFrame:
    return make_clean_daily()


def write_csv(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False)
