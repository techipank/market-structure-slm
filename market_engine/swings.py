"""Fractal swing detection with explicit confirmation lag.

The whole point of this module is the distinction between two indices:

``index``            the bar where the swing actually printed
``confirmed_index``  the first bar at which you could have *known* it printed

A swing high needs ``k`` bars on each side. At the moment it prints you cannot
tell it is a high, because the next ``k`` bars have not happened. Libraries
that report the swing at ``index`` leak the future into every downstream
feature, and the resulting backtest looks wonderful and is worthless.

Everything downstream filters on ``confirmed_index <= current_bar``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from data_pipeline.schema import HIGH_COL, LOW_COL


class SwingKind(StrEnum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingLabel(StrEnum):
    """Position of a swing relative to the previous swing of the same kind."""

    HH = "HH"  # higher high
    LH = "LH"  # lower high
    HL = "HL"  # higher low
    LL = "LL"  # lower low
    EQH = "EQH"  # equal high, within tolerance
    EQL = "EQL"  # equal low, within tolerance
    FIRST = "FIRST"  # no previous swing of this kind to compare against


@dataclass(frozen=True)
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: SwingKind
    confirmed_index: int
    confirmed_timestamp: pd.Timestamp
    label: SwingLabel = SwingLabel.FIRST


def detect_swings(frame: pd.DataFrame, lookback: int) -> list[SwingPoint]:
    """Find fractal swing highs and lows.

    Definition (swing high at ``i``, half-width ``k``):
        ``high[i] > high[i-j]`` for all ``j`` in 1..k   (strictly higher left)
        ``high[i] >= high[i+j]`` for all ``j`` in 1..k  (not exceeded right)

    Strict on the left and non-strict on the right resolves plateaus
    deterministically: in a run of equal highs the leftmost bar wins, and the
    rest are not swings. Using strict comparison on both sides would silently
    drop every swing that happens to have a matching neighbour; using
    non-strict on both would report all of them.

    Swing lows are the mirror image.
    """
    highs = frame[HIGH_COL].to_numpy(dtype=float)
    lows = frame[LOW_COL].to_numpy(dtype=float)
    stamps = frame.index if isinstance(frame.index, pd.DatetimeIndex) else None
    timestamps = frame["datetime"].to_numpy() if stamps is None else frame.index.to_numpy()

    n = len(frame)
    points: list[SwingPoint] = []
    if n < 2 * lookback + 1:
        return points

    for i in range(lookback, n - lookback):
        left_h = highs[i - lookback : i]
        right_h = highs[i + 1 : i + lookback + 1]
        if np.all(highs[i] > left_h) and np.all(highs[i] >= right_h):
            points.append(_make(i, highs[i], SwingKind.HIGH, lookback, timestamps))

        left_l = lows[i - lookback : i]
        right_l = lows[i + 1 : i + lookback + 1]
        if np.all(lows[i] < left_l) and np.all(lows[i] <= right_l):
            points.append(_make(i, lows[i], SwingKind.LOW, lookback, timestamps))

    # Sort by the bar that reveals them, then by the bar they printed on. This
    # is the order the structure state machine consumes them in.
    points.sort(key=lambda p: (p.confirmed_index, p.index))
    return points


def _make(i: int, price: float, kind: SwingKind, lookback: int, timestamps) -> SwingPoint:
    confirmed = i + lookback
    return SwingPoint(
        index=i,
        timestamp=pd.Timestamp(timestamps[i]),
        price=float(price),
        kind=kind,
        confirmed_index=confirmed,
        confirmed_timestamp=pd.Timestamp(timestamps[confirmed]),
    )


def label_swings(
    points: list[SwingPoint],
    atr_series: pd.Series,
    equal_level_atr: float,
) -> list[SwingPoint]:
    """Assign HH / LH / HL / LL / EQH / EQL by comparing to the previous swing
    of the same kind.

    The equality tolerance is expressed in ATR at the time of the *later*
    swing, not in absolute price or percent. A 20-cent difference is a
    meaningful higher high on a quiet instrument and noise on a volatile one;
    only a volatility-relative threshold behaves sensibly across symbols and
    across regimes within one symbol.
    """
    atr_values = atr_series.to_numpy(dtype=float)
    labelled: list[SwingPoint] = []
    prev: dict[SwingKind, SwingPoint] = {}

    # Label in the order the swings printed, so "previous" means the previous
    # swing in price history rather than the previous confirmation.
    for point in sorted(points, key=lambda p: p.index):
        previous = prev.get(point.kind)
        if previous is None:
            label = SwingLabel.FIRST
        else:
            tol = _tolerance(atr_values, point.index, equal_level_atr)
            diff = point.price - previous.price
            if abs(diff) <= tol:
                label = SwingLabel.EQH if point.kind is SwingKind.HIGH else SwingLabel.EQL
            elif point.kind is SwingKind.HIGH:
                label = SwingLabel.HH if diff > 0 else SwingLabel.LH
            else:
                label = SwingLabel.HL if diff > 0 else SwingLabel.LL

        updated = SwingPoint(
            index=point.index,
            timestamp=point.timestamp,
            price=point.price,
            kind=point.kind,
            confirmed_index=point.confirmed_index,
            confirmed_timestamp=point.confirmed_timestamp,
            label=label,
        )
        labelled.append(updated)
        prev[point.kind] = updated

    labelled.sort(key=lambda p: (p.confirmed_index, p.index))
    return labelled


def _tolerance(atr_values: np.ndarray, index: int, multiple: float) -> float:
    value = atr_values[index] if index < len(atr_values) else np.nan
    if not np.isfinite(value):
        return 0.0
    return float(value) * multiple
