"""Market-structure state machine: trend state, BOS and CHoCH.

Definitions used here
---------------------
**BOS — Break of Structure.** Price closes beyond the most recent confirmed
swing *in the direction of the prevailing bias*. It is a continuation event:
the trend did what the trend was already doing.

**CHoCH — Change of Character.** Price closes beyond the most recent confirmed
swing *against* the prevailing bias. It is the first objective evidence that
control may have changed hands.

The two are the same geometric test; only the prevailing bias distinguishes
them. Getting that distinction right is the entire value of this module, so
bias is carried explicitly as state rather than re-derived per bar.

Two rules keep the event stream honest:

1. Only swings with ``confirmed_index <= t`` are visible at bar ``t``.
2. A level that has been broken is *consumed*. Without this, an uptrend would
   emit a BOS on every single bar it spent above the old high, drowning the
   real events in duplicates. A new break requires a newly confirmed swing.

Breaks are measured on the **close**, not on the wick. A wick through a level
that closes back inside is a rejection of the level, not a break of it, and
treating it as a break produces a stream of phantom reversals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from data_pipeline.schema import CLOSE_COL
from market_engine.swings import SwingKind, SwingLabel, SwingPoint


class Bias(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class TrendState(StrEnum):
    UPTREND = "UPTREND"  # higher highs and higher lows
    DOWNTREND = "DOWNTREND"  # lower highs and lower lows
    EXPANSION = "EXPANSION"  # higher highs and lower lows: broadening, no control
    CONTRACTION = "CONTRACTION"  # lower highs and higher lows: coiling into a range
    RANGE = "RANGE"  # at least one side equal within tolerance
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class EventType(StrEnum):
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"


@dataclass(frozen=True)
class StructureEvent:
    index: int
    timestamp: pd.Timestamp
    type: EventType
    #: Price of the swing that was broken.
    level: float
    #: Bar on which the broken swing printed.
    level_swing_index: int
    #: Close that did the breaking.
    close: float


@dataclass
class StructureSeries:
    """Per-bar structure state, aligned to the frame by positional index."""

    bias: list[Bias]
    trend: list[TrendState]
    events: list[StructureEvent]
    #: Index into `events` of the most recent event at or before each bar,
    #: or -1 if none yet.
    last_event_at: list[int]
    #: Most recent confirmed swing high / low visible at each bar. None until
    #: one has been confirmed.
    active_high: list[SwingPoint | None]
    active_low: list[SwingPoint | None]
    #: All swings confirmed at or before each bar, cheap to slice.
    confirmed_upto: list[int]


def analyse_structure(frame: pd.DataFrame, swings: list[SwingPoint]) -> StructureSeries:
    """Walk the frame left to right, emitting structure state per bar.

    Written as an explicit loop rather than vectorized on purpose: the state
    (bias, which level is still live) genuinely depends on the previous bar's
    state, and a left-to-right fold is exactly what makes truncation
    invariance hold by construction.
    """
    closes = frame[CLOSE_COL].to_numpy(dtype=float)
    timestamps = (
        frame.index.to_numpy()
        if isinstance(frame.index, pd.DatetimeIndex)
        else frame["datetime"].to_numpy()
    )
    n = len(frame)

    by_confirmation: dict[int, list[SwingPoint]] = {}
    for swing in swings:
        by_confirmation.setdefault(swing.confirmed_index, []).append(swing)

    bias = Bias.NEUTRAL
    known_highs: list[SwingPoint] = []
    known_lows: list[SwingPoint] = []
    # The most recent confirmed swing that has not yet been broken. `None`
    # means the last one was consumed and no newer swing has confirmed.
    live_high: SwingPoint | None = None
    live_low: SwingPoint | None = None

    series = StructureSeries(
        bias=[], trend=[], events=[], last_event_at=[], active_high=[], active_low=[],
        confirmed_upto=[],
    )
    confirmed_count = 0

    for i in range(n):
        for swing in by_confirmation.get(i, []):
            confirmed_count += 1
            if swing.kind is SwingKind.HIGH:
                known_highs.append(swing)
                live_high = swing
            else:
                known_lows.append(swing)
                live_low = swing

        close = closes[i]

        if live_high is not None and close > live_high.price:
            kind = EventType.CHOCH_BULLISH if bias is Bias.BEARISH else EventType.BOS_BULLISH
            series.events.append(
                StructureEvent(
                    index=i,
                    timestamp=pd.Timestamp(timestamps[i]),
                    type=kind,
                    level=live_high.price,
                    level_swing_index=live_high.index,
                    close=float(close),
                )
            )
            bias = Bias.BULLISH
            live_high = None  # consumed
        elif live_low is not None and close < live_low.price:
            kind = EventType.CHOCH_BEARISH if bias is Bias.BULLISH else EventType.BOS_BEARISH
            series.events.append(
                StructureEvent(
                    index=i,
                    timestamp=pd.Timestamp(timestamps[i]),
                    type=kind,
                    level=live_low.price,
                    level_swing_index=live_low.index,
                    close=float(close),
                )
            )
            bias = Bias.BEARISH
            live_low = None  # consumed

        series.bias.append(bias)
        series.trend.append(_trend_from(known_highs, known_lows))
        series.last_event_at.append(len(series.events) - 1)
        series.active_high.append(live_high)
        series.active_low.append(live_low)
        series.confirmed_upto.append(confirmed_count)

    return series


def _trend_from(highs: list[SwingPoint], lows: list[SwingPoint]) -> TrendState:
    """Classify structure from the most recent swing of each kind.

    Deliberately based on the *labels* rather than on raw prices, so the
    equality tolerance applied during labelling carries through: two highs a
    tick apart give RANGE, not UPTREND.
    """
    if not highs or not lows:
        return TrendState.INSUFFICIENT_DATA
    high_label, low_label = highs[-1].label, lows[-1].label
    if high_label is SwingLabel.FIRST or low_label is SwingLabel.FIRST:
        return TrendState.INSUFFICIENT_DATA
    if high_label is SwingLabel.EQH or low_label is SwingLabel.EQL:
        return TrendState.RANGE
    if high_label is SwingLabel.HH and low_label is SwingLabel.HL:
        return TrendState.UPTREND
    if high_label is SwingLabel.LH and low_label is SwingLabel.LL:
        return TrendState.DOWNTREND
    if high_label is SwingLabel.HH and low_label is SwingLabel.LL:
        return TrendState.EXPANSION
    if high_label is SwingLabel.LH and low_label is SwingLabel.HL:
        return TrendState.CONTRACTION
    return TrendState.RANGE
