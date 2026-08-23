"""Volatility regime and trend-vs-range classification.

Two orthogonal questions, deliberately kept separate:

* *How much* is it moving?  -> volatility regime, from the ATR percentile.
* *How directionally* is it moving? -> trend regime, from the efficiency ratio.

A market can be quiet and trending, or violent and going nowhere. Collapsing
both into one "regime" label throws away the distinction that matters most for
describing structure.

Both measures are relative to the instrument's own recent history rather than
to any absolute threshold. An ATR of 3.0 means nothing without knowing whether
this instrument normally prints 0.5 or 30.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd


class VolatilityRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


class TrendRegime(StrEnum):
    TRENDING = "TRENDING"
    MIXED = "MIXED"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


def atr_percentile(atr_pct: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Trailing percentile rank of the current ATR% within its own history.

    ``rolling.rank(pct=True)`` includes the current observation and looks only
    backwards, so bar ``t`` is ranked against bars ``t-window+1 .. t``. That is
    the causal definition; ranking against the full series would be a textbook
    leak, since it would encode how volatile the *future* turned out to be.
    """
    return atr_pct.rolling(window, min_periods=min_periods).rank(pct=True)


def classify_volatility(percentile: float | None) -> VolatilityRegime:
    """Buckets chosen so NORMAL covers the middle half of history.

    Boundaries are quartile-based rather than tuned: 0-25 LOW, 25-75 NORMAL,
    75-90 HIGH, 90+ EXTREME. Any instrument therefore spends about 10% of its
    life in EXTREME by construction, which is the intended meaning of the word.
    """
    if percentile is None or not np.isfinite(percentile):
        return VolatilityRegime.UNKNOWN
    if percentile < 0.25:
        return VolatilityRegime.LOW
    if percentile < 0.75:
        return VolatilityRegime.NORMAL
    if percentile < 0.90:
        return VolatilityRegime.HIGH
    return VolatilityRegime.EXTREME


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio: net displacement / total path length.

    ``|close[t] - close[t-n]| / sum(|close[i] - close[i-1]|)``

    1.0 is a straight line; near 0 is a market that travelled a long way and
    ended where it started. It answers "trending or ranging?" with one
    parameter and no fitted model, unlike ADX which needs its own smoothing
    chain, or regression R-squared which needs a functional form.
    """
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def classify_trend_regime(er: float | None, trend_min: float, range_max: float) -> TrendRegime:
    if er is None or not np.isfinite(er):
        return TrendRegime.UNKNOWN
    if er >= trend_min:
        return TrendRegime.TRENDING
    if er <= range_max:
        return TrendRegime.RANGING
    return TrendRegime.MIXED
