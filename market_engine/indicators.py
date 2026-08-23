"""Vectorized, causal indicator columns.

Causality here is nearly free: every primitive used (`rolling`, `shift`,
`diff`, and the left-to-right recursion below) looks strictly backwards. The
two ways to break it would be `rolling(..., center=True)` or a negative
`shift`, and a test forbids both.

Warm-up is explicit. An EMA-200 computed from 12 bars is not an EMA-200, it is
a number that looks like one, and feeding it to a teacher LLM produces
confident nonsense about a "200-period trend". Every indicator returns NaN
until it has enough history, and the context layer omits NaN fields entirely
rather than emitting nulls the model might interpret as zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.schema import CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL
from market_engine.params import EngineParams


def _seeded_recursion(series: pd.Series, alpha: float, window: int) -> pd.Series:
    """Recursive average seeded with a simple average of the first `window` values.

    ``out[t] = alpha * x[t] + (1 - alpha) * out[t-1]``, with
    ``out[window-1] = mean(x[0..window-1])`` and NaN before that.

    The seed matters more than it looks. ``pandas.ewm(adjust=False)`` seeds
    with the *single first observation*, and the error decays as
    ``(1-alpha)^k`` - fine for a 5-period average, but for a 200-period EMA it
    is still 0.8% wrong at the bar where the value is first published and does
    not fall below 0.01% for roughly 600 bars. Measured on SPY daily. Charting
    platforms and Wilder's original definition both seed with the simple
    average, and a teacher LLM quoting "the 200 EMA is at 204.4" when every
    chart says 206.0 is a fabricated number as far as the reader is concerned.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=series.index, name=series.name)

    # First position with `window` consecutive finite values behind it.
    finite = np.isfinite(values)
    run = 0
    seed_at = -1
    for i in range(n):
        run = run + 1 if finite[i] else 0
        if run >= window:
            seed_at = i
            break
    if seed_at < 0:
        return pd.Series(out, index=series.index, name=series.name)

    out[seed_at] = values[seed_at - window + 1 : seed_at + 1].mean()
    for i in range(seed_at + 1, n):
        prev = out[i - 1]
        value = values[i]
        # A NaN input propagates rather than being silently skipped: a gap in
        # the input is a gap in the output, not an invisible interpolation.
        out[i] = alpha * value + (1.0 - alpha) * prev if np.isfinite(value) else np.nan
    return pd.Series(out, index=series.index, name=series.name)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, ``alpha = 2 / (span + 1)``, SMA-seeded.

    The recursive form traders actually mean:
    ``e[t] = a * x[t] + (1 - a) * e[t-1]``. Note this is *not* pandas'
    default `adjust=True`, which computes a re-weighted average of all history
    and matches no charting platform.
    """
    return _seeded_recursion(series, 2.0 / (span + 1.0), span)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: a recursive average with ``alpha = 1 / period``.

    Wilder defined RSI and ATR with this, not with the ``2 / (n + 1)`` EMA.
    Using the wrong one shifts RSI by several points and makes every value
    disagree with the chart a reader would check it against.
    """
    return _seeded_recursion(series, 1.0 / period, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder).

    Ratio of average gain to average loss over the period, mapped to 0-100.
    Interpretation is deliberately left to the teacher; the engine only states
    the number. Note the standard degenerate case: a window with no down-closes
    has zero average loss, giving RSI exactly 100.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> rs is inf -> the formula already yields 100, but
    # 0/0 (a perfectly flat window) yields NaN, which is the honest answer.
    return out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(high-low, |high-prev_close|, |low-prev_close|).

    The two gap terms are what distinguish true range from the bar's own
    range: an instrument that gapped 5% overnight and then traded in a narrow
    band did *not* have a quiet day.
    """
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder_rma(true_range(high, low, close), period)


def realized_volatility(close: pd.Series, window: int, bars_per_year: float) -> pd.Series:
    """Annualised standard deviation of log returns.

    Log returns rather than percent returns because they are additive over
    time, which is what makes the sqrt-of-time annualisation valid at all.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(bars_per_year)


def compute_indicators(frame: pd.DataFrame, params: EngineParams, interval: str) -> pd.DataFrame:
    """Return a new frame with indicator columns appended.

    The input frame is never mutated: the market engine is a pipeline of pure
    transformations, so the same frame can be re-run with different params.
    """
    out = frame.copy()
    close, high, low = out[CLOSE_COL], out[HIGH_COL], out[LOW_COL]

    for period in params.ema_periods:
        out[f"ema_{period}"] = ema(close, period)

    out["rsi"] = rsi(close, params.rsi_period)
    out["atr"] = atr(high, low, close, params.atr_period)
    out["atr_pct"] = out["atr"] / close

    for period in params.return_periods:
        out[f"return_{period}"] = close.pct_change(period)

    out["realized_vol"] = realized_volatility(
        close, params.realized_vol_window, params.bars_per_year(interval)
    )

    if has_usable_volume(out):
        vol = out[VOLUME_COL]
        sma = vol.rolling(params.volume_sma_window, min_periods=params.volume_sma_window).mean()
        std = vol.rolling(params.volume_sma_window, min_periods=params.volume_sma_window).std(
            ddof=1
        )
        out["volume_sma"] = sma
        # Relative volume is the readable one; the z-score is the comparable
        # one across instruments with wildly different share counts.
        out["volume_ratio"] = vol / sma
        out["volume_zscore"] = (vol - sma) / std.replace(0.0, np.nan)

    return out


def has_usable_volume(frame: pd.DataFrame) -> bool:
    """False for index series like ^VIX that report a constant zero.

    Emitting `volume_ratio: NaN` for those would be noise; emitting 0 would be
    a lie. The context omits the whole volume block instead.
    """
    if VOLUME_COL not in frame.columns:
        return False
    vol = frame[VOLUME_COL]
    return bool(vol.notna().any() and (vol.fillna(0.0) > 0).any())
