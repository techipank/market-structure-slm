"""The OHLCV data contract.

This is the single source of truth for what a valid candle file looks like.
Everything downstream (market engine, teacher prompts, datasets) may assume
these guarantees hold *only* for files whose validation report is PASS.

Field definitions
-----------------
datetime : timezone-aware UTC instant marking the OPEN of the candle interval.
open     : first traded price in the interval.
high     : maximum traded price in the interval.
low      : minimum traded price in the interval.
close    : last traded price in the interval.
volume   : units traded during the interval. May be 0 (illiquid / synthetic
           sessions) but never negative. May be absent for some FX sources;
           absence is handled by the caller, not by fabricating values.
"""

from __future__ import annotations

DATETIME_COL = "datetime"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOLUME_COL = "volume"

PRICE_COLS: tuple[str, ...] = (OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL)
REQUIRED_COLS: tuple[str, ...] = (DATETIME_COL, *PRICE_COLS, VOLUME_COL)

#: Canonical storage timezone. All datetimes are normalised to UTC on ingest so
#: that files fetched from different machines / DST states are comparable.
CANONICAL_TZ = "UTC"

#: Intervals we support, mapped to their nominal bar duration in seconds.
#: Daily and above are *calendar* intervals: their real spacing depends on the
#: exchange trading calendar, so gap checks treat them differently.
INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "2m": 120,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1h": 3600,
    "90m": 5400,
    "1d": 86_400,
    "1wk": 604_800,
    "1mo": 2_592_000,
}

#: Intervals whose spacing is governed by a trading calendar rather than a
#: fixed clock grid. Missing-candle detection is advisory (WARNING) for these.
CALENDAR_INTERVALS: frozenset[str] = frozenset({"1d", "1wk", "1mo"})
