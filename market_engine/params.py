"""Engine parameters.

Every tunable lives here, and the whole set is hashed into a fingerprint that
is stamped on each emitted context. Two contexts with different fingerprints
are not comparable, which matters once the dataset is being generated in
batches over weeks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/engine.yaml")

#: Bars per year, used to annualise realized volatility. Intraday figures
#: assume a ~6.5-hour session; they are a scaling convention, not a claim
#: about any instrument's real trading hours. NSE runs 09:15-15:30 IST (6.25h,
#: ~250 sessions), so hourly realized volatility on Indian symbols is scaled
#: about 4% high. That is a constant factor across every NSE bar, so it does
#: not distort comparisons within the dataset; it is recorded here rather than
#: special-cased because per-exchange calendars are not worth the machinery
#: for a cosmetic scaling difference.
BARS_PER_YEAR: dict[str, float] = {
    "1m": 98_280,
    "5m": 19_656,
    "15m": 6_552,
    "30m": 3_276,
    "60m": 1_638,
    "1h": 1_638,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}

#: Which higher timeframe to derive for a given base interval. The value is a
#: pandas resample rule.
HTF_RULE: dict[str, str] = {
    "5m": "1h",
    "15m": "4h",
    "30m": "1D",
    "60m": "1D",
    "1h": "1D",
    "1d": "1W",
    "1wk": "1ME",
}


@dataclass(frozen=True)
class EngineParams:
    # --- indicators -----------------------------------------------------
    ema_periods: tuple[int, ...] = (5, 20, 50, 100, 200)
    rsi_period: int = 14
    atr_period: int = 14
    return_periods: tuple[int, ...] = (1, 5, 20)
    realized_vol_window: int = 20
    volume_sma_window: int = 20

    # --- swings ---------------------------------------------------------
    #: Fractal half-width. A swing high needs `swing_lookback` strictly lower
    #: highs to its left and that many non-higher highs to its right, so it is
    #: only knowable `swing_lookback` bars after it printed. Larger = fewer,
    #: more significant swings, but a longer blind spot at the right edge.
    swing_lookback: int = 3
    #: Two swings are "equal" (EQH/EQL rather than HH/LL) when they differ by
    #: less than this multiple of ATR. Without a tolerance, a one-tick
    #: difference would be reported as a higher high, which is noise.
    equal_level_atr: float = 0.15

    # --- levels ---------------------------------------------------------
    #: Swing prices within this multiple of ATR collapse into one level.
    level_cluster_atr: float = 0.5
    level_min_touches: int = 2
    max_levels_per_side: int = 4

    # --- regime ---------------------------------------------------------
    #: Trailing window for the ATR percentile. Expanding until this many bars
    #: exist, so early bars are ranked against what was actually available.
    atr_percentile_window: int = 252
    atr_percentile_min_periods: int = 60
    #: Kaufman efficiency ratio window: net move divided by total path length.
    efficiency_window: int = 20
    #: ER above this is trending, below `range_er_max` is ranging, between is
    #: mixed. Calibrated in docs/market_engine.md against real data.
    trend_er_min: float = 0.35
    range_er_max: float = 0.20

    # --- setups ---------------------------------------------------------
    #: A structure break older than this many bars no longer counts as
    #: "recent" when evaluating reversal setups.
    event_recency_bars: int = 10
    #: How close price must be to a level to count as "at" it.
    level_proximity_atr: float = 0.5
    #: Which EMAs the trend-continuation rules use as the pullback band and the
    #: structural line. Configurable rather than hard-coded: a rule referring to
    #: an EMA that was not computed would silently never fire, which is the
    #: worst kind of bug because the output stays valid and just goes quiet.
    setup_fast_ema: int = 20
    setup_slow_ema: int = 50

    # --- context payload -------------------------------------------------
    #: Candles included verbatim in the emitted context. This is the single
    #: biggest driver of teacher prompt cost.
    ohlcv_window_bars: int = 60
    max_swings_in_context: int = 8

    # --- rounding --------------------------------------------------------
    price_decimals: int = 4
    ratio_decimals: int = 4

    extra: dict[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("setup_fast_ema", "setup_slow_ema"):
            period = getattr(self, name)
            if period not in self.ema_periods:
                raise ValueError(
                    f"{name}={period} is not in ema_periods={list(self.ema_periods)}; "
                    "the trend setup rules would never fire"
                )
        if self.setup_fast_ema >= self.setup_slow_ema:
            raise ValueError("setup_fast_ema must be shorter than setup_slow_ema")

    def fingerprint(self) -> str:
        """Stable hash of every value that affects output."""
        payload = {k: v for k, v in asdict(self).items() if k != "extra"}
        canonical = json.dumps(payload, sort_keys=True, default=list)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def bars_per_year(self, interval: str) -> float:
        return BARS_PER_YEAR.get(interval, 252.0)

    def htf_rule(self, interval: str) -> str | None:
        return HTF_RULE.get(interval)

    @property
    def max_ema_period(self) -> int:
        return max(self.ema_periods)


def load_params(path: Path | str = DEFAULT_CONFIG) -> EngineParams:
    p = Path(path)
    if not p.exists():
        return EngineParams()
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    kwargs: dict[str, object] = {}
    for key, value in doc.items():
        if key in {"ema_periods", "return_periods"}:
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return EngineParams(**kwargs)  # type: ignore[arg-type]
