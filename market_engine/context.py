"""Assembly: candles in, MarketContext out.

The engine is split in two on purpose.

``MarketEngine.compute(frame)`` runs every left-to-right pass once - indicators,
swings, structure - and caches the result. ``context_at(i)`` then slices that
result for one bar. Recomputing everything per bar would be O(n^2) and would
make generating tens of thousands of training examples painfully slow.

That split is only safe because every cached series is causal: the value at
bar ``i`` never depends on a bar after ``i``. If that ever stops being true,
the truncation-invariance test fails loudly rather than silently poisoning the
dataset.

Higher timeframe
----------------
The trap: resampling daily bars to weekly and reading "this week's" bar on a
Wednesday hands you a bar built from Monday through Friday - two days of the
future. Here every higher-timeframe bar carries the timestamp at which its
period *ends*, and only bars whose period has fully elapsed by the current bar
are visible. The cost is that the HTF view is up to one HTF period stale,
which is the honest price of not cheating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.schema import (
    CLOSE_COL,
    DATETIME_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)
from market_engine.indicators import compute_indicators, has_usable_volume
from market_engine.levels import build_levels
from market_engine.params import EngineParams
from market_engine.regime import (
    atr_percentile,
    classify_trend_regime,
    classify_volatility,
    efficiency_ratio,
)
from market_engine.schema import (
    Candle,
    HigherTimeframeSnapshot,
    IndicatorSnapshot,
    LevelRef,
    MarketContext,
    RegimeSnapshot,
    SetupRef,
    StructureEventRef,
    StructureSnapshot,
    SwingRef,
    VolatilitySnapshot,
)
from market_engine.setups import SetupInput, evaluate_setups
from market_engine.structure import StructureEvent, StructureSeries, analyse_structure
from market_engine.swings import SwingKind, SwingPoint, detect_swings, label_swings


class EngineError(RuntimeError):
    pass


@dataclass
class ComputedFrame:
    """Everything the engine derived from one frame, in bar order."""

    symbol: str
    interval: str
    frame: pd.DataFrame
    swings: list[SwingPoint]
    structure: StructureSeries
    atr_percentile: pd.Series
    efficiency: pd.Series
    has_volume: bool
    #: Set for the base timeframe only; None when this frame *is* a higher
    #: timeframe, which stops the recursion at one level.
    htf: ComputedFrame | None = None
    htf_rule: str | None = None
    #: For each higher-timeframe bar, the instant at which its period closes.
    htf_period_end: pd.Series | None = None

    @property
    def n(self) -> int:
        return len(self.frame)


def _f(value) -> float | None:
    """None for anything not finite, so warm-up NaNs never reach the JSON."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class MarketEngine:
    def __init__(self, params: EngineParams | None = None) -> None:
        self.params = params or EngineParams()

    # ------------------------------------------------------------- compute

    def compute(
        self,
        frame: pd.DataFrame,
        symbol: str,
        interval: str,
        with_htf: bool = True,
    ) -> ComputedFrame:
        if DATETIME_COL not in frame.columns:
            raise EngineError("frame must carry a 'datetime' column")
        if frame.empty:
            raise EngineError("frame is empty")

        p = self.params
        enriched = compute_indicators(frame.reset_index(drop=True), p, interval)

        raw_swings = detect_swings(enriched, p.swing_lookback)
        swings = label_swings(raw_swings, enriched["atr"], p.equal_level_atr)
        structure = analyse_structure(enriched, swings)

        computed = ComputedFrame(
            symbol=symbol,
            interval=interval,
            frame=enriched,
            swings=swings,
            structure=structure,
            atr_percentile=atr_percentile(
                enriched["atr_pct"], p.atr_percentile_window, p.atr_percentile_min_periods
            ),
            efficiency=efficiency_ratio(enriched[CLOSE_COL], p.efficiency_window),
            has_volume=has_usable_volume(enriched),
        )

        if with_htf:
            rule = p.htf_rule(interval)
            if rule:
                htf_frame, period_end = resample_ohlcv(enriched, rule)
                if len(htf_frame) > 2 * p.swing_lookback + 1:
                    computed.htf = self.compute(
                        htf_frame, symbol, rule, with_htf=False
                    )
                    computed.htf_rule = rule
                    computed.htf_period_end = period_end
        return computed

    # ------------------------------------------------------------- context

    def context_at(self, computed: ComputedFrame, i: int) -> MarketContext:
        if not 0 <= i < computed.n:
            raise EngineError(f"bar index {i} out of range for {computed.n} bars")

        p = self.params
        f = computed.frame
        row = f.iloc[i]
        close = float(row[CLOSE_COL])
        atr_value = _f(row.get("atr"))
        timestamps = f[DATETIME_COL]

        confirmed = [s for s in computed.swings if s.confirmed_index <= i]

        return MarketContext(
            params_fingerprint=p.fingerprint(),
            symbol=computed.symbol,
            interval=computed.interval,
            as_of=_iso(timestamps.iloc[i]),
            bar_index=i,
            bars_available=i + 1,
            ohlcv_window=self._window(computed, i),
            indicators=self._indicators(computed, i, close, atr_value),
            structure=self._structure(computed, i, confirmed),
            levels=self._levels(confirmed, close, atr_value),
            volatility=self._volatility(computed, i, atr_value, close),
            regime=self._regime(computed, i),
            higher_timeframe=self._htf(computed, timestamps.iloc[i]),
            **self._setups(computed, i, close, atr_value, confirmed),
        )

    # ----------------------------------------------------------- sections

    def _window(self, computed: ComputedFrame, i: int) -> list[Candle]:
        p = self.params
        start = max(0, i - p.ohlcv_window_bars + 1)
        chunk = computed.frame.iloc[start : i + 1]
        volume = computed.has_volume
        return [
            Candle(
                t=_iso(r[DATETIME_COL]),
                o=round(float(r[OPEN_COL]), p.price_decimals),
                h=round(float(r[HIGH_COL]), p.price_decimals),
                l=round(float(r[LOW_COL]), p.price_decimals),
                c=round(float(r[CLOSE_COL]), p.price_decimals),
                v=round(float(r[VOLUME_COL]), 2) if volume else None,
            )
            for _, r in chunk.iterrows()
        ]

    def _indicators(
        self, computed: ComputedFrame, i: int, close: float, atr_value: float | None
    ) -> IndicatorSnapshot:
        p = self.params
        row = computed.frame.iloc[i]

        emas: dict[str, float] = {}
        distances: dict[str, float] = {}
        for period in p.ema_periods:
            value = _f(row.get(f"ema_{period}"))
            if value is None:
                continue
            emas[str(period)] = round(value, p.price_decimals)
            if atr_value:
                distances[str(period)] = round((close - value) / atr_value, p.ratio_decimals)

        returns = {}
        for period in p.return_periods:
            value = _f(row.get(f"return_{period}"))
            if value is not None:
                returns[str(period)] = round(value, p.ratio_decimals)

        snapshot = IndicatorSnapshot(
            ema=emas,
            rsi=_round(_f(row.get("rsi")), 2),
            atr=_round(atr_value, p.price_decimals),
            atr_pct=_round(_f(row.get("atr_pct")), 6),
            returns=returns,
            realized_volatility=_round(_f(row.get("realized_vol")), p.ratio_decimals),
            close_vs_ema=distances,
        )
        if computed.has_volume:
            snapshot.volume_ratio = _round(_f(row.get("volume_ratio")), p.ratio_decimals)
            snapshot.volume_zscore = _round(_f(row.get("volume_zscore")), p.ratio_decimals)
        return snapshot

    def _structure(
        self, computed: ComputedFrame, i: int, confirmed: list[SwingPoint]
    ) -> StructureSnapshot:
        p = self.params
        s = computed.structure
        recent = confirmed[-p.max_swings_in_context :]

        events = [e for e in s.events if e.index <= i]
        last = events[-1] if events else None
        active_high = s.active_high[i]
        active_low = s.active_low[i]

        return StructureSnapshot(
            trend=s.trend[i].value,
            bias=s.bias[i].value,
            swings=[
                SwingRef(
                    index=sw.index,
                    timestamp=_iso(sw.timestamp),
                    price=round(sw.price, p.price_decimals),
                    kind=sw.kind.value,
                    label=sw.label.value,
                    confirmed_index=sw.confirmed_index,
                    bars_ago=i - sw.index,
                )
                for sw in sorted(recent, key=lambda x: x.index)
            ],
            last_event=_event_ref(last, i, p.price_decimals),
            recent_events=[
                ref
                for ref in (_event_ref(e, i, p.price_decimals) for e in events[-3:])
                if ref is not None
            ],
            active_swing_high=(
                round(active_high.price, p.price_decimals) if active_high else None
            ),
            active_swing_low=(round(active_low.price, p.price_decimals) if active_low else None),
        )

    def _levels(
        self, confirmed: list[SwingPoint], close: float, atr_value: float | None
    ) -> list[LevelRef]:
        p = self.params
        if atr_value is None:
            return []
        levels = build_levels(
            confirmed,
            close,
            atr_value,
            p.level_cluster_atr,
            p.level_min_touches,
            p.max_levels_per_side,
        )
        return [
            LevelRef(
                price=round(lv.price, p.price_decimals),
                side=lv.side,
                touches=lv.touches,
                distance_atr=round(lv.distance_atr, p.ratio_decimals),
                last_index=lv.last_index,
            )
            for lv in levels
        ]

    def _volatility(
        self, computed: ComputedFrame, i: int, atr_value: float | None, close: float
    ) -> VolatilitySnapshot:
        pct = _f(computed.atr_percentile.iloc[i])
        return VolatilitySnapshot(
            atr=_round(atr_value, self.params.price_decimals),
            atr_pct=_round(_f(computed.frame["atr_pct"].iloc[i]), 6),
            atr_percentile=_round(pct, 4),
            regime=classify_volatility(pct).value,
        )

    def _regime(self, computed: ComputedFrame, i: int) -> RegimeSnapshot:
        p = self.params
        er = _f(computed.efficiency.iloc[i])
        return RegimeSnapshot(
            trend_regime=classify_trend_regime(er, p.trend_er_min, p.range_er_max).value,
            efficiency_ratio=_round(er, 4),
        )

    def _htf(
        self, computed: ComputedFrame, now: pd.Timestamp
    ) -> HigherTimeframeSnapshot | None:
        if computed.htf is None or computed.htf_period_end is None:
            return None
        closed = computed.htf_period_end <= now
        count = int(closed.sum())
        if count == 0:
            return None

        j = count - 1  # most recent fully closed higher-timeframe bar
        htf = computed.htf
        p = self.params
        row = htf.frame.iloc[j]
        events = [e for e in htf.structure.events if e.index <= j]

        emas = {}
        for period in p.ema_periods:
            value = _f(row.get(f"ema_{period}"))
            if value is not None:
                emas[str(period)] = round(value, p.price_decimals)

        return HigherTimeframeSnapshot(
            interval=computed.htf_rule or htf.interval,
            bars_available=count,
            as_of=_iso(row[DATETIME_COL]),
            trend=htf.structure.trend[j].value,
            bias=htf.structure.bias[j].value,
            close=round(float(row[CLOSE_COL]), p.price_decimals),
            ema=emas,
            last_event=_event_ref(events[-1] if events else None, j, p.price_decimals),
        )

    def _setups(
        self,
        computed: ComputedFrame,
        i: int,
        close: float,
        atr_value: float | None,
        confirmed: list[SwingPoint],
    ) -> dict[str, object]:
        p = self.params
        if atr_value is None:
            return {"setups": [], "near_miss_setups": []}

        row = computed.frame.iloc[i]
        s = computed.structure
        events = [e for e in s.events if e.index <= i]
        last = events[-1] if events else None

        levels = build_levels(
            confirmed, close, atr_value, p.level_cluster_atr, p.level_min_touches,
            p.max_levels_per_side,
        )
        supports = [lv for lv in levels if lv.side == "SUPPORT"]
        resistances = [lv for lv in levels if lv.side == "RESISTANCE"]

        pct = _f(computed.atr_percentile.iloc[i])
        er = _f(computed.efficiency.iloc[i])

        inp = SetupInput(
            close=close,
            atr=atr_value,
            rsi=_f(row.get("rsi")),
            emas={
                period: value
                for period in p.ema_periods
                if (value := _f(row.get(f"ema_{period}"))) is not None
            },
            trend=s.trend[i],
            bias=s.bias[i],
            volatility_regime=classify_volatility(pct),
            trend_regime=classify_trend_regime(er, p.trend_er_min, p.range_er_max),
            atr_percentile=pct,
            last_event=last,
            bars_since_event=(i - last.index) if last else None,
            nearest_support=max(supports, key=lambda lv: lv.price) if supports else None,
            nearest_resistance=(
                min(resistances, key=lambda lv: lv.price) if resistances else None
            ),
            active_high=s.active_high[i],
            active_low=s.active_low[i],
            volume_ratio=_f(row.get("volume_ratio")) if computed.has_volume else None,
            event_recency_bars=p.event_recency_bars,
            level_proximity_atr=p.level_proximity_atr,
            fast_ema_period=p.setup_fast_ema,
            slow_ema_period=p.setup_slow_ema,
        )

        complete, near = evaluate_setups(inp)
        return {
            "setups": [
                SetupRef(
                    name=c.name,
                    direction=c.direction,
                    description=c.description,
                    conditions=[name for name, _ in c.conditions],
                    trigger_level=_round(c.trigger_level, p.price_decimals),
                    invalidation_level=_round(c.invalidation_level, p.price_decimals),
                )
                for c in complete
            ],
            "near_miss_setups": near,
        }


# --------------------------------------------------------------- helpers


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate to a higher timeframe, returning the bars and their end times.

    `label='left'` and `closed='left'` mean each bar is stamped with the
    instant its period *opened*, matching the base-timeframe convention that a
    candle's timestamp is its open. The separate period-end series is what
    makes causal filtering possible: a bar is usable only once its period has
    fully elapsed.

    Empty periods (weekends, holidays) are dropped rather than forward-filled;
    a week with no trading is not a bar with zero range.
    """
    indexed = frame.set_index(DATETIME_COL).sort_index()
    agg = {OPEN_COL: "first", HIGH_COL: "max", LOW_COL: "min", CLOSE_COL: "last"}
    if VOLUME_COL in indexed.columns:
        agg[VOLUME_COL] = "sum"

    resampler = indexed.resample(rule, label="left", closed="left")
    out = resampler.agg(agg).dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL])

    offset = pd.tseries.frequencies.to_offset(rule)
    period_end = pd.Series(
        [ts + offset for ts in out.index], index=range(len(out)), dtype="datetime64[ns, UTC]"
    )
    out = out.reset_index()
    if VOLUME_COL not in out.columns:
        out[VOLUME_COL] = np.nan
    return out, period_end


def _event_ref(
    event: StructureEvent | None, current_index: int, decimals: int
) -> StructureEventRef | None:
    if event is None:
        return None
    return StructureEventRef(
        type=event.type.value,
        timestamp=_iso(event.timestamp),
        level=round(event.level, decimals),
        close=round(event.close, decimals),
        bars_ago=current_index - event.index,
    )


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _round(value: float | None, decimals: int) -> float | None:
    return None if value is None else round(value, decimals)


def swing_counts(swings: list[SwingPoint]) -> dict[str, int]:
    """Small helper used by the CLI summary."""
    return {
        "highs": sum(1 for s in swings if s.kind is SwingKind.HIGH),
        "lows": sum(1 for s in swings if s.kind is SwingKind.LOW),
    }
