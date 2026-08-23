"""The structured market context: the engine's output contract.

Every field carries a `description`. This is not documentation politeness -
three things depend on it:

1. The data dictionary in `docs/` is generated from this model, so it cannot
   drift from the code.
2. The teacher prompt embeds field definitions, so the model is told what
   `distance_atr` means rather than guessing.
3. The dataset quality pipeline checks the teacher's numeric claims against
   named fields, which requires the names to have agreed meanings.

A test asserts that no field is left without a description.

Nothing in this model is time-varying or environment-dependent: no
`generated_at`, no random ids, no absolute file paths. A context is a pure
function of (candles, params, bar index), so two runs produce byte-identical
JSON and the truncation-invariance test can compare with `==`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from market_engine import CONTEXT_SCHEMA_VERSION, DISCLAIMER, ENGINE_VERSION


class Candle(BaseModel):
    t: str = Field(description="Bar open time, ISO-8601 UTC.")
    o: float = Field(description="Open price.")
    h: float = Field(description="High price.")
    l: float = Field(description="Low price.")  # noqa: E741 - terse keys are deliberate
    c: float = Field(description="Close price.")
    v: float | None = Field(
        default=None, description="Volume, omitted for instruments that do not report it."
    )


class IndicatorSnapshot(BaseModel):
    ema: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Exponential moving averages keyed by period, recursive form "
            "(alpha = 2/(period+1)). Periods without enough history are omitted."
        ),
    )
    rsi: float | None = Field(
        default=None,
        description="Wilder RSI over the configured period, 0-100. Omitted during warm-up.",
    )
    atr: float | None = Field(
        default=None, description="Wilder Average True Range in price units."
    )
    atr_pct: float | None = Field(
        default=None, description="ATR divided by close: volatility as a fraction of price."
    )
    returns: dict[str, float] = Field(
        default_factory=dict,
        description="Simple returns over N bars, keyed by N, as a fraction (0.01 = +1%).",
    )
    realized_volatility: float | None = Field(
        default=None,
        description=(
            "Annualised standard deviation of log returns over the configured window, "
            "as a fraction (0.20 = 20% annualised)."
        ),
    )
    close_vs_ema: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Signed distance from close to each EMA, in ATR multiples. "
            "Negative means price is below that EMA."
        ),
    )
    volume_ratio: float | None = Field(
        default=None,
        description=(
            "Current volume divided by its moving average. 1.0 is average. "
            "Omitted entirely for instruments without real volume."
        ),
    )
    volume_zscore: float | None = Field(
        default=None,
        description="Standard deviations of current volume from its rolling mean.",
    )


class SwingRef(BaseModel):
    index: int = Field(description="Positional bar index at which the swing printed.")
    timestamp: str = Field(description="Time of the bar at which the swing printed.")
    price: float = Field(description="Swing price: the bar's high for a HIGH, low for a LOW.")
    kind: str = Field(description="HIGH or LOW.")
    label: str = Field(
        description=(
            "Relation to the previous swing of the same kind: HH, LH, HL, LL, "
            "EQH/EQL when within the equality tolerance, FIRST when there is no predecessor."
        )
    )
    confirmed_index: int = Field(
        description=(
            "First bar at which this swing could be known, equal to index + swing_lookback. "
            "No swing is ever included in a context before this bar."
        )
    )
    bars_ago: int = Field(description="Bars between the swing and the bar being described.")


class StructureEventRef(BaseModel):
    type: str = Field(
        description=(
            "BOS_BULLISH / BOS_BEARISH for a close beyond the last confirmed swing in the "
            "direction of the prevailing bias (continuation); CHOCH_BULLISH / CHOCH_BEARISH "
            "for a close beyond it against the prevailing bias (potential reversal)."
        )
    )
    timestamp: str = Field(description="Bar on which the break closed.")
    level: float = Field(description="Price of the swing that was broken.")
    close: float = Field(description="Closing price that broke it.")
    bars_ago: int = Field(description="Bars between the event and the bar being described.")


class StructureSnapshot(BaseModel):
    trend: str = Field(
        description=(
            "UPTREND (HH+HL), DOWNTREND (LH+LL), EXPANSION (HH+LL), CONTRACTION (LH+HL), "
            "RANGE (an equal high or low), or INSUFFICIENT_DATA."
        )
    )
    bias: str = Field(
        description=(
            "Directional control implied by the most recent structure break: BULLISH, "
            "BEARISH, or NEUTRAL before any break has occurred."
        )
    )
    swings: list[SwingRef] = Field(
        default_factory=list,
        description="Most recent confirmed swings, oldest first.",
    )
    last_event: StructureEventRef | None = Field(
        default=None, description="Most recent structure break at or before this bar."
    )
    recent_events: list[StructureEventRef] = Field(
        default_factory=list, description="Recent structure breaks, oldest first."
    )
    active_swing_high: float | None = Field(
        default=None,
        description=(
            "Most recent confirmed swing high that has not yet been broken by a close. "
            "Null when the last one was broken and no newer swing has confirmed."
        ),
    )
    active_swing_low: float | None = Field(
        default=None,
        description="Most recent confirmed swing low that has not yet been broken by a close.",
    )


class LevelRef(BaseModel):
    price: float = Field(description="Mean price of the clustered swings forming this level.")
    side: str = Field(description="SUPPORT if below the current close, RESISTANCE if above.")
    touches: int = Field(description="Number of confirmed swings in the cluster.")
    distance_atr: float = Field(
        description="Signed distance from the current close in ATR multiples; negative is below."
    )
    last_index: int = Field(description="Bar index of the most recent swing in the cluster.")


class VolatilitySnapshot(BaseModel):
    atr: float | None = Field(default=None, description="Wilder ATR in price units.")
    atr_pct: float | None = Field(default=None, description="ATR as a fraction of close.")
    atr_percentile: float | None = Field(
        default=None,
        description=(
            "Trailing percentile rank of current ATR% within its own recent history, 0-1. "
            "Computed over a backward-looking window only."
        ),
    )
    regime: str = Field(
        description="LOW (<25th pct), NORMAL (25-75), HIGH (75-90), EXTREME (90+), or UNKNOWN."
    )


class RegimeSnapshot(BaseModel):
    trend_regime: str = Field(
        description="TRENDING, MIXED, RANGING, or UNKNOWN, from the efficiency ratio."
    )
    efficiency_ratio: float | None = Field(
        default=None,
        description=(
            "Net price displacement divided by total path length over the window, 0-1. "
            "1.0 is a straight line; near 0 is round-tripping."
        ),
    )


class SetupRef(BaseModel):
    name: str = Field(description="Rule identifier.")
    direction: str = Field(description="LONG or SHORT.")
    description: str = Field(description="Plain-language statement of the configuration.")
    conditions: list[str] = Field(
        default_factory=list,
        description="Every condition the rule required, all of which are satisfied.",
    )
    trigger_level: float | None = Field(
        default=None, description="Structural price that would confirm the pattern."
    )
    invalidation_level: float | None = Field(
        default=None, description="Structural price that would break the pattern."
    )


class HigherTimeframeSnapshot(BaseModel):
    interval: str = Field(description="Resample rule used to build the higher timeframe.")
    bars_available: int = Field(
        description="Number of fully closed higher-timeframe bars at or before this bar."
    )
    as_of: str = Field(description="Open time of the most recent fully closed higher-TF bar.")
    trend: str = Field(description="Higher-timeframe structure state, same vocabulary as below.")
    bias: str = Field(description="Higher-timeframe directional bias.")
    close: float = Field(description="Close of the most recent fully closed higher-TF bar.")
    ema: dict[str, float] = Field(
        default_factory=dict, description="Higher-timeframe EMAs keyed by period."
    )
    last_event: StructureEventRef | None = Field(
        default=None, description="Most recent higher-timeframe structure break."
    )


class MarketContext(BaseModel):
    """Everything the teacher LLM is allowed to reason from."""

    schema_version: str = Field(
        default=CONTEXT_SCHEMA_VERSION, description="Version of this context schema."
    )
    engine_version: str = Field(
        default=ENGINE_VERSION,
        description="Version of the feature engine that produced the values.",
    )
    params_fingerprint: str = Field(
        description="Short hash of every engine parameter, so runs can be compared safely."
    )
    symbol: str = Field(description="Instrument identifier.")
    interval: str = Field(description="Bar interval of the base timeframe.")
    as_of: str = Field(description="Open time of the bar being described, ISO-8601 UTC.")
    bar_index: int = Field(description="Positional index of that bar within the source file.")
    bars_available: int = Field(
        description="Bars of history at or before this bar; the engine saw no others."
    )

    ohlcv_window: list[Candle] = Field(
        default_factory=list,
        description="Most recent candles up to and including the bar being described.",
    )
    indicators: IndicatorSnapshot = Field(description="Indicator values at this bar.")
    structure: StructureSnapshot = Field(description="Price-action structure at this bar.")
    levels: list[LevelRef] = Field(
        default_factory=list,
        description="Support and resistance clusters nearest the current close.",
    )
    volatility: VolatilitySnapshot = Field(description="Volatility state at this bar.")
    regime: RegimeSnapshot = Field(description="Trend-versus-range state at this bar.")
    higher_timeframe: HigherTimeframeSnapshot | None = Field(
        default=None,
        description=(
            "Context from the next timeframe up, built only from fully closed higher-TF bars."
        ),
    )
    setups: list[SetupRef] = Field(
        default_factory=list,
        description=(
            "Rule-based configurations whose conditions are all satisfied. Descriptive only: "
            "not signals, not predictions, not recommendations."
        ),
    )
    near_miss_setups: list[str] = Field(
        default_factory=list,
        description="Rules that were close to firing, with the conditions they failed.",
    )
    disclaimer: str = Field(
        default=DISCLAIMER, description="Educational and research use only."
    )
