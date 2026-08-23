# Market context — data dictionary

> Deterministic description of historical price data for educational and research use only. Not trading advice, not a prediction, not a signal.

Schema version `1.0.0` · engine version `1.0.0`

**Generated from `market_engine/schema.py` — do not edit by hand.**
Regenerate with `msengine schema`.

Every value below is computed deterministically from candles at or before
the bar being described. No field may depend on a later bar.

## `MarketContext`

Everything the teacher LLM is allowed to reason from.

| field | type | definition |
|---|---|---|
| `schema_version` | `str` | Version of this context schema. |
| `engine_version` | `str` | Version of the feature engine that produced the values. |
| `params_fingerprint` | `str` | Short hash of every engine parameter, so runs can be compared safely. |
| `symbol` | `str` | Instrument identifier. |
| `interval` | `str` | Bar interval of the base timeframe. |
| `as_of` | `str` | Open time of the bar being described, ISO-8601 UTC. |
| `bar_index` | `int` | Positional index of that bar within the source file. |
| `bars_available` | `int` | Bars of history at or before this bar; the engine saw no others. |
| `ohlcv_window` | `list[Candle]` | Most recent candles up to and including the bar being described. |
| `indicators` | `IndicatorSnapshot` | Indicator values at this bar. |
| `structure` | `StructureSnapshot` | Price-action structure at this bar. |
| `levels` | `list[LevelRef]` | Support and resistance clusters nearest the current close. |
| `volatility` | `VolatilitySnapshot` | Volatility state at this bar. |
| `regime` | `RegimeSnapshot` | Trend-versus-range state at this bar. |
| `higher_timeframe` | `HigherTimeframeSnapshot (optional)` | Context from the next timeframe up, built only from fully closed higher-TF bars. |
| `setups` | `list[SetupRef]` | Rule-based configurations whose conditions are all satisfied. Descriptive only: not signals, not predictions, not recommendations. |
| `near_miss_setups` | `list[str]` | Rules that were close to firing, with the conditions they failed. |
| `disclaimer` | `str` | Educational and research use only. |

## `Candle`

| field | type | definition |
|---|---|---|
| `t` | `str` | Bar open time, ISO-8601 UTC. |
| `o` | `float` | Open price. |
| `h` | `float` | High price. |
| `l` | `float` | Low price. |
| `c` | `float` | Close price. |
| `v` | `float (optional)` | Volume, omitted for instruments that do not report it. |

## `IndicatorSnapshot`

| field | type | definition |
|---|---|---|
| `ema` | `dict[str, float]` | Exponential moving averages keyed by period, recursive form (alpha = 2/(period+1)). Periods without enough history are omitted. |
| `rsi` | `float (optional)` | Wilder RSI over the configured period, 0-100. Omitted during warm-up. |
| `atr` | `float (optional)` | Wilder Average True Range in price units. |
| `atr_pct` | `float (optional)` | ATR divided by close: volatility as a fraction of price. |
| `returns` | `dict[str, float]` | Simple returns over N bars, keyed by N, as a fraction (0.01 = +1%). |
| `realized_volatility` | `float (optional)` | Annualised standard deviation of log returns over the configured window, as a fraction (0.20 = 20% annualised). |
| `close_vs_ema` | `dict[str, float]` | Signed distance from close to each EMA, in ATR multiples. Negative means price is below that EMA. |
| `volume_ratio` | `float (optional)` | Current volume divided by its moving average. 1.0 is average. Omitted entirely for instruments without real volume. |
| `volume_zscore` | `float (optional)` | Standard deviations of current volume from its rolling mean. |

## `StructureSnapshot`

| field | type | definition |
|---|---|---|
| `trend` | `str` | UPTREND (HH+HL), DOWNTREND (LH+LL), EXPANSION (HH+LL), CONTRACTION (LH+HL), RANGE (an equal high or low), or INSUFFICIENT_DATA. |
| `bias` | `str` | Directional control implied by the most recent structure break: BULLISH, BEARISH, or NEUTRAL before any break has occurred. |
| `swings` | `list[SwingRef]` | Most recent confirmed swings, oldest first. |
| `last_event` | `StructureEventRef (optional)` | Most recent structure break at or before this bar. |
| `recent_events` | `list[StructureEventRef]` | Recent structure breaks, oldest first. |
| `active_swing_high` | `float (optional)` | Most recent confirmed swing high that has not yet been broken by a close. Null when the last one was broken and no newer swing has confirmed. |
| `active_swing_low` | `float (optional)` | Most recent confirmed swing low that has not yet been broken by a close. |

## `SwingRef`

| field | type | definition |
|---|---|---|
| `index` | `int` | Positional bar index at which the swing printed. |
| `timestamp` | `str` | Time of the bar at which the swing printed. |
| `price` | `float` | Swing price: the bar's high for a HIGH, low for a LOW. |
| `kind` | `str` | HIGH or LOW. |
| `label` | `str` | Relation to the previous swing of the same kind: HH, LH, HL, LL, EQH/EQL when within the equality tolerance, FIRST when there is no predecessor. |
| `confirmed_index` | `int` | First bar at which this swing could be known, equal to index + swing_lookback. No swing is ever included in a context before this bar. |
| `bars_ago` | `int` | Bars between the swing and the bar being described. |

## `StructureEventRef`

| field | type | definition |
|---|---|---|
| `type` | `str` | BOS_BULLISH / BOS_BEARISH for a close beyond the last confirmed swing in the direction of the prevailing bias (continuation); CHOCH_BULLISH / CHOCH_BEARISH for a close beyond it against the prevailing bias (potential reversal). |
| `timestamp` | `str` | Bar on which the break closed. |
| `level` | `float` | Price of the swing that was broken. |
| `close` | `float` | Closing price that broke it. |
| `bars_ago` | `int` | Bars between the event and the bar being described. |

## `LevelRef`

| field | type | definition |
|---|---|---|
| `price` | `float` | Mean price of the clustered swings forming this level. |
| `side` | `str` | SUPPORT if below the current close, RESISTANCE if above. |
| `touches` | `int` | Number of confirmed swings in the cluster. |
| `distance_atr` | `float` | Signed distance from the current close in ATR multiples; negative is below. |
| `last_index` | `int` | Bar index of the most recent swing in the cluster. |

## `VolatilitySnapshot`

| field | type | definition |
|---|---|---|
| `atr` | `float (optional)` | Wilder ATR in price units. |
| `atr_pct` | `float (optional)` | ATR as a fraction of close. |
| `atr_percentile` | `float (optional)` | Trailing percentile rank of current ATR% within its own recent history, 0-1. Computed over a backward-looking window only. |
| `regime` | `str` | LOW (<25th pct), NORMAL (25-75), HIGH (75-90), EXTREME (90+), or UNKNOWN. |

## `RegimeSnapshot`

| field | type | definition |
|---|---|---|
| `trend_regime` | `str` | TRENDING, MIXED, RANGING, or UNKNOWN, from the efficiency ratio. |
| `efficiency_ratio` | `float (optional)` | Net price displacement divided by total path length over the window, 0-1. 1.0 is a straight line; near 0 is round-tripping. |

## `HigherTimeframeSnapshot`

| field | type | definition |
|---|---|---|
| `interval` | `str` | Resample rule used to build the higher timeframe. |
| `bars_available` | `int` | Number of fully closed higher-timeframe bars at or before this bar. |
| `as_of` | `str` | Open time of the most recent fully closed higher-TF bar. |
| `trend` | `str` | Higher-timeframe structure state, same vocabulary as below. |
| `bias` | `str` | Higher-timeframe directional bias. |
| `close` | `float` | Close of the most recent fully closed higher-TF bar. |
| `ema` | `dict[str, float]` | Higher-timeframe EMAs keyed by period. |
| `last_event` | `StructureEventRef (optional)` | Most recent higher-timeframe structure break. |

## `SetupRef`

| field | type | definition |
|---|---|---|
| `name` | `str` | Rule identifier. |
| `direction` | `str` | LONG or SHORT. |
| `description` | `str` | Plain-language statement of the configuration. |
| `conditions` | `list[str]` | Every condition the rule required, all of which are satisfied. |
| `trigger_level` | `float (optional)` | Structural price that would confirm the pattern. |
| `invalidation_level` | `float (optional)` | Structural price that would break the pattern. |
