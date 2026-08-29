# Market engine — design notes

> Deterministic description of historical price data for educational and
> research use only. Not trading advice, not a prediction, not a signal.

The engine turns validated candles into a versioned `MarketContext` JSON
object. Field-by-field definitions are generated into
[market_context.md](market_context.md) from the pydantic models; this document
covers the decisions behind them.

## The one property that matters: causality

Every value in a context for bar `t` is a function of bars `0..t` only.

This is enforced, not promised. `tests/test_engine_causality.py` runs
**truncation invariance**: for bars spread across every fetched file, it computes
the context with the full series and again with every bar after `t` deleted,
and requires the two to be identical. One assertion covers every feature that
exists and every feature added later.

The test was verified to have teeth by injecting two leaks and confirming both
were caught:

| injected leak | detected |
|---|---|
| swings revealed at print time instead of `index + lookback` | yes |
| `rolling(21, center=True)` substituted for the 20 EMA | yes |

Two further guards: a static check rejects `center=True` and negative
`.shift(-n)` anywhere in the package, and the higher-timeframe layer exposes
only fully closed periods.

### Why swings are the usual leak

A fractal swing high at bar `i` is defined by `k` bars on *both* sides, so it
cannot be known until bar `i + k`. Every `SwingPoint` therefore carries two
indices — `index` (when it printed) and `confirmed_index` (when it became
knowable) — and context assembly filters on the second. Libraries that report
only the first make every downstream backtest look excellent and be worthless.

The cost is a genuine blind spot: the engine cannot see the last `k` bars'
worth of structure. That is not a bug to fix, it is what not cheating costs.

## Definitions

### Swings

A swing high at `i` with half-width `k`:

```
high[i] >  high[i-j]   for j = 1..k     (strictly higher than the left)
high[i] >= high[i+j]   for j = 1..k     (not exceeded on the right)
```

Strict left, non-strict right resolves plateaus deterministically: in a run of
equal highs the leftmost bar is the swing and the rest are not. Strict on both
sides would silently drop any swing with a matching neighbour; non-strict on
both would report all of them.

### HH / HL / LH / LL

Each swing is compared to the previous swing of the same kind. Differences
smaller than `equal_level_atr × ATR` are labelled `EQH`/`EQL` rather than a
higher high or lower low.

The tolerance is in ATR, not in price or percent, because a 20-cent difference
is a meaningful new high on a quiet instrument and pure noise on a volatile
one — and the same instrument is both, at different times.

### Trend state

From the most recent swing of each kind, using labels so the equality
tolerance carries through:

| highs | lows | state |
|---|---|---|
| HH | HL | `UPTREND` |
| LH | LL | `DOWNTREND` |
| HH | LL | `EXPANSION` — broadening, neither side in control |
| LH | HL | `CONTRACTION` — coiling into a range |
| EQH or EQL on either side | | `RANGE` |

### BOS vs CHoCH

The same geometric test; the prevailing bias is what distinguishes them.

- **BOS** — close beyond the last confirmed swing *in the direction of the
  prevailing bias*. Continuation.
- **CHoCH** — close beyond it *against* the prevailing bias. First objective
  evidence that control may have changed.

Two rules keep the stream honest:

1. Breaks are measured on the **close**, not the wick. A wick through a level
   that closes back inside is a rejection of that level, not a break of it.
2. A broken level is **consumed**. Without this an uptrend emits a BOS on
   every bar it spends above the old high, and the real events drown in
   duplicates. A new break requires a newly confirmed swing.

A real sequence from INFY.NS, late 2025, showing all three states:

```
2025-09-22  BOS_BEARISH    breaks 1501.20  (bias not yet bullish -> continuation)
2025-10-23  CHOCH_BULLISH  breaks 1521.70  (bias was BEARISH     -> change of character)
2025-11-12  BOS_BULLISH    breaks 1546.00  (bias now BULLISH     -> continuation)
```

The middle event is the one that matters: identical geometry to the other two —
a close above the last confirmed swing high — but the prevailing bias was
bearish, so it is a change of character rather than a continuation.

### Support and resistance

Confirmed swing prices are clustered greedily: swings within
`level_cluster_atr × ATR` of a running cluster mean join it. Greedy rather than
k-means because it is deterministic, needs no `k`, and runs left to right,
which keeps truncation invariance trivially true.

`side` is assigned relative to the **current** price, not to how the level
formed. A broken ceiling becomes a floor — that is the whole idea of polarity,
and freezing a level as "resistance" because it began as a swing high would
misdescribe about half of them.

### Volatility regime and trend regime

Two orthogonal questions, deliberately not merged. A market can be quiet and
trending, or violent and going nowhere.

- **Volatility** — trailing percentile rank of ATR% within its own history.
  `LOW` <25th, `NORMAL` 25–75, `HIGH` 75–90, `EXTREME` 90+. Quartile-based, so
  an instrument spends about 10% of its life in `EXTREME` by construction,
  which is what the word should mean.
- **Trend** — Kaufman efficiency ratio: net displacement over total path
  length across the window. 1.0 is a straight line, near 0 is a round trip.
  One parameter, no fitted model, unlike ADX (its own smoothing chain) or
  regression R² (needs a functional form).

### Setups

Rule-based, descriptive, **not signals**. Each rule states every condition it
tested, and a setup is emitted only when all hold. Rules that came close are
reported separately as near misses, so the teacher can see what was *almost*
true without being told it was true.

Setup rules reference EMA periods by configuration (`setup_fast_ema`,
`setup_slow_ema`), and the engine refuses to start if those periods are not in
`ema_periods`. An earlier version hard-coded 20 and 50: with a config that
computed different periods, every trend rule silently never fired and the
output stayed perfectly valid. Failing loudly at construction is much better
than going quiet.

## Higher timeframe

The base frame is resampled and **the same engine** runs over the result, so
higher-timeframe structure is computed by exactly the same code as the base.

The trap this avoids: resampling daily bars to weekly and reading "this week's"
bar on a Wednesday hands you a bar built from Monday through Friday — two days
of the future. Each higher-timeframe bar therefore carries the instant its
period *ends*, and only periods that have fully elapsed by the current bar are
visible. The view is up to one higher-timeframe period stale; that staleness is
the price of not cheating.

## Calibration against real data

Measured over 2,747 sampled bars across the six NSE daily symbols, with the
defaults in `configs/engine.yaml`:

| structure state | share | | volatility | share | | trend regime | share |
|---|---|---|---|---|---|---|---|
| UPTREND | 30.9% | | NORMAL | 45.5% | | RANGING | 47.7% |
| DOWNTREND | 21.4% | | LOW | 31.5% | | MIXED | 27.3% |
| RANGE | 16.1% | | HIGH | 12.7% | | TRENDING | 25.0% |
| CONTRACTION | 16.0% | | EXTREME | 10.3% | | | |
| EXPANSION | 15.7% | | | | | | |

All five structure states, all four volatility regimes and all three trend
regimes are well populated, which matters because the evaluation suite later
needs enough examples in each category to report per-category accuracy.

**These figures barely moved when the universe changed from six US
instruments to six Indian equities** — the previous set gave 32.8 / 22.5 /
15.9 / 15.4 / 13.4 on structure and near-identical volatility shares. The
thresholds are relative to each instrument's own history rather than to
absolute levels, so they transfer across markets without retuning. That is
evidence the design is right, and it is the reason the volatility percentile
is a trailing rank rather than a fixed ATR cutoff.

Setup firing rates over the same sample — no setup on 66.5% of bars, and every
rule reachable:

| rule | share of bars |
|---|---|
| REVERSAL_LONG | 12.9% |
| REVERSAL_SHORT | 10.3% |
| TREND_CONTINUATION_LONG | 4.2% |
| VOLATILITY_BREAKOUT_LONG | 3.4% |
| TREND_CONTINUATION_SHORT | 2.1% |
| RANGE_FADE_SHORT | 2.1% |
| RANGE_FADE_LONG | 1.5% |

A test asserts every rule stays reachable on real data. A rule that never
fires is dead code, and unit tests on hand-built fixtures cannot detect it.

## Payload size and cost

The context is the teacher's prompt payload, so feature design *is* cost
design. Measured on INFY.NS daily, with the candle window dominating:

| `ohlcv_window_bars` | total chars | ≈ tokens | candles' share |
|---|---|---|---|
| 20 | 6,203 | 1,550 | 34% |
| 40 | 8,292 | 2,073 | 50% |
| 60 (default) | 10,378 | 2,594 | 60% |
| 100 | 14,552 | 3,638 | 72% |

60 bars is the default: enough for a model to see the swing structure it is
being told about, before the candle list starts crowding out the derived
features. Halving it to 30 would cut input cost by roughly a third — worth
revisiting once the teacher benchmark has real per-token prices to trade
against.

## Gotchas hit while building this

- **`pandas.ewm(adjust=False)` seeds from the first observation, not an SMA.**
  For a 200-period EMA the value is 0.4% wrong at the bar where it is first
  published and takes about 600 bars to fall below 0.01% (measured on INFY.NS;
  it was 0.8% on the previous US universe — the size depends on how far the
  first price sits from the first two hundred, so it varies by series, but the
  ~600-bar decay does not).
  Charting platforms and Wilder both seed with the simple average, so the
  engine implements its own SMA-seeded recursion. A teacher confidently
  reporting "the 200 EMA is at 204.4" when every chart says 206.0 is
  fabricating a number as far as any reader is concerned.
- **Wilder smoothing is `alpha = 1/n`, not the `2/(n+1)` EMA.** Using the wrong
  one shifts RSI by several points.
- **True range must include the gap terms.** An instrument that gapped 5% and
  then traded in a narrow band did not have a quiet day.
- **Warm-up values are omitted, not nulled.** An EMA-200 computed from 12 bars
  is not an EMA-200, and a `null` in the payload invites a model to read it as
  zero. Fields simply do not appear until they are real.
- **A pullback rule needs a shallow pullback.** The first synthetic fixture
  used 9% retracements, fired nothing, and looked like a broken rule; it was a
  broken fixture. Rule calibration has to be checked against real data.
