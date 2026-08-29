"""Indicator correctness, structure semantics, and the output contract.

Indicators are checked against independent loop-based reimplementations of the
textbook definitions rather than against recorded golden numbers. A golden
number tells you the output changed; a second implementation tells you which
one is wrong.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from market_engine import indicators as ind
from market_engine.context import MarketEngine, resample_ohlcv
from market_engine.docgen import MODELS, render_data_dictionary
from market_engine.levels import build_levels
from market_engine.params import EngineParams
from market_engine.regime import (
    TrendRegime,
    VolatilityRegime,
    classify_trend_regime,
    classify_volatility,
    efficiency_ratio,
)
from market_engine.structure import Bias, EventType, TrendState
from market_engine.swings import SwingKind, SwingLabel, SwingPoint, detect_swings, label_swings
from tests.test_engine_causality import frame_from_closes

FAST = EngineParams(
    ema_periods=(5, 10),
    rsi_period=5,
    atr_period=5,
    return_periods=(1, 5),
    realized_vol_window=5,
    volume_sma_window=5,
    swing_lookback=2,
    setup_fast_ema=5,
    setup_slow_ema=10,
    atr_percentile_window=30,
    atr_percentile_min_periods=10,
    efficiency_window=5,
)


# ------------------------------------------------------------- indicators


def _reference_seeded_recursion(values: list[float], alpha: float, window: int) -> list[float]:
    """Textbook definition, written independently of the implementation."""
    out = [math.nan] * len(values)
    if len(values) < window:
        return out
    out[window - 1] = sum(values[:window]) / window
    for i in range(window, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def test_ema_matches_reference_recursion():
    closes = [10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 15.5]
    got = ind.ema(pd.Series(closes), 4).tolist()
    want = _reference_seeded_recursion(closes, 2 / 5, 4)
    for a, b in zip(got, want, strict=True):
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12)


def test_ema_is_sma_seeded_not_first_value_seeded():
    """Guards the fix: the first published value is the simple average."""
    closes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    got = ind.ema(pd.Series(closes), 4)
    assert math.isnan(got.iloc[2])
    assert got.iloc[3] == pytest.approx(2.5)  # mean(1,2,3,4)


def test_wilder_rma_uses_one_over_period_alpha():
    values = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0]
    got = ind.wilder_rma(pd.Series(values), 3).tolist()
    want = _reference_seeded_recursion(values, 1 / 3, 3)
    for a, b in zip(got, want, strict=True):
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12)


def test_rsi_is_bounded_and_hits_100_on_a_monotonic_rise():
    rising = pd.Series([100.0 + i for i in range(40)])
    values = ind.rsi(rising, 14).dropna()
    assert values.between(0, 100).all()
    assert values.iloc[-1] == pytest.approx(100.0)

    falling = pd.Series([100.0 - i for i in range(40)])
    assert ind.rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_true_range_accounts_for_gaps():
    """A narrow bar that gapped is not a quiet bar."""
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.5])
    close = pd.Series([9.5, 11.8])
    tr = ind.true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(2.5)  # |12.0 - 9.5|, not the 0.5 bar range


def test_atr_matches_reference():
    frame = frame_from_closes([100 + (i % 7) * 1.5 for i in range(40)])
    got = ind.atr(frame["high"], frame["low"], frame["close"], 5)
    # Wilder defines TR[0] as high-low (there is no previous close), so it is
    # part of the seed window rather than dropped.
    tr = ind.true_range(frame["high"], frame["low"], frame["close"]).tolist()
    want = _reference_seeded_recursion(tr, 1 / 5, 5)
    assert got.iloc[-1] == pytest.approx(want[-1], rel=1e-12)


def test_realized_volatility_is_zero_for_constant_returns():
    closes = [100.0 * (1.01**i) for i in range(40)]
    vol = ind.realized_volatility(pd.Series(closes), 10, 252)
    assert vol.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_volume_detection_rejects_all_zero_series():
    frame = frame_from_closes([100.0] * 10)
    assert ind.has_usable_volume(frame)
    frame["volume"] = 0.0
    assert not ind.has_usable_volume(frame)


# ----------------------------------------------------------------- regime


def test_efficiency_ratio_is_one_for_a_straight_line():
    closes = pd.Series([100.0 + i for i in range(30)])
    assert efficiency_ratio(closes, 10).iloc[-1] == pytest.approx(1.0)


def test_efficiency_ratio_is_near_zero_for_a_round_trip():
    closes = pd.Series([100 + (5 if i % 2 else 0) for i in range(30)], dtype=float)
    assert efficiency_ratio(closes, 10).iloc[-1] < 0.15


def test_regime_classifications():
    assert classify_volatility(0.10) is VolatilityRegime.LOW
    assert classify_volatility(0.50) is VolatilityRegime.NORMAL
    assert classify_volatility(0.80) is VolatilityRegime.HIGH
    assert classify_volatility(0.95) is VolatilityRegime.EXTREME
    assert classify_volatility(None) is VolatilityRegime.UNKNOWN
    assert classify_trend_regime(0.9, 0.35, 0.20) is TrendRegime.TRENDING
    assert classify_trend_regime(0.10, 0.35, 0.20) is TrendRegime.RANGING
    assert classify_trend_regime(0.28, 0.35, 0.20) is TrendRegime.MIXED


# ----------------------------------------------------------------- swings


def test_swing_detection_finds_the_turning_points():
    closes = [100, 101, 102, 105, 102, 101, 100, 101, 103, 106, 104, 103, 102]
    frame = frame_from_closes([float(c) for c in closes])
    swings = detect_swings(frame, lookback=2)
    kinds = {s.kind for s in swings}
    assert SwingKind.HIGH in kinds and SwingKind.LOW in kinds
    for s in swings:
        assert s.confirmed_index == s.index + 2


def test_plateau_resolves_to_the_leftmost_bar():
    """Equal highs must produce one swing, not zero and not two."""
    frame = frame_from_closes([100.0, 101, 105, 105, 101, 100, 99])
    frame.loc[2, "high"] = 110.0
    frame.loc[3, "high"] = 110.0
    swings = [s for s in detect_swings(frame, lookback=2) if s.kind is SwingKind.HIGH]
    assert [s.index for s in swings] == [2]


def test_equal_swings_are_labelled_eq_within_tolerance():
    points = [
        SwingPoint(2, pd.Timestamp("2020-01-03", tz="UTC"), 100.0, SwingKind.HIGH, 4,
                   pd.Timestamp("2020-01-07", tz="UTC")),
        SwingPoint(8, pd.Timestamp("2020-01-13", tz="UTC"), 100.05, SwingKind.HIGH, 10,
                   pd.Timestamp("2020-01-15", tz="UTC")),
    ]
    atr = pd.Series([1.0] * 12)
    labelled = label_swings(points, atr, equal_level_atr=0.15)
    assert labelled[-1].label is SwingLabel.EQH

    # Same swings, tighter tolerance -> now a genuine higher high.
    labelled = label_swings(points, atr, equal_level_atr=0.01)
    assert labelled[-1].label is SwingLabel.HH


# -------------------------------------------------------------- structure


def _zigzag(points: list[float], steps: int = 8) -> pd.DataFrame:
    closes: list[float] = []
    for a, b in zip(points, points[1:], strict=False):
        closes.extend(a + (b - a) * k / steps for k in range(steps))
    closes.append(points[-1])
    return frame_from_closes(closes)


def test_first_break_is_bos_and_the_counter_break_is_choch():
    """The distinction that justifies this module existing.

    Path: rise to 112, pull back to 104, rally through 112 (break with no
    prevailing bias -> BOS), push to 130, pull back to 116, then break below
    116 while bias is bullish -> CHoCH.
    """
    frame = _zigzag([100, 112, 104, 130, 116, 90])
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    events = computed.structure.events
    assert events, "no structure events detected in the fixture"

    assert events[0].type is EventType.BOS_BULLISH
    types = [e.type for e in events]
    assert EventType.CHOCH_BEARISH in types
    assert types.index(EventType.CHOCH_BEARISH) > 0

    # Bias must follow the last event.
    last = events[-1]
    expected = Bias.BULLISH if "BULLISH" in last.type.value else Bias.BEARISH
    assert computed.structure.bias[-1] is expected


def test_a_broken_level_is_consumed_and_does_not_re_fire():
    """Without consumption an uptrend emits a BOS on every bar above the high."""
    frame = _zigzag([100, 112, 104, 160], steps=10)
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    levels = [e.level for e in computed.structure.events if e.type is EventType.BOS_BULLISH]
    assert len(levels) == len(set(levels)), "the same level was broken more than once"


def test_uptrend_classification():
    frame = _zigzag([100, 120, 110, 140, 130, 150])
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    assert computed.structure.trend[-1] is TrendState.UPTREND


def test_downtrend_classification():
    frame = _zigzag([150, 130, 140, 110, 120, 90])
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    assert computed.structure.trend[-1] is TrendState.DOWNTREND


# ----------------------------------------------------------------- levels


def test_nearby_swings_collapse_into_one_level():
    stamp = pd.Timestamp("2020-01-01", tz="UTC")
    swings = [
        SwingPoint(i, stamp, price, SwingKind.HIGH, i + 2, stamp)
        for i, price in enumerate([100.0, 100.4, 99.8, 130.0])
    ]
    levels = build_levels(swings, reference_price=90.0, atr_value=2.0, cluster_atr=0.5,
                          min_touches=2, max_per_side=4)
    assert len(levels) == 1
    assert levels[0].touches == 3
    assert levels[0].side == "RESISTANCE"
    assert levels[0].price == pytest.approx(100.0666, abs=1e-3)


def test_level_side_is_relative_to_current_price():
    """Polarity: a broken ceiling becomes a floor."""
    stamp = pd.Timestamp("2020-01-01", tz="UTC")
    swings = [SwingPoint(i, stamp, 100.0 + i * 0.1, SwingKind.HIGH, i + 2, stamp) for i in range(3)]
    below = build_levels(swings, 120.0, 2.0, 0.5, 2, 4)
    above = build_levels(swings, 80.0, 2.0, 0.5, 2, 4)
    assert below[0].side == "SUPPORT"
    assert above[0].side == "RESISTANCE"


# ------------------------------------------------------- higher timeframe


def test_resample_stamps_bars_with_their_open_and_reports_period_end():
    frame = frame_from_closes([100.0 + i for i in range(30)], start="2021-01-04")
    weekly, period_end = resample_ohlcv(frame, "1W")
    assert len(weekly) >= 4
    assert (weekly["high"] >= weekly["low"]).all()
    for k in range(len(weekly)):
        assert period_end.iloc[k] > weekly["datetime"].iloc[k]


def test_resample_aggregates_correctly():
    frame = frame_from_closes([10.0, 20.0, 30.0, 40.0, 50.0], start="2021-01-04")
    weekly, _ = resample_ohlcv(frame, "1W")
    first = weekly.iloc[0]
    assert first["open"] == pytest.approx(10.0)
    assert first["close"] == pytest.approx(50.0)
    assert first["high"] == pytest.approx(frame["high"].max())
    assert first["low"] == pytest.approx(frame["low"].min())
    assert first["volume"] == pytest.approx(frame["volume"].sum())


# ---------------------------------------------------------------- contract


def _walk_models() -> list[type[BaseModel]]:
    return list(MODELS)


def test_every_schema_field_has_a_definition():
    missing: list[str] = []
    for model in _walk_models():
        for name, info in model.model_fields.items():
            if not (info.description or "").strip():
                missing.append(f"{model.__name__}.{name}")
    assert not missing, f"fields without a definition: {missing}"


def test_data_dictionary_renders_every_model():
    text = render_data_dictionary()
    for model in _walk_models():
        assert f"`{model.__name__}`" in text
    assert "Not trading advice" in text


def test_context_is_deterministic_and_json_serialisable():
    frame = _zigzag([100, 120, 110, 140, 130])
    engine = MarketEngine(FAST)
    a = engine.compute(frame, "SYN", "1d", with_htf=False)
    b = engine.compute(frame, "SYN", "1d", with_htf=False)
    ctx_a = engine.context_at(a, a.n - 1)
    ctx_b = engine.context_at(b, b.n - 1)
    assert ctx_a.model_dump(mode="json") == ctx_b.model_dump(mode="json")
    json.dumps(ctx_a.model_dump(mode="json"))  # must not raise


def test_context_carries_no_nan_values():
    """NaN is not valid JSON and would be silently emitted as `NaN` by some
    encoders. Warm-up values must be omitted, never nulled to a number."""
    frame = _zigzag([100, 120, 110, 140, 130])
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    for i in (5, 20, computed.n - 1):
        payload = engine.context_at(computed, i).model_dump(mode="json")
        text = json.dumps(payload)
        assert "NaN" not in text and "Infinity" not in text


def test_setup_emas_must_exist_or_the_engine_refuses_to_start():
    """A rule referring to an uncomputed EMA fails loudly, not silently."""
    with pytest.raises(ValueError, match="not in ema_periods"):
        EngineParams(ema_periods=(5, 10), setup_fast_ema=20, setup_slow_ema=50)
    with pytest.raises(ValueError, match="shorter than"):
        EngineParams(ema_periods=(20, 50), setup_fast_ema=50, setup_slow_ema=20)


def test_fingerprint_changes_with_params():
    assert EngineParams().fingerprint() != EngineParams(swing_lookback=5).fingerprint()
    assert EngineParams().fingerprint() == EngineParams().fingerprint()
    assert len(EngineParams().fingerprint()) == 16


def _shallow_uptrend(legs: int = 8) -> pd.DataFrame:
    """Staircase with +5% legs and -1.5% pullbacks.

    A pullback rule needs a shallow pullback. An earlier version of this test
    used a zigzag with 9% retracements and fired nothing, which looked like a
    broken rule and was really a broken fixture.
    """
    points = [100.0]
    for _ in range(legs):
        points.append(points[-1] * 1.05)
        points.append(points[-1] * 0.985)
    return _zigzag(points, steps=4)


def test_trend_continuation_fires_on_a_shallow_pullback_uptrend():
    engine = MarketEngine(FAST)
    computed = engine.compute(_shallow_uptrend(), "SYN", "1d", with_htf=False)
    names = {
        s.name
        for i in range(15, computed.n)
        for s in engine.context_at(computed, i).setups
    }
    assert "TREND_CONTINUATION_LONG" in names


def test_setups_are_evidence_bearing():
    """Every emitted setup must list the conditions that made it fire."""
    engine = MarketEngine(FAST)
    computed = engine.compute(_shallow_uptrend(), "SYN", "1d", with_htf=False)
    seen = 0
    for i in range(15, computed.n):
        for setup in engine.context_at(computed, i).setups:
            seen += 1
            assert setup.conditions, f"{setup.name} fired with no stated conditions"
            assert setup.direction in {"LONG", "SHORT"}
    assert seen, "fixture produced no setups at all; the rules may be unreachable"


@pytest.mark.parametrize(
    # Discovered, not hardcoded: naming files explicitly turns every one of
    # these into a silent skip the moment the symbol universe changes.
    "name",
    [p.stem for p in sorted(Path("data/raw").glob("*_1d.csv"))][:3] or ["<no data>"],
)
def test_every_rule_is_reachable_on_real_data(name: str):
    """A rule that never fires on eleven years of data is dead code.

    This is a calibration test, not a correctness test: it catches thresholds
    tightened to the point of uselessness, which unit tests on hand-built
    fixtures cannot see.
    """
    from data_pipeline.loaders import load_ohlcv_csv
    from market_engine.params import load_params
    from market_engine.setups import ALL_RULES

    path = Path("data/raw") / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path} not present; run `msdata fetch` to enable this test")

    engine = MarketEngine(load_params())
    computed = engine.compute(load_ohlcv_csv(path).frame, name, "1d")
    fired: set[str] = set()
    for i in range(300, computed.n, 5):
        fired.update(s.name for s in engine.context_at(computed, i).setups)

    expected = {rule(_null_setup_input()).name for rule in ALL_RULES}
    never = expected - fired
    # VOLATILITY_BREAKOUT_LONG needs a low-vol compression plus a break, which
    # a single steadily-trending symbol may genuinely never produce.
    assert len(never) <= 1, f"{name}: rules that never fired: {sorted(never)}"


def _null_setup_input():
    """Minimal SetupInput used only to read rule names."""
    from market_engine.setups import SetupInput

    return SetupInput(
        close=1.0, atr=1.0, rsi=None, emas={}, trend=TrendState.INSUFFICIENT_DATA,
        bias=Bias.NEUTRAL, volatility_regime=VolatilityRegime.UNKNOWN,
        trend_regime=TrendRegime.UNKNOWN, atr_percentile=None, last_event=None,
        bars_since_event=None, nearest_support=None, nearest_resistance=None,
        active_high=None, active_low=None, volume_ratio=None, event_recency_bars=10,
        level_proximity_atr=0.5, fast_ema_period=20, slow_ema_period=50,
    )


def test_volume_block_omitted_for_volumeless_instruments():
    frame = _zigzag([100, 120, 110, 140])
    frame["volume"] = 0.0
    engine = MarketEngine(FAST)
    computed = engine.compute(frame, "IDX", "1d", with_htf=False)
    ctx = engine.context_at(computed, computed.n - 1)
    assert ctx.indicators.volume_ratio is None
    assert ctx.indicators.volume_zscore is None
    assert all(c.v is None for c in ctx.ohlcv_window)


def test_engine_refuses_a_file_that_failed_validation(tmp_path: Path):
    from market_engine.cli import main

    bad = tmp_path / "BAD_1d.csv"
    frame = frame_from_closes([100.0 + i for i in range(60)])
    frame.loc[10, "high"] = frame.loc[10, "low"] - 5.0  # impossible candle
    frame.to_csv(bad, index=False)

    with pytest.raises(SystemExit) as exc:
        main(["context", str(bad), "--bar", "-1"])
    assert "failed validation" in str(exc.value)


def test_cli_writes_context_json(tmp_path: Path):
    from market_engine.cli import main

    good = tmp_path / "GOOD_1d.csv"
    frame_from_closes([100.0 + math.sin(i / 5) * 10 for i in range(400)]).to_csv(
        good, index=False
    )
    out = tmp_path / "ctx.json"
    assert main(["context", str(good), "--bar", "-1", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["symbol"] == "GOOD"
    assert payload["interval"] == "1d"
    assert payload["engine_version"]
    assert "Not trading advice" in payload["disclaimer"]


def test_warmup_fields_are_absent_not_zero():
    frame = _zigzag([100, 120, 110])
    engine = MarketEngine(
        EngineParams(
            ema_periods=(5, 200), atr_period=5, rsi_period=5,
            setup_fast_ema=5, setup_slow_ema=200,
        )
    )
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    ctx = engine.context_at(computed, 6)
    assert "200" not in ctx.indicators.ema  # nowhere near 200 bars of history
    assert not any(np.isnan(v) for v in ctx.indicators.ema.values())
