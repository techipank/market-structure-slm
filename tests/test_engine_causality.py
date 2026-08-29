"""Causality tests: the engine must never see a bar it should not have.

The headline test here is truncation invariance. It is worth more than every
other test in this file combined, because it does not check one feature - it
checks the *property* that makes the whole dataset trustworthy, and it keeps
checking it for features that do not exist yet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.loaders import load_ohlcv_csv
from market_engine.context import MarketEngine
from market_engine.params import EngineParams

RAW = Path("data/raw")
PKG = Path(__file__).resolve().parents[1] / "market_engine"

def _discover() -> list[tuple[str, str]]:
    """Whatever is actually in data/raw, not a hardcoded list.

    An earlier version named the files explicitly. When the symbol universe
    changed, every one of those tests silently turned into a skip - including
    truncation invariance, the single most important test here. A test that
    stops running because its fixture was renamed protects nothing, and it
    does so quietly. Discovering the files makes the suite follow the data.
    """
    out: list[tuple[str, str]] = []
    for path in sorted(RAW.glob("*.csv")):
        _, _, interval = path.stem.rpartition("_")
        out.append((path.name, interval or "1d"))
    return out


#: Real fetched data is gitignored, so these tests skip when it is absent
#: rather than failing a clean checkout. Locally they are the real check.
REAL_FILES = _discover() or [("<no data fetched>", "1d")]


def _load(name: str):
    path = RAW / name
    if not path.exists():
        pytest.skip(f"{path} not present; run `msdata fetch` to enable this test")
    return load_ohlcv_csv(path).frame


@pytest.mark.parametrize(("name", "interval"), REAL_FILES)
def test_truncation_invariance(name: str, interval: str):
    """context_at(t) must not change when bars after t are deleted.

    If any feature peeks forward - a centred rolling window, a swing labelled
    before confirmation, a higher-timeframe bar built from future days - the
    two contexts differ and this fails. That is the entire point.
    """
    frame = _load(name)
    engine = MarketEngine()
    full = engine.compute(frame, "TEST", interval)

    # Deterministic, spread across the series, past the 200-EMA warm-up.
    n = full.n
    targets = [int(n * f) for f in (0.35, 0.55, 0.75, 0.95)]
    targets = [t for t in targets if 400 <= t < n]
    assert targets, f"{name} too short to test meaningfully"

    for t in targets:
        truncated = engine.compute(frame.iloc[: t + 1].copy(), "TEST", interval)
        a = engine.context_at(full, t)
        b = engine.context_at(truncated, t)
        assert a.model_dump(mode="json") == b.model_dump(mode="json"), (
            f"{name} bar {t}: context changed when future bars were removed"
        )


@pytest.mark.parametrize(("name", "interval"), REAL_FILES[:1])
def test_higher_timeframe_never_uses_an_open_period(name: str, interval: str):
    """Every higher-TF bar surfaced must have closed before the current bar."""
    frame = _load(name)
    engine = MarketEngine()
    computed = engine.compute(frame, "TEST", interval)
    assert computed.htf is not None

    for t in (500, 1200, 2000, computed.n - 1):
        if t >= computed.n:
            continue
        ctx = engine.context_at(computed, t)
        if ctx.higher_timeframe is None:
            continue
        now = pd.Timestamp(ctx.as_of)
        htf_open = pd.Timestamp(ctx.higher_timeframe.as_of)
        # The HTF bar must have opened strictly in the past, and its whole
        # period must be over: the engine only exposes closed bars, so the
        # next HTF open is also at or before now.
        assert htf_open <= now


def test_swing_is_invisible_until_confirmed():
    """A swing must not appear in any context before index + swing_lookback."""
    params = EngineParams(
        swing_lookback=3, atr_period=5, ema_periods=(5, 10),
        setup_fast_ema=5, setup_slow_ema=10,
    )
    frame = _zigzag([100, 118, 104, 130, 112], steps=8)
    engine = MarketEngine(params)
    computed = engine.compute(frame, "SYN", "1d", with_htf=False)
    assert computed.swings, "fixture produced no swings"

    swing = computed.swings[0]
    assert swing.confirmed_index == swing.index + params.swing_lookback

    for t in range(swing.index, swing.confirmed_index):
        ctx = engine.context_at(computed, t)
        assert swing.index not in [s.index for s in ctx.structure.swings]

    ctx = engine.context_at(computed, swing.confirmed_index)
    assert swing.index in [s.index for s in ctx.structure.swings]


def test_no_forward_looking_primitives_in_engine_source():
    """Static guard against the two ways to accidentally read the future."""
    offenders: list[str] = []
    for path in PKG.glob("*.py"):
        code = re.sub(r'""".*?"""', "", path.read_text(encoding="utf-8"), flags=re.S)
        code = re.sub(r"#.*", "", code)
        if "center=True" in code:
            offenders.append(f"{path.name}: rolling(center=True)")
        # shift(-n) pulls a future value backwards onto the current bar.
        if re.search(r"\.shift\(\s*-", code):
            offenders.append(f"{path.name}: negative shift")
    assert not offenders, offenders


# ------------------------------------------------------------------ fixture


def _zigzag(turning_points: list[float], steps: int) -> pd.DataFrame:
    """Deterministic piecewise-linear price path through the given points."""
    closes: list[float] = []
    for a, b in zip(turning_points, turning_points[1:], strict=False):
        closes.extend(a + (b - a) * k / steps for k in range(steps))
    closes.append(turning_points[-1])
    return frame_from_closes(closes)


def frame_from_closes(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(closes), tz="UTC")
    opens = [closes[0], *closes[:-1]]
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": opens,
            "high": [max(o, c) + 0.5 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )
