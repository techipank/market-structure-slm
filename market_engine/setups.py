"""Candidate setup detection.

These are **descriptions of recognisable configurations**, not signals, not
predictions, and not recommendations. A setup firing means "the geometry
currently matches this named pattern", nothing more. No rule here has been
backtested and none should be traded.

Why rules and not a model: the point of this engine is to give the teacher LLM
facts it cannot argue with. A rule-based setup carries its own evidence - the
exact list of conditions that were true - so the dataset quality pipeline can
later check that the teacher's prose matches the conditions rather than
inventing new ones.

Each rule states every condition explicitly. A setup is emitted only when all
of its conditions hold; rules that came close are reported separately as near
misses so the teacher can see what was *almost* true without being told it was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from market_engine.levels import PriceLevel
from market_engine.regime import TrendRegime, VolatilityRegime
from market_engine.structure import Bias, EventType, StructureEvent, TrendState
from market_engine.swings import SwingPoint


@dataclass(frozen=True)
class SetupInput:
    close: float
    atr: float
    rsi: float | None
    emas: dict[int, float]
    trend: TrendState
    bias: Bias
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    atr_percentile: float | None
    last_event: StructureEvent | None
    bars_since_event: int | None
    nearest_support: PriceLevel | None
    nearest_resistance: PriceLevel | None
    active_high: SwingPoint | None
    active_low: SwingPoint | None
    volume_ratio: float | None
    event_recency_bars: int
    level_proximity_atr: float
    fast_ema_period: int
    slow_ema_period: int


@dataclass(frozen=True)
class SetupCandidate:
    name: str
    direction: str  # LONG | SHORT
    description: str
    #: Every condition the rule tested, with the outcome. Present even for
    #: emitted setups so the teacher must cite real evidence.
    conditions: list[tuple[str, bool]] = field(default_factory=list)
    #: Price that would confirm the pattern, and the price that would break it.
    #: Both are derived from structure, never invented.
    trigger_level: float | None = None
    invalidation_level: float | None = None

    @property
    def complete(self) -> bool:
        return all(ok for _, ok in self.conditions)

    @property
    def met(self) -> int:
        return sum(1 for _, ok in self.conditions if ok)


def _near(price: float, level: float | None, atr: float, multiple: float) -> bool:
    if level is None or not np.isfinite(atr) or atr <= 0:
        return False
    return abs(price - level) <= atr * multiple


def _recent(inp: SetupInput, kinds: set[EventType]) -> bool:
    return (
        inp.last_event is not None
        and inp.last_event.type in kinds
        and inp.bars_since_event is not None
        and inp.bars_since_event <= inp.event_recency_bars
    )


def trend_continuation_long(inp: SetupInput) -> SetupCandidate:
    """Uptrend that has pulled back into the moving-average band.

    The RSI ceiling is the interesting condition: it is there to exclude
    buying the vertical part of the move, which is the opposite of a pullback.
    """
    fast = inp.emas.get(inp.fast_ema_period)
    slow = inp.emas.get(inp.slow_ema_period)
    pulled_back = fast is not None and inp.close <= fast + 0.25 * inp.atr
    above_structure = slow is not None and inp.close > slow
    return SetupCandidate(
        name="TREND_CONTINUATION_LONG",
        direction="LONG",
        description=(
            f"Uptrend structure pulled back toward the {inp.fast_ema_period} EMA "
            f"while holding above the {inp.slow_ema_period}."
        ),
        conditions=[
            ("trend is UPTREND", inp.trend is TrendState.UPTREND),
            ("bias is BULLISH", inp.bias is Bias.BULLISH),
            (f"close pulled back to or below the {inp.fast_ema_period} EMA", bool(pulled_back)),
            (f"close still above the {inp.slow_ema_period} EMA", bool(above_structure)),
            ("RSI below 65 (not extended)", inp.rsi is not None and inp.rsi < 65),
        ],
        trigger_level=inp.active_high.price if inp.active_high else None,
        invalidation_level=inp.active_low.price if inp.active_low else None,
    )


def trend_continuation_short(inp: SetupInput) -> SetupCandidate:
    fast = inp.emas.get(inp.fast_ema_period)
    slow = inp.emas.get(inp.slow_ema_period)
    pulled_back = fast is not None and inp.close >= fast - 0.25 * inp.atr
    below_structure = slow is not None and inp.close < slow
    return SetupCandidate(
        name="TREND_CONTINUATION_SHORT",
        direction="SHORT",
        description=(
            f"Downtrend structure rallied back toward the {inp.fast_ema_period} EMA "
            f"while capped by the {inp.slow_ema_period}."
        ),
        conditions=[
            ("trend is DOWNTREND", inp.trend is TrendState.DOWNTREND),
            ("bias is BEARISH", inp.bias is Bias.BEARISH),
            (f"close rallied to or above the {inp.fast_ema_period} EMA", bool(pulled_back)),
            (f"close still below the {inp.slow_ema_period} EMA", bool(below_structure)),
            ("RSI above 35 (not extended)", inp.rsi is not None and inp.rsi > 35),
        ],
        trigger_level=inp.active_low.price if inp.active_low else None,
        invalidation_level=inp.active_high.price if inp.active_high else None,
    )


def range_fade_long(inp: SetupInput) -> SetupCandidate:
    support = inp.nearest_support
    return SetupCandidate(
        name="RANGE_FADE_LONG",
        direction="LONG",
        description="Non-trending market trading down into an established support cluster.",
        conditions=[
            ("regime is RANGING", inp.trend_regime is TrendRegime.RANGING),
            (
                "structure is RANGE or CONTRACTION",
                inp.trend in {TrendState.RANGE, TrendState.CONTRACTION},
            ),
            (
                "price is at a support level",
                _near(inp.close, support.price if support else None, inp.atr,
                      inp.level_proximity_atr),
            ),
            ("support has 2+ touches", support is not None and support.touches >= 2),
            ("RSI below 45", inp.rsi is not None and inp.rsi < 45),
        ],
        trigger_level=support.price if support else None,
        invalidation_level=(support.price - inp.atr) if support else None,
    )


def range_fade_short(inp: SetupInput) -> SetupCandidate:
    resistance = inp.nearest_resistance
    return SetupCandidate(
        name="RANGE_FADE_SHORT",
        direction="SHORT",
        description="Non-trending market trading up into an established resistance cluster.",
        conditions=[
            ("regime is RANGING", inp.trend_regime is TrendRegime.RANGING),
            (
                "structure is RANGE or CONTRACTION",
                inp.trend in {TrendState.RANGE, TrendState.CONTRACTION},
            ),
            (
                "price is at a resistance level",
                _near(inp.close, resistance.price if resistance else None, inp.atr,
                      inp.level_proximity_atr),
            ),
            ("resistance has 2+ touches", resistance is not None and resistance.touches >= 2),
            ("RSI above 55", inp.rsi is not None and inp.rsi > 55),
        ],
        trigger_level=resistance.price if resistance else None,
        invalidation_level=(resistance.price + inp.atr) if resistance else None,
    )


def volatility_breakout_long(inp: SetupInput) -> SetupCandidate:
    """Compression followed by an upside break.

    The volatility condition is the point: breakouts out of an already-violent
    market are far less distinctive than breakouts out of a quiet one, and the
    ATR percentile is what separates the two.
    """
    return SetupCandidate(
        name="VOLATILITY_BREAKOUT_LONG",
        direction="LONG",
        description="Upside structure break emerging from a low-volatility compression.",
        conditions=[
            (
                "volatility was LOW before the break",
                inp.volatility_regime is VolatilityRegime.LOW,
            ),
            ("structure was coiling (CONTRACTION or RANGE)",
             inp.trend in {TrendState.CONTRACTION, TrendState.RANGE}),
            ("a bullish break occurred recently",
             _recent(inp, {EventType.BOS_BULLISH, EventType.CHOCH_BULLISH})),
            ("close is above the broken level",
             inp.last_event is not None and inp.close > inp.last_event.level),
        ],
        trigger_level=inp.last_event.level if inp.last_event else None,
        invalidation_level=inp.active_low.price if inp.active_low else None,
    )


def reversal_long(inp: SetupInput) -> SetupCandidate:
    """A bearish trend that just produced its first bullish CHoCH.

    CHoCH rather than BOS is the whole definition: a break in the direction of
    the existing trend says nothing about reversal.
    """
    return SetupCandidate(
        name="REVERSAL_LONG",
        direction="LONG",
        description="First bullish change of character after a downtrend.",
        conditions=[
            ("a bullish CHoCH occurred recently", _recent(inp, {EventType.CHOCH_BULLISH})),
            ("close holds above the broken level",
             inp.last_event is not None and inp.close > inp.last_event.level),
            ("RSI has recovered above 45", inp.rsi is not None and inp.rsi > 45),
        ],
        trigger_level=inp.last_event.level if inp.last_event else None,
        invalidation_level=inp.active_low.price if inp.active_low else None,
    )


def reversal_short(inp: SetupInput) -> SetupCandidate:
    return SetupCandidate(
        name="REVERSAL_SHORT",
        direction="SHORT",
        description="First bearish change of character after an uptrend.",
        conditions=[
            ("a bearish CHoCH occurred recently", _recent(inp, {EventType.CHOCH_BEARISH})),
            ("close holds below the broken level",
             inp.last_event is not None and inp.close < inp.last_event.level),
            ("RSI has rolled below 55", inp.rsi is not None and inp.rsi < 55),
        ],
        trigger_level=inp.last_event.level if inp.last_event else None,
        invalidation_level=inp.active_high.price if inp.active_high else None,
    )


ALL_RULES = (
    trend_continuation_long,
    trend_continuation_short,
    range_fade_long,
    range_fade_short,
    volatility_breakout_long,
    reversal_long,
    reversal_short,
)

#: A rule with at least this fraction of its conditions satisfied is reported
#: as a near miss. Two thirds keeps the list short while still surfacing the
#: "one condition away" cases that make a description honest.
NEAR_MISS_FRACTION = 0.66


def evaluate_setups(inp: SetupInput) -> tuple[list[SetupCandidate], list[str]]:
    """Return (complete setups, near-miss descriptions)."""
    complete: list[SetupCandidate] = []
    near: list[str] = []
    for rule in ALL_RULES:
        candidate = rule(inp)
        if candidate.complete:
            complete.append(candidate)
            continue
        total = len(candidate.conditions)
        if total and candidate.met / total >= NEAR_MISS_FRACTION:
            unmet = [name for name, ok in candidate.conditions if not ok]
            near.append(
                f"{candidate.name}: {candidate.met}/{total} met, "
                f"missing {'; '.join(unmet)}"
            )
    return complete, near
