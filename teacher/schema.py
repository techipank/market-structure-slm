"""The teacher's output contract.

The central design decision in this module: **every numeric claim is forced
into a structured field that names the context field it came from.**

The lazy alternative is to let the model write prose full of prices and then
hunt for hallucinations with a regex. That is guesswork - "452.10" in a
sentence might be a level, a target, a typo, or invented, and no amount of
pattern matching recovers the model's intent. By requiring

    {"price": 452.10, "context_field": "levels[2].price", "role": "SUPPORT"}

verification becomes an exact lookup: does `levels[2].price` exist, and is it
452.10? Designing the output shape around how it will be checked is worth more
than any amount of downstream cleverness.

Two enums (`StructureState`, `SetupType`) deliberately mirror the market
engine's vocabulary. The teacher is restating something the engine already
computed deterministically, which makes *agreement* a measurable quantity
rather than a matter of opinion.

`conflicting_evidence` is mandatory. Models are agreeable; asked for an
analysis they produce a one-sided case. Making the counter-argument a required
field is a structural fix for that, and structure is far more reliable than
politely asking for balance in the prompt.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from teacher import TEACHER_SCHEMA_VERSION


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class HigherTimeframeBias(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    #: The context carried no closed higher-timeframe bar. Saying so is
    #: correct behaviour; inventing a bias is not.
    UNAVAILABLE = "UNAVAILABLE"


class StructureState(StrEnum):
    """Same vocabulary as the engine's TrendState, so agreement is measurable."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    EXPANSION = "EXPANSION"
    CONTRACTION = "CONTRACTION"
    RANGE = "RANGE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class MarketState(StrEnum):
    """Coarser than structure: what the market is *doing* right now.

    These values are the evaluation categories the final benchmark reports
    per-class accuracy over, so they are fixed here rather than left to prose.
    """

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    REVERSAL = "REVERSAL"
    CHOPPY = "CHOPPY"


class SetupType(StrEnum):
    """Engine setup vocabulary plus NONE.

    Constrained to the engine's names on purpose: a teacher free to invent
    setup names produces a label space that cannot be scored, and a student
    model cannot be evaluated against a moving target.
    """

    TREND_CONTINUATION_LONG = "TREND_CONTINUATION_LONG"
    TREND_CONTINUATION_SHORT = "TREND_CONTINUATION_SHORT"
    RANGE_FADE_LONG = "RANGE_FADE_LONG"
    RANGE_FADE_SHORT = "RANGE_FADE_SHORT"
    VOLATILITY_BREAKOUT_LONG = "VOLATILITY_BREAKOUT_LONG"
    REVERSAL_LONG = "REVERSAL_LONG"
    REVERSAL_SHORT = "REVERSAL_SHORT"
    NONE = "NONE"


class LevelRole(StrEnum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    TRIGGER = "TRIGGER"
    INVALIDATION = "INVALIDATION"


class Evidence(BaseModel):
    statement: str = Field(
        description=(
            "One qualitative sentence. Must contain no numbers: the number "
            "belongs in `value`, so it can be checked."
        )
    )
    context_field: str = Field(
        description=(
            "Dotted path to the field in the supplied context that this claim "
            "rests on, for example 'indicators.rsi' or 'structure.last_event.type'."
        )
    )
    value: str = Field(
        description=(
            "The value found at that path, copied verbatim as a string. "
            "Not reformatted, not rounded, not recalculated."
        )
    )


class LevelClaim(BaseModel):
    price: float = Field(
        description="A price copied from the context. Never computed, never rounded."
    )
    role: LevelRole = Field(description="What this price is, relative to the current close.")
    context_field: str = Field(
        description="Dotted path in the context where this exact price appears."
    )
    note: str = Field(description="Why this level matters. No numbers.")


class Scenario(BaseModel):
    name: str = Field(description="Short label, for example 'Continuation higher'.")
    direction: Direction = Field(description="Direction this scenario describes.")
    condition: str = Field(
        description="What would have to happen for this path to unfold. No numbers."
    )
    trigger_price: float | None = Field(
        default=None,
        description="Price that would confirm it, copied from the context, or null.",
    )
    invalidation_price: float | None = Field(
        default=None,
        description="Price that would rule it out, copied from the context, or null.",
    )
    expectation: str = Field(
        description="What the structure would then imply. Descriptive, not predictive. No numbers."
    )


class TeacherAnalysis(BaseModel):
    """Structured interpretation of one MarketContext."""

    higher_timeframe_bias: HigherTimeframeBias = Field(
        description=(
            "Directional bias of the higher timeframe. UNAVAILABLE when the context "
            "carries no closed higher-timeframe bar - do not infer one from the base "
            "timeframe."
        )
    )
    higher_timeframe_rationale: str = Field(
        description="Why that bias, referring to higher-timeframe fields. No numbers."
    )

    market_structure: StructureState = Field(
        description=(
            "Swing structure, using the same vocabulary as the context's "
            "structure.trend field."
        )
    )
    market_structure_explanation: str = Field(
        description="Which swing labels and structure breaks lead to that reading. No numbers."
    )

    market_state: MarketState = Field(description="What the market is doing at this bar.")
    setup_type: SetupType = Field(
        description=(
            "Best-matching configuration from the fixed vocabulary, or NONE. "
            "Choosing NONE is a valid and often correct answer."
        )
    )

    supporting_evidence: list[Evidence] = Field(
        min_length=2,
        description="Context facts that support the reading above.",
    )
    conflicting_evidence: list[Evidence] = Field(
        min_length=1,
        description=(
            "Context facts that argue against it. Mandatory. If the picture looks "
            "one-sided, state the strongest risk to the reading rather than omitting it."
        ),
    )

    important_levels: list[LevelClaim] = Field(
        min_length=1, description="Prices that matter, each copied from the context."
    )
    scenarios: list[Scenario] = Field(
        min_length=2,
        description=(
            "At least two paths the market could take, covering more than one "
            "direction. Descriptions of conditional structure, not forecasts."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the context supports this reading, 0 to 1. Low confidence is "
            "the correct answer for ambiguous or data-poor bars, and is not a failure."
        ),
    )
    reasoning_summary: str = Field(
        description=(
            "Two to four sentences tying the analysis together. Must contain no "
            "numbers: every numeric claim belongs in a structured field above."
        )
    )

    schema_version: str = Field(
        default=TEACHER_SCHEMA_VERSION, description="Version of this analysis schema."
    )
