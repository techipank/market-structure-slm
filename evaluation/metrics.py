"""Scoring one analysis against the context it was produced from.

Three independent families of check, which fail in different ways and must not
be averaged into a single number:

**Schema compliance** - did it produce the right shape, and how many attempts
did that take? Handled upstream by pydantic; recorded here.

**Agreement** - does the analysis match what the engine computed
deterministically? Only meaningful where the engine is genuinely authoritative.
`structure.trend` is a definition, so disagreement is error. `setup_type` is a
judgement, so disagreement is just disagreement, and calling it "accuracy"
would be dishonest.

**Internal consistency** - is the analysis self-contradictory? This needs no
ground truth at all. A model can be schema-valid and perfectly grounded while
claiming an uptrend in a downtrend structure, or placing support above price.
These are the failures that survive every other check.

Everything is a pure function of `(analysis, context)`, so the same code scores
the teacher during model selection, the dataset during curation, and the
student during final evaluation. Scoring the three differently would make
their numbers incomparable, which is the only reason to compute them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from teacher.schema import (
    Direction,
    HigherTimeframeBias,
    MarketState,
    SetupType,
    StructureState,
    TeacherAnalysis,
)

# --------------------------------------------------------------- agreement

#: Market states that are only coherent alongside a bullish-leaning structure,
#: and vice versa. Used for contradiction detection, not for grading: a market
#: can be RANGING inside any structure, so those pairs are simply absent.
_BULLISH_STATES = {MarketState.TRENDING_UP, MarketState.BREAKOUT}
_BEARISH_STATES = {MarketState.TRENDING_DOWN, MarketState.BREAKDOWN}

_LONG_SETUPS = {
    SetupType.TREND_CONTINUATION_LONG,
    SetupType.RANGE_FADE_LONG,
    SetupType.VOLATILITY_BREAKOUT_LONG,
    SetupType.REVERSAL_LONG,
}
_SHORT_SETUPS = {
    SetupType.TREND_CONTINUATION_SHORT,
    SetupType.RANGE_FADE_SHORT,
    SetupType.REVERSAL_SHORT,
}


class Contradiction(StrEnum):
    STATE_VS_STRUCTURE = "STATE_VS_STRUCTURE"
    SETUP_VS_STATE = "SETUP_VS_STATE"
    SETUP_NOT_OFFERED = "SETUP_NOT_OFFERED"
    LEVEL_SIDE_WRONG = "LEVEL_SIDE_WRONG"
    SCENARIOS_ONE_SIDED = "SCENARIOS_ONE_SIDED"
    SCENARIO_LEVELS_INVERTED = "SCENARIO_LEVELS_INVERTED"
    EVIDENCE_DOUBLE_COUNTED = "EVIDENCE_DOUBLE_COUNTED"
    CONFIDENCE_VS_EVIDENCE = "CONFIDENCE_VS_EVIDENCE"
    HTF_CLAIMED_WHEN_ABSENT = "HTF_CLAIMED_WHEN_ABSENT"


@dataclass
class AnalysisScore:
    """Per-example scoring. Every field is countable across a benchmark run."""

    # --- agreement with the deterministic engine ------------------------
    #: None when the engine had no authoritative value to compare against.
    structure_agrees: bool | None = None
    htf_bias_agrees: bool | None = None
    setup_agrees: bool | None = None

    # --- grounding (rolled up from the teacher's own verifier) ----------
    grounded_claims: int = 0
    total_claims: int = 0
    ungrounded_prices: int = 0
    value_mismatches: int = 0
    unresolvable_fields: int = 0
    numbers_in_prose: int = 0

    # --- internal consistency -------------------------------------------
    contradictions: list[Contradiction] = field(default_factory=list)

    @property
    def hallucinated_numbers(self) -> int:
        """Numeric claims that cannot be traced to the context.

        A misquoted value and an invented price are both hallucinated numbers.
        A correct number cited at the wrong path is not - that is a citation
        error, counted separately, because conflating the two would overstate
        how often the model fabricates.
        """
        return self.ungrounded_prices + self.value_mismatches

    @property
    def is_grounded(self) -> bool:
        return (
            self.hallucinated_numbers == 0
            and self.numbers_in_prose == 0
            and self.unresolvable_fields == 0
        )

    @property
    def contradiction_count(self) -> int:
        return len(self.contradictions)

    @property
    def agreement_points(self) -> tuple[int, int]:
        """(agreed, comparable) across the three agreement checks."""
        checks = [self.structure_agrees, self.htf_bias_agrees, self.setup_agrees]
        comparable = [c for c in checks if c is not None]
        return sum(comparable), len(comparable)


def score_analysis(
    analysis: TeacherAnalysis,
    context: dict[str, Any],
    grounding_counts: dict[str, int] | None = None,
    grounded_claims: int = 0,
    total_claims: int = 0,
) -> AnalysisScore:
    counts = grounding_counts or {}
    score = AnalysisScore(
        grounded_claims=grounded_claims,
        total_claims=total_claims,
        ungrounded_prices=counts.get("UNGROUNDED_PRICE", 0),
        value_mismatches=counts.get("VALUE_MISMATCH", 0),
        unresolvable_fields=counts.get("UNRESOLVABLE_FIELD", 0),
        numbers_in_prose=counts.get("NUMBER_IN_PROSE", 0),
    )
    _score_agreement(score, analysis, context)
    score.contradictions = find_contradictions(analysis, context)
    return score


def _score_agreement(
    score: AnalysisScore, analysis: TeacherAnalysis, context: dict[str, Any]
) -> None:
    structure = context.get("structure") or {}
    engine_trend = structure.get("trend")
    if engine_trend and engine_trend != StructureState.INSUFFICIENT_DATA.value:
        # `structure.trend` is a definition applied to confirmed swings, not an
        # opinion. Disagreeing with it is being wrong.
        score.structure_agrees = analysis.market_structure.value == engine_trend

    htf = context.get("higher_timeframe")
    if htf:
        score.htf_bias_agrees = analysis.higher_timeframe_bias.value == htf.get("bias")
    else:
        # No closed higher-timeframe bar: the only correct answer is
        # UNAVAILABLE. Inventing a bias here is the failure being measured.
        score.htf_bias_agrees = (
            analysis.higher_timeframe_bias is HigherTimeframeBias.UNAVAILABLE
        )

    offered = {s.get("name") for s in (context.get("setups") or [])}
    if offered:
        score.setup_agrees = analysis.setup_type.value in offered
    else:
        score.setup_agrees = analysis.setup_type is SetupType.NONE


# ---------------------------------------------------------- contradictions


def find_contradictions(
    analysis: TeacherAnalysis, context: dict[str, Any]
) -> list[Contradiction]:
    """Self-consistency checks that need no ground truth.

    Each one encodes a statement that cannot be true of the same analysis at
    the same time. Nothing here is a matter of taste - a support level above
    the current price is not a debatable reading, it is a contradiction in
    terms.
    """
    found: list[Contradiction] = []

    # A directional market state inside the opposite swing structure.
    if (
        analysis.market_state in _BULLISH_STATES
        and analysis.market_structure is StructureState.DOWNTREND
    ) or (
        analysis.market_state in _BEARISH_STATES
        and analysis.market_structure is StructureState.UPTREND
    ):
        found.append(Contradiction.STATE_VS_STRUCTURE)

    # A long setup in a market the analysis itself called falling.
    if (analysis.setup_type in _LONG_SETUPS and analysis.market_state in _BEARISH_STATES) or (
        analysis.setup_type in _SHORT_SETUPS and analysis.market_state in _BULLISH_STATES
    ):
        found.append(Contradiction.SETUP_VS_STATE)

    # Naming a setup the engine did not detect. Not automatically wrong - the
    # engine's rules are narrow - but it means the analysis asserted a pattern
    # whose conditions were not met, which is worth counting.
    offered = {s.get("name") for s in (context.get("setups") or [])}
    if analysis.setup_type is not SetupType.NONE and analysis.setup_type.value not in offered:
        found.append(Contradiction.SETUP_NOT_OFFERED)

    close = _current_close(context)
    if close is not None:
        for level in analysis.important_levels:
            if level.role.value == "SUPPORT" and level.price > close:
                found.append(Contradiction.LEVEL_SIDE_WRONG)
                break
            if level.role.value == "RESISTANCE" and level.price < close:
                found.append(Contradiction.LEVEL_SIDE_WRONG)
                break

    # The schema can require two scenarios but not two *different* ones.
    directions = {s.direction for s in analysis.scenarios}
    if len(directions) < 2 and Direction.NEUTRAL not in directions:
        found.append(Contradiction.SCENARIOS_ONE_SIDED)

    for scenario in analysis.scenarios:
        trigger, invalidation = scenario.trigger_price, scenario.invalidation_price
        if trigger is None or invalidation is None:
            continue
        # A bullish path confirmed below the price that would kill it, or the
        # bearish mirror, is geometrically impossible.
        if scenario.direction is Direction.BULLISH and invalidation > trigger:
            found.append(Contradiction.SCENARIO_LEVELS_INVERTED)
            break
        if scenario.direction is Direction.BEARISH and invalidation < trigger:
            found.append(Contradiction.SCENARIO_LEVELS_INVERTED)
            break

    # The same fact used as evidence both for and against.
    supporting = {(e.context_field, e.value) for e in analysis.supporting_evidence}
    conflicting = {(e.context_field, e.value) for e in analysis.conflicting_evidence}
    if supporting & conflicting:
        found.append(Contradiction.EVIDENCE_DOUBLE_COUNTED)

    # High confidence while the analysis's own counter-case outweighs its case.
    if analysis.confidence >= 0.8 and len(analysis.conflicting_evidence) > len(
        analysis.supporting_evidence
    ):
        found.append(Contradiction.CONFIDENCE_VS_EVIDENCE)

    if not context.get("higher_timeframe") and (
        analysis.higher_timeframe_bias is not HigherTimeframeBias.UNAVAILABLE
    ):
        found.append(Contradiction.HTF_CLAIMED_WHEN_ABSENT)

    return found


def _current_close(context: dict[str, Any]) -> float | None:
    window = context.get("ohlcv_window") or []
    if not window:
        return None
    value = window[-1].get("c")
    return float(value) if isinstance(value, (int, float)) else None


# ------------------------------------------------------------ consistency


#: Categorical fields compared when measuring whether a model answers the same
#: question the same way twice.
CONSISTENCY_FIELDS: tuple[str, ...] = (
    "higher_timeframe_bias",
    "market_structure",
    "market_state",
    "setup_type",
)


def self_consistency(analyses: list[TeacherAnalysis]) -> dict[str, float]:
    """Mean pairwise agreement across repeated runs of one example.

    Why this matters more than it looks: a model that answers differently each
    time cannot be a teacher, however good any individual answer is. Training a
    student on its output distils noise. Sampling temperature is zero here, so
    remaining variance comes from the provider - batching, expert routing -
    and is outside our control, which is exactly why it must be measured
    rather than assumed away.
    """
    if len(analyses) < 2:
        return {name: 1.0 for name in CONSISTENCY_FIELDS} | {"confidence_spread": 0.0}

    out: dict[str, float] = {}
    pairs = [(i, j) for i in range(len(analyses)) for j in range(i + 1, len(analyses))]
    for name in CONSISTENCY_FIELDS:
        agreements = [
            getattr(analyses[i], name) == getattr(analyses[j], name) for i, j in pairs
        ]
        out[name] = sum(agreements) / len(agreements)

    confidences = [a.confidence for a in analyses]
    out["confidence_spread"] = max(confidences) - min(confidences)
    return out
