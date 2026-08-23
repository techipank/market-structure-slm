"""Verify that every number the teacher stated came from the context.

Two checks, because the analysis has two kinds of numeric surface.

**Structured claims** (`Evidence.context_field` / `value`, `LevelClaim.price`,
scenario trigger and invalidation prices) are checked exactly: resolve the
dotted path in the context, compare the value. This is possible only because
the output schema was designed to make it possible - see `schema.py`.

**Prose fields** must contain no numbers at all. That is a much stronger and
much cheaper rule than "prose numbers must be grounded". Deciding whether
"452.10" in a sentence is a level, a target, or an invention requires knowing
the model's intent; deciding whether a sentence contains a digit does not.
Numbers were given a place to live, so their appearance anywhere else is a
protocol violation regardless of whether the value happens to exist somewhere
in the context.

What this is not
----------------
This is a *grounding* check, not a correctness check. It proves the teacher
copied a number that exists in the context; it says nothing about whether
citing that number supports the claim being made. Semantic agreement is a
separate problem, handled later by the dataset quality pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from teacher.schema import TeacherAnalysis

#: Numbers in prose. Handles 1,234.56, .5, 1e-3, and leading signs. Ordinals
#: and digits embedded in words (RSI14) are matched too and that is intended:
#: the rule is that prose carries no digits.
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: Digits that are part of an accepted term rather than a numeric claim.
#: Kept deliberately short: every entry is a hole in the rule.
_PROSE_ALLOWANCES = re.compile(
    r"\b(?:20|50|100|200|5)[ -](?:EMA|period|bar)\b|\bEMA[ -](?:5|20|50|100|200)\b",
    re.IGNORECASE,
)

#: Relative tolerance when comparing a claimed price to a context value. The
#: context is emitted rounded to 4 decimals, so an exact match is expected;
#: this only absorbs float representation noise.
_REL_TOL = 1e-6
_ABS_TOL = 1e-4


class Finding(StrEnum):
    UNRESOLVABLE_FIELD = "UNRESOLVABLE_FIELD"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    UNGROUNDED_PRICE = "UNGROUNDED_PRICE"
    NUMBER_IN_PROSE = "NUMBER_IN_PROSE"


@dataclass(frozen=True)
class GroundingIssue:
    finding: Finding
    location: str
    detail: str


@dataclass
class GroundingReport:
    issues: list[GroundingIssue] = field(default_factory=list)
    #: Numeric claims that resolved to a real context value.
    grounded_claims: int = 0
    #: Numeric claims checked in total.
    total_claims: int = 0

    @property
    def is_grounded(self) -> bool:
        return not self.issues

    @property
    def grounded_fraction(self) -> float:
        return self.grounded_claims / self.total_claims if self.total_claims else 1.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            out[issue.finding.value] = out.get(issue.finding.value, 0) + 1
        return out


# ------------------------------------------------------------ path lookup


def resolve_path(context: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Resolve a dotted path such as ``levels[2].price`` against the context.

    Returns ``(found, value)`` rather than raising or returning a sentinel,
    because ``None`` is a legitimate value in this data and must not be
    confused with "absent".
    """
    node: Any = context
    for raw in path.replace("]", "").replace("[", ".").split("."):
        token = raw.strip()
        if not token:
            continue
        if isinstance(node, dict):
            if token not in node:
                return False, None
            node = node[token]
        elif isinstance(node, list):
            if not token.lstrip("-").isdigit():
                return False, None
            index = int(token)
            if not -len(node) <= index < len(node):
                return False, None
            node = node[index]
        else:
            return False, None
    return True, node


def collect_numeric_values(node: Any, out: set[float] | None = None) -> set[float]:
    """Every number appearing anywhere in the context.

    Used as the fallback check for a price whose stated path does not resolve:
    the model may have cited the wrong path for a real number, which is a
    different (and milder) error than inventing the number outright.
    """
    values = out if out is not None else set()
    if isinstance(node, bool):
        return values
    if isinstance(node, (int, float)) and math.isfinite(node):
        values.add(round(float(node), 6))
    elif isinstance(node, dict):
        for value in node.values():
            collect_numeric_values(value, values)
    elif isinstance(node, list):
        for value in node:
            collect_numeric_values(value, values)
    return values


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _matches_any(value: float, allowed: set[float]) -> bool:
    return any(_close(value, candidate) for candidate in allowed)


# ------------------------------------------------------------- the checks


def verify(analysis: TeacherAnalysis, context: dict[str, Any]) -> GroundingReport:
    report = GroundingReport()
    allowed = collect_numeric_values(context)

    for index, evidence in enumerate(analysis.supporting_evidence):
        _check_citation(report, context, allowed, f"supporting_evidence[{index}]", evidence)
    for index, evidence in enumerate(analysis.conflicting_evidence):
        _check_citation(report, context, allowed, f"conflicting_evidence[{index}]", evidence)

    for index, level in enumerate(analysis.important_levels):
        location = f"important_levels[{index}]"
        _check_price(report, context, allowed, location, level.price, level.context_field)

    for index, scenario in enumerate(analysis.scenarios):
        for name in ("trigger_price", "invalidation_price"):
            price = getattr(scenario, name)
            if price is not None:
                _check_price(
                    report, context, allowed, f"scenarios[{index}].{name}", price, None
                )

    _check_prose(report, analysis)
    return report


def _check_citation(
    report: GroundingReport,
    context: dict[str, Any],
    allowed: set[float],
    location: str,
    evidence: Any,
) -> None:
    report.total_claims += 1
    found, actual = resolve_path(context, evidence.context_field)
    if not found:
        report.issues.append(
            GroundingIssue(
                Finding.UNRESOLVABLE_FIELD,
                location,
                f"context_field {evidence.context_field!r} does not exist in the context",
            )
        )
        return

    if _value_agrees(evidence.value, actual):
        report.grounded_claims += 1
        return

    report.issues.append(
        GroundingIssue(
            Finding.VALUE_MISMATCH,
            location,
            f"cited {evidence.context_field!r} as {evidence.value!r} "
            f"but the context holds {actual!r}",
        )
    )


def _value_agrees(claimed: str, actual: Any) -> bool:
    """Compare a string-typed citation against a context value of any type.

    Numeric comparison when both sides parse as numbers, so "54.23" matches
    54.23; otherwise a normalised string comparison, so "BOS_BULLISH" matches
    regardless of case or surrounding whitespace.
    """
    text = str(claimed).strip()
    try:
        claimed_number = float(text.replace(",", ""))
    except ValueError:
        claimed_number = None

    if claimed_number is not None and isinstance(actual, (int, float)):
        return _close(claimed_number, float(actual))
    if isinstance(actual, (dict, list)):
        # Citing a whole object is allowed but cannot be compared exactly;
        # treat a non-empty container as agreeing so this is not a false alarm.
        return bool(actual) or text.lower() in {"[]", "{}", "none", "null", "empty"}
    return text.strip().lower() == str(actual).strip().lower()


def _check_price(
    report: GroundingReport,
    context: dict[str, Any],
    allowed: set[float],
    location: str,
    price: float,
    path: str | None,
) -> None:
    report.total_claims += 1

    if path:
        found, actual = resolve_path(context, path)
        if found and isinstance(actual, (int, float)) and _close(price, float(actual)):
            report.grounded_claims += 1
            return

    # The path was wrong or absent. The number may still be real, which is a
    # milder failure than fabrication, so it is distinguished here.
    if _matches_any(price, allowed):
        report.grounded_claims += 1
        if path:
            report.issues.append(
                GroundingIssue(
                    Finding.UNRESOLVABLE_FIELD,
                    location,
                    f"price {price} exists in the context but not at the cited path {path!r}",
                )
            )
        return

    report.issues.append(
        GroundingIssue(
            Finding.UNGROUNDED_PRICE,
            location,
            f"price {price} does not appear anywhere in the context",
        )
    )


#: Fields required to be free of digits, and why: each is a place the model
#: might otherwise smuggle an unverifiable number into fluent prose.
PROSE_FIELDS: tuple[str, ...] = (
    "higher_timeframe_rationale",
    "market_structure_explanation",
    "reasoning_summary",
)


def _check_prose(report: GroundingReport, analysis: TeacherAnalysis) -> None:
    def scan(location: str, text: str) -> None:
        cleaned = _PROSE_ALLOWANCES.sub(" ", text or "")
        for match in _NUMBER.finditer(cleaned):
            report.issues.append(
                GroundingIssue(
                    Finding.NUMBER_IN_PROSE,
                    location,
                    f"prose contains the number {match.group()!r}; numeric claims "
                    "belong in a structured field where they can be checked",
                )
            )

    for name in PROSE_FIELDS:
        scan(name, getattr(analysis, name, "") or "")

    for index, evidence in enumerate(analysis.supporting_evidence):
        scan(f"supporting_evidence[{index}].statement", evidence.statement)
    for index, evidence in enumerate(analysis.conflicting_evidence):
        scan(f"conflicting_evidence[{index}].statement", evidence.statement)
    for index, level in enumerate(analysis.important_levels):
        scan(f"important_levels[{index}].note", level.note)
    for index, scenario in enumerate(analysis.scenarios):
        scan(f"scenarios[{index}].condition", scenario.condition)
        scan(f"scenarios[{index}].expectation", scenario.expectation)
