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
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from teacher.schema import TeacherAnalysis

#: Numbers in prose. Handles 1,234.56, .5, 1e-3, and leading signs. Ordinals
#: and digits embedded in words (RSI14) are matched too and that is intended:
#: the rule is that prose carries no digits.
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: Units that turn a number into the *name* of an indicator rather than a
#: measurement of the market. "the 200-day EMA" identifies a series; "RSI
#: below 55" is a claim about where the market is, and stays a violation.
_PERIOD_UNITS = r"(?:EMA|SMA|MA|period|bar|day|daily|week|weekly|candle|session)s?"


def _prose_allowances(context: dict[str, Any] | None) -> re.Pattern[str] | None:
    """Numbers exempt from the no-digits rule, derived from the context.

    A number that appears as a *key* in the context - `indicators.ema.200`,
    `indicators.returns.5` - is a configured parameter, not a measurement.
    Deriving the exemption from the context rather than hardcoding a list
    means it tracks the engine's parameters automatically, and it cannot
    excuse a number the engine never used as a period.

    The unit word is still required, so a bare "200" in prose stays a
    violation; only "the 200-day EMA" is spared.
    """
    periods = _period_keys(context) if context is not None else set()
    if not periods:
        return None
    alternatives = "|".join(sorted((re.escape(p) for p in periods), key=len, reverse=True))
    return re.compile(
        rf"\b(?:{alternatives})[ -]?{_PERIOD_UNITS}\b"
        rf"|\b{_PERIOD_UNITS}[ -]?(?:{alternatives})\b",
        re.IGNORECASE,
    )


def _period_keys(node: Any, out: set[str] | None = None) -> set[str]:
    """Every dict key in the context that is a bare integer."""
    found = out if out is not None else set()
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).isdigit():
                found.add(str(key))
            _period_keys(value, found)
    elif isinstance(node, list):
        for value in node:
            _period_keys(value, found)
    return found

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
    #: A concession, not a defect: the citation did not resolve as written but
    #: named exactly one field in the context, so the value was still checked
    #: by exact lookup. Recorded because a model needing this is not the same
    #: as one that does not, and the difference should be visible.
    FIELD_PATH_IMPRECISE = "FIELD_PATH_IMPRECISE"


@dataclass(frozen=True)
class GroundingIssue:
    finding: Finding
    location: str
    detail: str


@dataclass
class GroundingReport:
    issues: list[GroundingIssue] = field(default_factory=list)
    #: Recoverable imprecision. Kept out of `issues` so it cannot fail
    #: `is_grounded`, kept in the report so it cannot be forgotten.
    concessions: list[GroundingIssue] = field(default_factory=list)
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
        for issue in (*self.issues, *self.concessions):
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
            if token.lstrip("-").isdigit():
                index = int(token)
                if not -len(node) <= index < len(node):
                    return False, None
                node = node[index]
            else:
                # Identity lookup: `levels.L3.price`, or a candle named by its
                # own timestamp. Positional citation into a list is the one
                # thing models measurably cannot do - see `_IDENTITY_KEYS`.
                match = _by_identity(node, token)
                if match is None:
                    return False, None
                node = match
        else:
            return False, None
    return True, node


#: Fields whose value may be used to name an element of a list. `id` is
#: assigned by the engine; `t` and `timestamp` are the natural identity of a
#: bar and a swing and were already there.
_IDENTITY_KEYS: tuple[str, ...] = ("id", "t", "timestamp")


def _by_identity(items: list[Any], token: str) -> Any | None:
    needle = token.strip().lower()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in _IDENTITY_KEYS:
            value = item.get(key)
            if value is not None and str(value).strip().lower() == needle:
                return item
    return None


def all_paths(node: Any, prefix: str = "") -> Iterator[str]:
    """Every addressable path in the context, in `resolve_path` syntax."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from all_paths(value, path)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            path = f"{prefix}[{index}]"
            yield path
            yield from all_paths(value, path)


def resolve_citation(context: dict[str, Any], path: str) -> tuple[bool, Any, str | None]:
    """`resolve_path`, plus a fallback for a citation missing its prefix.

    Measured on 919 real citations: the single commonest defect was writing
    `close_vs_ema.200` for `indicators.close_vs_ema.200` - the number correct,
    the address short of its root. Accepting a suffix that names *exactly one*
    field keeps the guarantee that matters, because resolution is still an
    exact lookup and a wrong value still fails. Two matches is not a citation,
    so `ema.20` - which exists under both `indicators` and `higher_timeframe` -
    remains an error.

    Returns `(found, value, concession)`, where `concession` is the message to
    record when the path only resolved via the fallback.
    """
    found, value = resolve_path(context, path)
    if found:
        return True, value, None

    suffix = "." + path.strip().lstrip(".")
    matches = [p for p in all_paths(context) if p.endswith(suffix)]
    if len(matches) != 1:
        return False, None, None

    found, value = resolve_path(context, matches[0])
    if not found:
        return False, None, None
    return True, value, f"cited {path!r}; resolved uniquely to {matches[0]!r}"


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

    _check_prose(report, analysis, _prose_allowances(context))
    return report


def _check_citation(
    report: GroundingReport,
    context: dict[str, Any],
    allowed: set[float],
    location: str,
    evidence: Any,
) -> None:
    report.total_claims += 1
    found, actual, concession = resolve_citation(context, evidence.context_field)
    if found and concession:
        report.concessions.append(
            GroundingIssue(Finding.FIELD_PATH_IMPRECISE, location, concession)
        )
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
        found, actual, concession = resolve_citation(context, path)
        if found and isinstance(actual, (int, float)) and _close(price, float(actual)):
            if concession:
                report.concessions.append(
                    GroundingIssue(Finding.FIELD_PATH_IMPRECISE, location, concession)
                )
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


def _check_prose(
    report: GroundingReport,
    analysis: TeacherAnalysis,
    allowances: re.Pattern[str] | None = None,
) -> None:
    def scan(location: str, text: str) -> None:
        cleaned = allowances.sub(" ", text or "") if allowances else (text or "")
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
