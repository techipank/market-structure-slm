"""The gate: which teacher answers are fit to train on.

Every rule here was chosen from a measurement, not from taste. The teacher
benchmark ran 219 analyses over 163 examples and reported exactly where the
model goes wrong, so the gate rejects those failure modes by name rather than
applying a general notion of quality:

| defect                      | measured rate | why it is disqualifying          |
|-----------------------------|---------------|----------------------------------|
| ungrounded / misquoted number | 3 in 219    | teaches the student to invent    |
| unresolvable citation         | 8 in 219    | teaches an address that is a lie |
| digits in prose               | 16 in 219   | teaches numbers to escape checks |
| internal contradiction        | 26 in 163   | teaches incoherent reasoning     |

Why reject rather than repair
-----------------------------
A flawed answer could be sent back to the teacher for correction, and the
runner already knows how to do that for schema failures. It is not done here.
A repaired answer is a different distribution from a first answer - the model
is now imitating its own correction - and mixing the two silently would make
the corpus something other than what it claims to be. Rejection is cheap
(examples are plentiful, calls are not expensive) and it keeps the corpus
honestly described as "answers the teacher got right unaided".

Why the gate is all-or-nothing
------------------------------
A partial-credit gate ("accept if 90% of claims are grounded") sounds more
forgiving and is worse. The student cannot see which 10% was wrong, so the
lesson it takes from a mostly-correct example includes the incorrect part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Reject(StrEnum):
    """Why an analysis was refused. One value per distinct failure mode, so
    the rejection log can be counted rather than read."""

    CALL_FAILED = "CALL_FAILED"
    UNGROUNDED_NUMBER = "UNGROUNDED_NUMBER"
    UNRESOLVABLE_CITATION = "UNRESOLVABLE_CITATION"
    NUMBER_IN_PROSE = "NUMBER_IN_PROSE"
    CONTRADICTION = "CONTRADICTION"
    #: The teacher answered, but only after being told what it got wrong. Off
    #: by default - see `GateSpec.require_first_attempt`.
    NEEDED_REPAIR = "NEEDED_REPAIR"


@dataclass(frozen=True)
class GateSpec:
    """What the gate enforces. Every flag defaults to the strict reading.

    `require_first_attempt` is the one genuinely optional rule. The measured
    first-attempt rate was 100%, so it costs nothing today; it is here so that
    a future teacher with a weaker rate cannot quietly fill the corpus with
    repaired answers.
    """

    require_grounded: bool = True
    forbid_contradictions: bool = True
    require_first_attempt: bool = False


@dataclass
class Verdict:
    accepted: bool
    reasons: list[Reject] = field(default_factory=list)
    #: Human-readable specifics, for the rejection log. The enum says what
    #: kind of failure; this says which field and which number.
    detail: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": [str(r) for r in self.reasons],
            "detail": self.detail,
        }


#: The strict reading, as a singleton. A dataclass call in an argument
#: default is evaluated once at import anyway; naming it says so.
STRICT = GateSpec()


def judge(row: dict[str, Any], spec: GateSpec = STRICT) -> Verdict:
    """Judge one scored result row, as produced by the generation batch.

    Takes the row rather than the analysis because scoring has already
    happened: the grounding verifier and the contradiction finder both ran
    when the response arrived, and re-running them here would risk the two
    stages disagreeing about the same answer.
    """
    reasons: list[Reject] = []
    detail: list[str] = []

    if not row.get("ok") or row.get("analysis") is None:
        return Verdict(False, [Reject.CALL_FAILED], [str(row.get("error") or "no analysis")])

    if spec.require_grounded:
        hallucinated = int(row.get("hallucinated_numbers") or 0)
        if hallucinated:
            reasons.append(Reject.UNGROUNDED_NUMBER)
            detail.append(
                f"{hallucinated} number(s) not traceable to the context "
                f"({row.get('ungrounded_prices', 0)} invented, "
                f"{row.get('value_mismatches', 0)} misquoted)"
            )
        if int(row.get("unresolvable_fields") or 0):
            reasons.append(Reject.UNRESOLVABLE_CITATION)
            detail.append(
                f"{row['unresolvable_fields']} citation(s) name a field that does not exist"
            )
        if int(row.get("numbers_in_prose") or 0):
            reasons.append(Reject.NUMBER_IN_PROSE)
            detail.append(f"{row['numbers_in_prose']} number(s) in prose fields")

    if spec.forbid_contradictions and int(row.get("contradiction_count") or 0):
        reasons.append(Reject.CONTRADICTION)
        kinds = row.get("contradictions") or []
        detail.append("internally inconsistent: " + ", ".join(str(k) for k in kinds))

    if spec.require_first_attempt and not row.get("first_attempt_ok", True):
        reasons.append(Reject.NEEDED_REPAIR)
        detail.append(f"valid only after {row.get('attempts', 0)} attempts")

    return Verdict(not reasons, reasons, detail)
