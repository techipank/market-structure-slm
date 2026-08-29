"""Turn corpus rows into training pairs, in more than one input view.

Why views exist
---------------
The corpus deliberately stores the context and the analysis separately rather
than a rendered prompt, because the biggest open question in the project is
how much of the context the student actually needs. The candle window is
**68% of the context and 48% of the whole prompt** (8,826 chars, against a
5,656-char system message and 12,890 chars of context), and the student's job
is to interpret facts the engine already computed - not to re-derive them from
raw candles. Sequence length is the dominant cost in fine-tuning, so the
difference between an 11,400-token and a 6,800-token input decides what
hardware the run needs.

That question is settled by training both and measuring, not by guessing. So
this module renders the same corpus several ways and leaves the comparison to
the training stage.

Why the tail view is the default
--------------------------------
Dropping candles entirely is not free. 2.0% of citations point into
`ohlcv_window`, and those are spread across **20.8% of rows** - so a
candle-free prompt would leave one row in five citing evidence the student
cannot see, which is a direct lesson in fabrication.

Measured on the corpus: every cited candle sits within the **last 10 bars**.
A 20-bar tail therefore preserves 100% of citations with margin, at roughly a
fifth of the candle cost. That is the same kind of measurement that chose the
100-bar window for the teacher, and it points somewhere different because the
question is different: the teacher had to *find* swings in raw candles, while
the student is handed them already computed.

Every view is verified rather than trusted - `render_pair` re-resolves every
citation against the trimmed context and refuses to emit a row whose evidence
it just removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from teacher.grounding import resolve_citation
from teacher.prompt import PromptTemplate, build_messages


@dataclass(frozen=True)
class View:
    """One way of showing the context to the student.

    `candle_bars` is the number of trailing candles kept: None keeps the whole
    window (parity with what the teacher saw), 0 removes it entirely.
    """

    name: str
    candle_bars: int | None
    description: str


VIEWS: dict[str, View] = {
    "full": View("full", None, "the whole context, exactly as the teacher saw it"),
    "tail": View("tail", 20, "computed features plus the last 20 candles"),
    "compact": View("compact", 0, "computed features only, no raw candles"),
}

DEFAULT_VIEW = "tail"


def render_context(context: dict[str, Any], view: View) -> dict[str, Any]:
    """Trim the candle window. Nothing else is altered.

    Key order is preserved so the rendered prompt differs from the full view
    only in the candles, which keeps the comparison between views clean.
    """
    if view.candle_bars is None:
        return context

    trimmed = dict(context)
    window = context.get("ohlcv_window") or []
    if view.candle_bars == 0:
        trimmed.pop("ohlcv_window", None)
    else:
        trimmed["ohlcv_window"] = window[-view.candle_bars :]
    return trimmed


def citations(analysis: Any, found: list[str] | None = None) -> list[str]:
    """Every `context_field` anywhere in the analysis."""
    out = found if found is not None else []
    if isinstance(analysis, dict):
        for key, value in analysis.items():
            if key == "context_field" and isinstance(value, str):
                out.append(value)
            else:
                citations(value, out)
    elif isinstance(analysis, list):
        for value in analysis:
            citations(value, out)
    return out


def unresolvable(analysis: Any, context: dict[str, Any]) -> list[str]:
    """Citations that do not resolve against this (possibly trimmed) context.

    The corpus gate already proved every citation resolved against the *full*
    context. This asks the narrower question a view creates: does it still
    resolve against what the student will actually be shown?
    """
    missing = []
    for path in citations(analysis):
        found, _value, _concession = resolve_citation(context, path)
        if not found:
            missing.append(path)
    return missing


def render_pair(
    row: dict[str, Any], view: View, template: PromptTemplate, schema_outline: str | None = None
) -> dict[str, Any] | None:
    """One chat-format training example, or None if the view broke its evidence.

    Returning None rather than raising is deliberate: a view that cannot show
    some rows is a fact about the view, to be counted and reported, not an
    error in the corpus.
    """
    context = render_context(row["context"], view)
    if unresolvable(row["analysis"], context):
        return None

    messages = build_messages(template, context, schema_outline)
    return {
        "example_id": row["example_id"],
        "split": row["split"],
        "view": view.name,
        "messages": [
            *messages,
            # The target is the analysis as JSON: the student is being taught
            # to emit the same structured object the teacher did, which is
            # what makes the two directly comparable at evaluation.
            {"role": "assistant", "content": json.dumps(row["analysis"], separators=(",", ":"))},
        ],
        "lineage": row["lineage"],
    }


#: Characters per token for this prompt family, calibrated against 655 real
#: billed calls: median 1.62, range 1.57-1.68. The generic `chars/4` rule is
#: off by a factor of 2.47 here, and an earlier 1.35x correction - taken from
#: a single call on a shorter context - was still 45% low. JSON structure,
#: ISO timestamps and long decimal prices tokenise far worse than prose.
#:
#: This matters beyond tidiness: it is the number that decides whether a
#: sequence fits in a training run's context budget, and under-reporting it
#: produces an out-of-memory failure halfway through an epoch.
CHARS_PER_TOKEN = 1.62


def estimate_tokens(text: str) -> int:
    """Token count for reporting, calibrated on measured billing."""
    return int(len(text) / CHARS_PER_TOKEN)
