"""Generate teacher analyses over the example set, gate them, and split.

Three files come out, and the split between them is the point:

* `corpus.jsonl`  - accepted rows, each carrying its split assignment
* `rejected.jsonl` - refused rows with the reason, kept deliberately
* `results.jsonl` - the raw checkpoint, every call as it happened

Keeping rejects is not sentimentality. A corpus you cannot interrogate is one
you cannot debug: when the student later fabricates a number, the first
question is whether it learned that from the data, and that question is
unanswerable if the discarded 13% was thrown away.

What a row does *not* contain
-----------------------------
A rendered prompt. The row stores the engine context and the teacher's
analysis; turning those into a training pair is a decision for the training
stage, because the biggest open question - whether the student needs the
100-bar candle window that accounts for 71% of the prompt, or only the
computed features - is answered by training both and measuring, not by
guessing now. Baking a prompt in here would foreclose that experiment.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset import DATASET_SCHEMA_VERSION
from dataset.quality import GateSpec, Verdict, judge
from evaluation.benchmark import score_row
from evaluation.sampling import Example
from teacher.batch import Checkpoint, run_batch
from teacher.registry import Registry, load_registry
from teacher.runner import TeacherRunner

#: Examples at or after this date belong to the validation split. Everything
#: before is training. See `assign_split` for why the boundary is a date.
DEFAULT_VAL_START = "2024-01-01"


@dataclass(frozen=True)
class BuildSpec:
    model: str
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_repairs: int = 2
    concurrency: int = 8
    requests_per_minute: float = 90.0
    allow_mode_fallback: bool = False
    val_start: str = DEFAULT_VAL_START
    gate: GateSpec = field(default_factory=GateSpec)


def assign_split(as_of: str, val_start: str) -> str:
    """Temporal split, not random.

    A random split leaks. Adjacent examples share a market regime - the same
    trend, the same volatility state, often the same news - so a validation
    example drawn from the middle of the training period is partly predictable
    from its neighbours, and the validation loss flatters the model.

    Splitting on a date is also the honest simulation of use: the model will
    be asked about bars later than any it trained on.

    No separation band is needed at the boundary because the sampler already
    guarantees examples within a symbol are at least a full context window
    apart, so no training example's window can reach into a validation one.
    """
    return "val" if str(as_of) >= val_start else "train"


def build(
    examples: list[Example],
    spec: BuildSpec,
    out_dir: Path,
    on_progress=None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    registry = registry or load_registry()
    runner = TeacherRunner(
        registry.build(spec.model, temperature=spec.temperature),
        prompt_version=spec.prompt_version,
        max_repairs=spec.max_repairs,
        allow_mode_fallback=spec.allow_mode_fallback,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(out_dir / "results.jsonl")

    def work(example: Example) -> dict[str, Any]:
        return score_row(spec.model, example, 0, runner.analyse(example.context))

    run_batch(
        examples,
        key_of=lambda e: f"{spec.model}::{e.example_id}",
        work=work,
        checkpoint=checkpoint,
        concurrency=spec.concurrency,
        per_minute=spec.requests_per_minute,
        on_progress=on_progress,
    )

    return finalise(examples, checkpoint.records(), spec, out_dir)


def finalise(
    examples: list[Example], rows: list[dict[str, Any]], spec: BuildSpec, out_dir: Path
) -> dict[str, Any]:
    """Gate, split and write. Separate from `build` so a corpus can be rebuilt
    from an existing checkpoint without spending a single call - which is what
    happens every time the gate is tightened."""
    contexts = {e.example_id: e for e in examples}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for row in rows:
        example = contexts.get(row.get("example_id", ""))
        if example is None:
            continue  # a result whose example is no longer in the set
        verdict = judge(row, spec.gate)
        record = to_record(example, row, verdict, spec)
        (accepted if verdict.accepted else rejected).append(record)

    _write(out_dir / "corpus.jsonl", accepted)
    _write(out_dir / "rejected.jsonl", rejected)
    return summarise(accepted, rejected, spec)


def to_record(
    example: Example, row: dict[str, Any], verdict: Verdict, spec: BuildSpec
) -> dict[str, Any]:
    """One corpus row: the facts, the interpretation, and its provenance.

    Lineage is not decoration. A corpus is a claim about how it was made, and
    a row that cannot name the teacher, the prompt, the engine version and the
    parameter fingerprint that produced it cannot be reproduced, compared
    against a later vintage, or selectively withdrawn when one of those
    changes.
    """
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "example_id": example.example_id,
        "split": assign_split(example.as_of, spec.val_start),
        "symbol": example.symbol,
        "interval": example.context.get("interval"),
        "as_of": example.as_of,
        "stratum": example.stratum,
        "context": example.context,
        "analysis": row.get("analysis"),
        "quality": verdict.as_record()
        | {
            "grounded_claims": row.get("grounded_claims"),
            "total_claims": row.get("total_claims"),
            "contradiction_count": row.get("contradiction_count"),
            "first_attempt_ok": row.get("first_attempt_ok"),
        },
        "lineage": {
            "teacher_model": spec.model,
            "resolved_model": row.get("resolved_model"),
            "served_by": row.get("served_by"),
            "prompt_version": spec.prompt_version,
            "prompt_hash": row.get("prompt_hash"),
            "structured_mode": row.get("structured_mode"),
            "temperature": spec.temperature,
            "engine_version": example.context.get("engine_version"),
            "context_schema_version": example.context.get("schema_version"),
            "params_fingerprint": example.context.get("params_fingerprint"),
        },
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def summarise(
    accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], spec: BuildSpec
) -> dict[str, Any]:
    total = len(accepted) + len(rejected)
    reasons: Counter[str] = Counter()
    for row in rejected:
        for reason in row["quality"]["reasons"]:
            reasons[reason] += 1

    return {
        "teacher": spec.model,
        "generated": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accept_rate": round(len(accepted) / total, 4) if total else 0.0,
        "splits": dict(Counter(row["split"] for row in accepted)),
        "strata": dict(Counter(row["stratum"] for row in accepted)),
        "symbols": len({row["symbol"] for row in accepted}),
        "intervals": dict(Counter(str(row["interval"]) for row in accepted)),
        "reject_reasons": dict(reasons),
        "date_range": [
            min((row["as_of"] for row in accepted), default=""),
            max((row["as_of"] for row in accepted), default=""),
        ],
    }


def render(summary: dict[str, Any], cutoff: str) -> str:
    lines = [
        f"teacher            {summary['teacher']}",
        f"generated          {summary['generated']}",
        f"accepted           {summary['accepted']}  ({summary['accept_rate'] * 100:.1f}%)",
        f"rejected           {summary['rejected']}",
        "",
        "split              " + ", ".join(f"{k}={v}" for k, v in sorted(summary["splits"].items())),
        "interval           "
        + ", ".join(f"{k}={v}" for k, v in sorted(summary["intervals"].items())),
        f"symbols            {summary['symbols']}",
        f"date range         {summary['date_range'][0][:10]} .. {summary['date_range'][1][:10]}",
        f"holdout            everything at or after {cutoff} was never sampled",
    ]
    if summary["reject_reasons"]:
        lines += ["", "rejected because:"]
        for reason, count in sorted(summary["reject_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:24} {count}")

    strata = summary["strata"]
    if strata:
        low = min(strata.values())
        high = max(strata.values())
        lines += ["", f"strata             {len(strata)} populated, {low}-{high} examples each"]
        if low < 5:
            thin = [k for k, v in strata.items() if v < 5]
            lines.append(f"  thin strata: {', '.join(sorted(thin))}")
    return "\n".join(lines)
