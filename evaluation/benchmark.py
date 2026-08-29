"""Run every candidate model over the frozen sample set and aggregate.

One row per (model, example) goes to the checkpoint; the aggregate is computed
from those rows afterwards. Keeping raw per-example results rather than only
running totals means a new metric can be added later without paying for the
calls again - which matters when the calls cost money.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    AnalysisScore,
    Contradiction,
    score_analysis,
    self_consistency,
)
from evaluation.sampling import Example
from teacher.batch import Checkpoint, iter_pairs, run_batch
from teacher.registry import Registry, load_registry
from teacher.runner import TeacherResult, TeacherRunner
from teacher.schema import TeacherAnalysis


@dataclass(frozen=True)
class BenchmarkSpec:
    models: tuple[str, ...]
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_repairs: int = 2
    concurrency: int = 4
    requests_per_minute: float = 60.0
    #: Examples re-run this many times to measure self-consistency. Applied to
    #: a small subset only - it multiplies cost directly.
    consistency_repeats: int = 3
    consistency_examples: int = 15
    #: Disabling the structured-mode fallback keeps the comparison honest: a
    #: model that cannot do strict schema should show up as failing, not be
    #: quietly given an easier task than its competitors.
    allow_mode_fallback: bool = False


def _consistency_subset(examples: list[Example], count: int) -> set[str]:
    """Evenly strided, not the first N.

    The sample list is sorted by symbol, so taking the head would measure
    self-consistency on one instrument in one era. Striding spreads the subset
    across symbols and strata. Deterministic, so every model is re-run on the
    same examples.
    """
    if count <= 0 or not examples:
        return set()
    if count >= len(examples):
        return {e.example_id for e in examples}
    step = len(examples) / count
    return {examples[int(i * step)].example_id for i in range(count)}


def _runner(alias: str, spec: BenchmarkSpec, registry: Registry) -> TeacherRunner:
    return TeacherRunner(
        registry.build(alias, temperature=spec.temperature),
        prompt_version=spec.prompt_version,
        max_repairs=spec.max_repairs,
        allow_mode_fallback=spec.allow_mode_fallback,
    )


def run_benchmark(
    examples: list[Example],
    spec: BenchmarkSpec,
    checkpoint_path: Path,
    on_progress=None,
    registry: Registry | None = None,
) -> Checkpoint:
    registry = registry or load_registry()
    runners = {alias: _runner(alias, spec, registry) for alias in spec.models}
    checkpoint = Checkpoint(checkpoint_path)

    consistency_ids = _consistency_subset(examples, spec.consistency_examples)

    jobs: list[tuple[str, Example, int]] = []
    for model, example in iter_pairs(spec.models, examples):
        repeats = spec.consistency_repeats if example.example_id in consistency_ids else 1
        jobs.extend((model, example, run) for run in range(repeats))

    def key_of(job: tuple[str, Example, int]) -> str:
        model, example, run = job
        return f"{model}::{example.example_id}::{run}"

    def work(job: tuple[str, Example, int]) -> dict[str, Any]:
        model, example, run = job
        result = runners[model].analyse(example.context)
        return score_row(model, example, run, result)

    run_batch(
        jobs,
        key_of=key_of,
        work=work,
        checkpoint=checkpoint,
        concurrency=spec.concurrency,
        per_minute=spec.requests_per_minute,
        on_progress=on_progress,
    )
    return checkpoint


def score_row(model: str, example: Example, run: int, result: TeacherResult) -> dict[str, Any]:
    """Flatten one call into a checkpoint row, scored.

    Public because the corpus builder scores its calls the same way. Two
    scorers over the same responses would eventually disagree, and then no
    one could say which number was right.
    """
    record = result.to_record()
    counts: dict[str, int] = defaultdict(int)
    for issue in record["grounding"]["issues"]:
        counts[str(issue["finding"])] += 1

    row: dict[str, Any] = {
        "model": model,
        "example_id": example.example_id,
        "run": run,
        "symbol": example.symbol,
        "stratum": example.stratum,
        "as_of": example.as_of,
        "ok": result.ok,
        "error": result.error,
        "attempts": len(result.attempts),
        # Valid on the first try, before any repair: the honest measure of how
        # well a model follows the contract unaided.
        "first_attempt_ok": bool(result.attempts and result.attempts[0].ok),
        "structured_mode": result.structured_mode,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost": result.cost,
        "latency_seconds": round(result.latency_seconds, 3),
        "resolved_model": result.resolved_model,
        "served_by": result.served_by,
        "prompt_hash": result.prompt_hash,
    }

    if result.ok and result.analysis is not None:
        score = score_analysis(
            result.analysis,
            example.context,
            grounding_counts=dict(counts),
            grounded_claims=record["grounding"]["grounded_claims"],
            total_claims=record["grounding"]["total_claims"],
        )
        row.update(_score_fields(score))
        row["analysis"] = record["analysis"]
    return row


def _score_fields(score: AnalysisScore) -> dict[str, Any]:
    agreed, comparable = score.agreement_points
    return {
        "structure_agrees": score.structure_agrees,
        "htf_bias_agrees": score.htf_bias_agrees,
        "setup_agrees": score.setup_agrees,
        "agreement_points": agreed,
        "agreement_comparable": comparable,
        "grounded_claims": score.grounded_claims,
        "total_claims": score.total_claims,
        "hallucinated_numbers": score.hallucinated_numbers,
        "ungrounded_prices": score.ungrounded_prices,
        "value_mismatches": score.value_mismatches,
        "unresolvable_fields": score.unresolvable_fields,
        "numbers_in_prose": score.numbers_in_prose,
        "is_grounded": score.is_grounded,
        "contradictions": [c.value for c in score.contradictions],
        "contradiction_count": score.contradiction_count,
    }


# ------------------------------------------------------------- aggregation


@dataclass
class ModelSummary:
    model: str
    examples: int = 0
    calls: int = 0
    valid_json_rate: float = 0.0
    schema_compliance_rate: float = 0.0
    first_attempt_rate: float = 0.0
    mean_attempts: float = 0.0
    fact_agreement_rate: float = 0.0
    structure_agreement_rate: float = 0.0
    htf_agreement_rate: float = 0.0
    setup_agreement_rate: float = 0.0
    grounded_rate: float = 0.0
    hallucinated_numbers_per_example: float = 0.0
    numbers_in_prose_per_example: float = 0.0
    contradictions_per_example: float = 0.0
    consistency: float = 0.0
    confidence_spread: float = 0.0
    mean_confidence: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    #: None when the provider reported no cost for any call.
    mean_cost: float | None = None
    total_cost: float | None = None
    contradiction_breakdown: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)


def summarise(rows: list[dict[str, Any]]) -> list[ModelSummary]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    return [_summarise_model(model, group) for model, group in sorted(by_model.items())]


def _summarise_model(model: str, rows: list[dict[str, Any]]) -> ModelSummary:
    # First run of each example only, so the consistency subset does not get
    # triple weight in every other metric.
    primary = [r for r in rows if r.get("run", 0) == 0]
    ok_rows = [r for r in primary if r.get("ok")]

    summary = ModelSummary(model=model, examples=len(primary), calls=len(rows))
    if not primary:
        return summary

    n = len(primary)
    summary.schema_compliance_rate = len(ok_rows) / n
    summary.first_attempt_rate = sum(bool(r.get("first_attempt_ok")) for r in primary) / n
    # A first attempt that parsed is valid JSON by definition; a later success
    # means the first response was not usable as-is.
    summary.valid_json_rate = summary.first_attempt_rate
    summary.mean_attempts = statistics.fmean(r.get("attempts", 0) for r in primary)

    for row in primary:
        if row.get("error"):
            label = str(row["error"]).split(":")[0][:60]
            summary.errors[label] = summary.errors.get(label, 0) + 1

    if ok_rows:
        agreed = sum(r.get("agreement_points", 0) for r in ok_rows)
        comparable = sum(r.get("agreement_comparable", 0) for r in ok_rows)
        summary.fact_agreement_rate = agreed / comparable if comparable else 0.0
        summary.structure_agreement_rate = _rate(ok_rows, "structure_agrees")
        summary.htf_agreement_rate = _rate(ok_rows, "htf_bias_agrees")
        summary.setup_agreement_rate = _rate(ok_rows, "setup_agrees")

        summary.grounded_rate = sum(bool(r.get("is_grounded")) for r in ok_rows) / len(ok_rows)
        summary.hallucinated_numbers_per_example = statistics.fmean(
            r.get("hallucinated_numbers", 0) for r in ok_rows
        )
        summary.numbers_in_prose_per_example = statistics.fmean(
            r.get("numbers_in_prose", 0) for r in ok_rows
        )
        summary.contradictions_per_example = statistics.fmean(
            r.get("contradiction_count", 0) for r in ok_rows
        )
        for row in ok_rows:
            for name in row.get("contradictions", []):
                summary.contradiction_breakdown[name] = (
                    summary.contradiction_breakdown.get(name, 0) + 1
                )
        confidences = [
            (r.get("analysis") or {}).get("confidence")
            for r in ok_rows
            if (r.get("analysis") or {}).get("confidence") is not None
        ]
        if confidences:
            summary.mean_confidence = statistics.fmean(confidences)

    latencies = sorted(r.get("latency_seconds", 0.0) for r in primary)
    if latencies:
        summary.p50_latency = _quantile(latencies, 0.50)
        summary.p95_latency = _quantile(latencies, 0.95)

    summary.mean_prompt_tokens = statistics.fmean(r.get("prompt_tokens", 0) for r in primary)
    summary.mean_completion_tokens = statistics.fmean(
        r.get("completion_tokens", 0) for r in primary
    )

    costs = [r["cost"] for r in rows if r.get("cost") is not None]
    if costs:
        summary.total_cost = sum(costs)
        # Per *example*, using primary rows only, so the consistency repeats
        # do not inflate the projected dataset cost.
        primary_costs = [r["cost"] for r in primary if r.get("cost") is not None]
        if primary_costs:
            summary.mean_cost = statistics.fmean(primary_costs)

    _add_consistency(summary, rows)
    return summary


def _add_consistency(summary: ModelSummary, rows: list[dict[str, Any]]) -> None:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("ok") and row.get("analysis"):
            by_example[row["example_id"]].append(row)

    agreements: list[float] = []
    spreads: list[float] = []
    for group in by_example.values():
        if len(group) < 2:
            continue
        analyses = [TeacherAnalysis.model_validate(r["analysis"]) for r in group]
        scores = self_consistency(analyses)
        categorical = [v for k, v in scores.items() if k != "confidence_spread"]
        agreements.append(statistics.fmean(categorical))
        spreads.append(scores["confidence_spread"])

    summary.consistency = statistics.fmean(agreements) if agreements else float("nan")
    summary.confidence_spread = statistics.fmean(spreads) if spreads else float("nan")


def _rate(rows: list[dict[str, Any]], field_name: str) -> float:
    values = [r[field_name] for r in rows if r.get(field_name) is not None]
    return sum(bool(v) for v in values) / len(values) if values else 0.0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


# ------------------------------------------------------------------- csv

CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "examples",
    "calls",
    "valid_json_rate",
    "schema_compliance_rate",
    "first_attempt_rate",
    "mean_attempts",
    "fact_agreement_rate",
    "structure_agreement_rate",
    "htf_agreement_rate",
    "setup_agreement_rate",
    "grounded_rate",
    "hallucinated_numbers_per_example",
    "numbers_in_prose_per_example",
    "contradictions_per_example",
    "consistency",
    "confidence_spread",
    "mean_confidence",
    "p50_latency",
    "p95_latency",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "mean_cost",
    "total_cost",
)


def write_csv(summaries: list[ModelSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            row = {name: getattr(summary, name) for name in CSV_COLUMNS}
            for name, value in row.items():
                if isinstance(value, float):
                    row[name] = round(value, 6)
            writer.writerow(row)


def all_contradictions() -> list[str]:
    return [c.value for c in Contradiction]
