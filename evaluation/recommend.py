"""Turn benchmark numbers into an argued recommendation.

The instruction this module exists to obey: **do not automatically choose the
largest model.** So the scoring is written down, the weights are visible, and
the tradeoff table is printed alongside the winner. A recommendation you cannot
argue with is a recommendation you cannot check.

Quality is a weighted sum of six rates, all already on a 0-1 scale. The weights
encode what actually matters for *this* job - producing training data:

    schema compliance   0.20   unusable output is worth nothing, whatever it says
    grounded rate       0.25   the single largest source of poisoned labels
    fact agreement      0.20   disagreement with a deterministic fact is error
    no contradictions   0.15   internally inconsistent reasoning teaches nonsense
    self consistency    0.15   an inconsistent teacher distils noise
    valid first attempt 0.05   convenience and cost, not correctness

Grounding outweighs raw compliance because a schema-valid analysis full of
invented prices is worse than a malformed one: the malformed one gets rejected,
the invented one gets trained on.

Gates come before scoring. A model below the minimum on compliance or grounding
is disqualified regardless of how it scores elsewhere - averaging cannot rescue
output that is unusable as ground truth, and a weighted mean would let a fast
cheap model buy its way past a 60% hallucination rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evaluation.benchmark import ModelSummary

WEIGHTS: dict[str, float] = {
    "schema_compliance_rate": 0.20,
    "grounded_rate": 0.25,
    "fact_agreement_rate": 0.20,
    "no_contradiction_rate": 0.15,
    "consistency": 0.15,
    "first_attempt_rate": 0.05,
}


@dataclass(frozen=True)
class Gates:
    """Minimums a model must clear to be considered at all."""

    min_schema_compliance: float = 0.90
    min_grounded_rate: float = 0.80
    max_contradictions_per_example: float = 1.0
    max_p95_latency: float = 120.0


#: Module singleton so the default is not constructed per call.
DEFAULT_GATES = Gates()


@dataclass
class Candidate:
    summary: ModelSummary
    quality: float
    components: dict[str, float]
    disqualified_by: list[str] = field(default_factory=list)
    projected_cost: float | None = None
    quality_per_cost: float | None = None

    @property
    def model(self) -> str:
        return self.summary.model

    @property
    def eligible(self) -> bool:
        return not self.disqualified_by


def quality_components(summary: ModelSummary) -> dict[str, float]:
    """Each component normalised to 0-1, higher is better."""
    # Contradictions are a count, not a rate. One per example is treated as the
    # floor of usefulness, matching the gate, so the two cannot disagree.
    no_contradiction = max(0.0, 1.0 - min(1.0, summary.contradictions_per_example))
    consistency = summary.consistency
    if math.isnan(consistency):
        # Not measured (no repeats). Neutral rather than zero: scoring a model
        # down for a measurement we chose not to take would be our error.
        consistency = 0.5
    return {
        "schema_compliance_rate": summary.schema_compliance_rate,
        "grounded_rate": summary.grounded_rate,
        "fact_agreement_rate": summary.fact_agreement_rate,
        "no_contradiction_rate": no_contradiction,
        "consistency": consistency,
        "first_attempt_rate": summary.first_attempt_rate,
    }


def quality_score(summary: ModelSummary) -> tuple[float, dict[str, float]]:
    components = quality_components(summary)
    total = sum(components[name] * weight for name, weight in WEIGHTS.items())
    return total, components


def check_gates(summary: ModelSummary, gates: Gates) -> list[str]:
    failures: list[str] = []
    if summary.schema_compliance_rate < gates.min_schema_compliance:
        failures.append(
            f"schema compliance {summary.schema_compliance_rate:.0%} "
            f"< {gates.min_schema_compliance:.0%}"
        )
    if summary.grounded_rate < gates.min_grounded_rate:
        failures.append(
            f"grounded rate {summary.grounded_rate:.0%} < {gates.min_grounded_rate:.0%}"
        )
    if summary.contradictions_per_example > gates.max_contradictions_per_example:
        failures.append(
            f"contradictions/example {summary.contradictions_per_example:.2f} "
            f"> {gates.max_contradictions_per_example:.2f}"
        )
    if summary.p95_latency > gates.max_p95_latency:
        failures.append(
            f"p95 latency {summary.p95_latency:.1f}s > {gates.max_p95_latency:.0f}s"
        )
    return failures


def evaluate(
    summaries: list[ModelSummary],
    dataset_size: int,
    gates: Gates = DEFAULT_GATES,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for summary in summaries:
        quality, components = quality_score(summary)
        candidate = Candidate(
            summary=summary,
            quality=quality,
            components=components,
            disqualified_by=check_gates(summary, gates),
        )
        if summary.mean_cost is not None:
            # Projected on cost per *successful* example: a model that fails a
            # tenth of the time still has to produce the full dataset, so the
            # retries are part of its real price.
            success = max(summary.schema_compliance_rate, 1e-6)
            candidate.projected_cost = summary.mean_cost * dataset_size / success
            if candidate.projected_cost > 0:
                candidate.quality_per_cost = quality / candidate.projected_cost
        candidates.append(candidate)

    candidates.sort(key=lambda c: (c.eligible, c.quality), reverse=True)
    return candidates


def recommend(candidates: list[Candidate], tolerance: float = 0.03) -> Candidate | None:
    """Cheapest model within `tolerance` quality of the best eligible one.

    This is the rule that stops the answer being "use the biggest". If a model
    costing a tenth as much scores within three points of the leader, the extra
    spend is buying noise - three points on a two-hundred example benchmark is
    inside the sampling error anyway.

    Cost-blind models (no reported cost) can win on quality but never on price.
    """
    eligible = [c for c in candidates if c.eligible]
    if not eligible:
        return None

    best = max(eligible, key=lambda c: c.quality)
    contenders = [c for c in eligible if c.quality >= best.quality - tolerance]
    priced = [c for c in contenders if c.projected_cost is not None]
    if not priced:
        return best
    # Cheapest, then highest quality. The tie-break matters more than it
    # looks: when several candidates are free the projected costs are all
    # 0.0, and without it the winner would be decided by list order - which
    # is to say, arbitrarily. With it, a field of free models collapses
    # sensibly to "best quality".
    return min(priced, key=lambda c: (c.projected_cost, -c.quality))


# ------------------------------------------------------------------ report


def render_report(
    candidates: list[Candidate],
    dataset_size: int,
    gates: Gates = DEFAULT_GATES,
    tolerance: float = 0.03,
) -> str:
    winner = recommend(candidates, tolerance)
    lines = [
        "# Teacher model benchmark",
        "",
        "> Educational and research use only. Not trading advice.",
        "",
        f"Every model saw the same frozen examples. Projections assume a "
        f"{dataset_size:,}-example dataset.",
        "",
        "## Quality / cost / latency",
        "",
        "| model | quality | schema | grounded | fact agree | contradictions | "
        "consistency | p50 lat | cost/example | projected |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate in candidates:
        s = candidate.summary
        cost = "—" if s.mean_cost is None else f"{s.mean_cost:.5f}"
        projected = (
            "—" if candidate.projected_cost is None else f"{candidate.projected_cost:.2f}"
        )
        consistency = "—" if math.isnan(s.consistency) else f"{s.consistency:.0%}"
        flag = "" if candidate.eligible else " ⚠"
        lines.append(
            f"| `{s.model}`{flag} | {candidate.quality:.3f} | "
            f"{s.schema_compliance_rate:.0%} | {s.grounded_rate:.0%} | "
            f"{s.fact_agreement_rate:.0%} | {s.contradictions_per_example:.2f} | "
            f"{consistency} | {s.p50_latency:.1f}s | {cost} | {projected} |"
        )

    lines += ["", "Costs are in OpenRouter credits, as reported by the API.", ""]

    disqualified = [c for c in candidates if not c.eligible]
    if disqualified:
        lines += ["## Disqualified", ""]
        for candidate in disqualified:
            reasons = "; ".join(candidate.disqualified_by)
            lines.append(f"- `{candidate.model}` — {reasons}")
        lines.append("")

    lines += ["## Scoring", "", "Quality is a weighted sum, all components 0-1:", ""]
    lines += [f"- `{name}` × {weight}" for name, weight in WEIGHTS.items()]
    lines += [
        "",
        "Gates are applied first and are not tradeable: "
        f"schema compliance ≥ {gates.min_schema_compliance:.0%}, "
        f"grounded ≥ {gates.min_grounded_rate:.0%}, "
        f"contradictions ≤ {gates.max_contradictions_per_example:.2f}/example, "
        f"p95 latency ≤ {gates.max_p95_latency:.0f}s.",
        "",
    ]

    lines += ["## Recommendation", ""]
    if winner is None:
        lines += [
            "**No model cleared the gates.** Do not start dataset generation. "
            "Either the prompt needs work or the candidate list does — "
            "check the contradiction breakdown and error columns before "
            "spending anything.",
            "",
        ]
    else:
        best = max((c for c in candidates if c.eligible), key=lambda c: c.quality)
        lines.append(f"**`{winner.model}`**")
        lines.append("")
        if winner.model != best.model:
            lines.append(
                f"Not the highest-quality model. `{best.model}` scores "
                f"{best.quality:.3f} against {winner.quality:.3f}, a gap of "
                f"{best.quality - winner.quality:.3f} — within the "
                f"{tolerance:.2f} tolerance, and inside the sampling error of a "
                f"{candidates[0].summary.examples}-example benchmark. "
            )
            if winner.projected_cost and best.projected_cost:
                ratio = best.projected_cost / winner.projected_cost
                lines.append(
                    f"The projected dataset cost differs by {ratio:.1f}×, so the "
                    "extra spend would buy noise."
                )
        else:
            lines.append("Highest quality among eligible models, and not beaten on cost.")
        lines.append("")
        if winner.projected_cost is not None:
            lines.append(
                f"Projected cost for {dataset_size:,} examples: "
                f"**{winner.projected_cost:.2f} credits** "
                f"(includes a retry allowance for its "
                f"{1 - winner.summary.schema_compliance_rate:.0%} failure rate)."
            )
        lines.append("")

    lines += ["## Component detail", ""]
    for candidate in candidates:
        lines.append(f"### `{candidate.model}`")
        lines.append("")
        for name, value in candidate.components.items():
            lines.append(f"- {name}: {value:.3f}")
        breakdown = candidate.summary.contradiction_breakdown
        if breakdown:
            top = sorted(breakdown.items(), key=lambda kv: -kv[1])
            lines.append(
                "- contradictions: "
                + ", ".join(f"{name} ×{count}" for name, count in top)
            )
        if candidate.summary.errors:
            lines.append(
                "- errors: "
                + ", ".join(f"{name} ×{count}" for name, count in candidate.summary.errors.items())
            )
        lines.append("")

    return "\n".join(lines)
