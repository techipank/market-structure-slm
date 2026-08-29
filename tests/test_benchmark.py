"""Benchmark machinery: metrics, sampling, batching, scoring, recommendation.

All offline. The API-calling layer was already covered by the teacher tests
against a stub provider; what is tested here is everything that decides what
gets called and what the numbers mean afterwards.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from evaluation.benchmark import ModelSummary, summarise, write_csv
from evaluation.metrics import (
    Contradiction,
    find_contradictions,
    score_analysis,
    self_consistency,
)
from evaluation.recommend import Gates, evaluate, quality_score, recommend
from evaluation.sampling import SampleSpec, build_samples, read_samples, write_samples
from teacher.batch import Checkpoint, RateLimiter, iter_pairs, run_batch
from teacher.schema import TeacherAnalysis
from tests.test_teacher import CONTEXT, valid_payload


def analysis(**overrides) -> TeacherAnalysis:
    return TeacherAnalysis.model_validate(valid_payload(**overrides))


# ----------------------------------------------------------- agreement


def test_structure_agreement_against_the_engine():
    """`structure.trend` is a definition, so disagreement is error."""
    assert score_analysis(analysis(market_structure="EXPANSION"), CONTEXT).structure_agrees
    assert not score_analysis(analysis(market_structure="UPTREND"), CONTEXT).structure_agrees


def test_structure_agreement_is_skipped_when_the_engine_has_no_opinion():
    context = {**CONTEXT, "structure": {**CONTEXT["structure"], "trend": "INSUFFICIENT_DATA"}}
    assert score_analysis(analysis(), context).structure_agrees is None


def test_claiming_a_higher_timeframe_bias_that_does_not_exist_is_wrong():
    """No closed higher-TF bar means UNAVAILABLE is the only correct answer."""
    assert not score_analysis(analysis(higher_timeframe_bias="BULLISH"), CONTEXT).htf_bias_agrees
    assert score_analysis(analysis(higher_timeframe_bias="UNAVAILABLE"), CONTEXT).htf_bias_agrees


def test_higher_timeframe_bias_compared_when_present():
    context = {**CONTEXT, "higher_timeframe": {"bias": "BULLISH", "trend": "UPTREND"}}
    assert score_analysis(analysis(higher_timeframe_bias="BULLISH"), context).htf_bias_agrees
    assert not score_analysis(analysis(higher_timeframe_bias="BEARISH"), context).htf_bias_agrees


def test_setup_agreement_when_the_engine_offered_none():
    assert score_analysis(analysis(setup_type="NONE"), CONTEXT).setup_agrees
    assert not score_analysis(analysis(setup_type="REVERSAL_LONG"), CONTEXT).setup_agrees


def test_setup_agreement_when_the_engine_offered_one():
    context = {**CONTEXT, "setups": [{"name": "REVERSAL_LONG", "direction": "LONG"}]}
    assert score_analysis(analysis(setup_type="REVERSAL_LONG"), context).setup_agrees
    assert not score_analysis(analysis(setup_type="RANGE_FADE_SHORT"), context).setup_agrees


def test_hallucinated_numbers_exclude_mere_citation_errors():
    """Fabricating a price and misfiling a real one are different failures."""
    score = score_analysis(
        analysis(), CONTEXT,
        grounding_counts={"UNRESOLVABLE_FIELD": 3, "UNGROUNDED_PRICE": 1, "VALUE_MISMATCH": 2},
    )
    assert score.hallucinated_numbers == 3
    assert score.unresolvable_fields == 3
    assert not score.is_grounded


# -------------------------------------------------------- contradictions


def test_no_contradictions_in_a_clean_analysis():
    assert find_contradictions(analysis(higher_timeframe_bias="UNAVAILABLE"), CONTEXT) == []


def test_directional_state_inside_the_opposite_structure():
    found = find_contradictions(
        analysis(market_state="TRENDING_UP", market_structure="DOWNTREND"), CONTEXT
    )
    assert Contradiction.STATE_VS_STRUCTURE in found


def test_long_setup_in_a_falling_market():
    found = find_contradictions(
        analysis(setup_type="TREND_CONTINUATION_LONG", market_state="TRENDING_DOWN"), CONTEXT
    )
    assert Contradiction.SETUP_VS_STATE in found


def test_support_above_the_current_price():
    payload = valid_payload()
    payload["important_levels"][0]["price"] = 770.0  # close is 768.5
    payload["important_levels"][0]["role"] = "SUPPORT"
    found = find_contradictions(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Contradiction.LEVEL_SIDE_WRONG in found


def test_scenarios_that_all_point_the_same_way():
    """The schema can require two scenarios but not two different ones."""
    payload = valid_payload()
    payload["scenarios"][1]["direction"] = "BULLISH"
    found = find_contradictions(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Contradiction.SCENARIOS_ONE_SIDED in found


def test_bullish_scenario_invalidated_above_its_own_trigger():
    payload = valid_payload()
    payload["scenarios"][0].update(
        {"direction": "BULLISH", "trigger_price": 739.51, "invalidation_price": 776.85}
    )
    found = find_contradictions(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Contradiction.SCENARIO_LEVELS_INVERTED in found


def test_the_same_fact_used_for_and_against():
    payload = valid_payload()
    payload["conflicting_evidence"] = [dict(payload["supporting_evidence"][0])]
    found = find_contradictions(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Contradiction.EVIDENCE_DOUBLE_COUNTED in found


def test_high_confidence_against_its_own_counter_case():
    # The schema already demands two supporting items, so the counter-case
    # has to reach three before it outweighs them.
    payload = valid_payload(confidence=0.95)
    payload["conflicting_evidence"] = [
        {"statement": "One.", "context_field": "regime.trend_regime", "value": "MIXED"},
        {"statement": "Two.", "context_field": "volatility.regime", "value": "LOW"},
        {"statement": "Three.", "context_field": "structure.trend", "value": "EXPANSION"},
    ]
    found = find_contradictions(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Contradiction.CONFIDENCE_VS_EVIDENCE in found


def test_setup_named_that_the_engine_never_offered():
    found = find_contradictions(analysis(setup_type="RANGE_FADE_LONG"), CONTEXT)
    assert Contradiction.SETUP_NOT_OFFERED in found


# ---------------------------------------------------------- consistency


def test_identical_analyses_are_perfectly_consistent():
    scores = self_consistency([analysis(), analysis(), analysis()])
    assert scores["market_state"] == 1.0
    assert scores["confidence_spread"] == 0.0


def test_disagreeing_analyses_score_below_one():
    scores = self_consistency([
        analysis(market_state="TRENDING_UP", confidence=0.4),
        analysis(market_state="RANGING", confidence=0.9),
    ])
    assert scores["market_state"] == 0.0
    assert scores["confidence_spread"] == pytest.approx(0.5)


def test_single_run_is_treated_as_consistent_not_as_zero():
    assert self_consistency([analysis()])["market_state"] == 1.0


# ------------------------------------------------------------- sampling


@pytest.fixture(scope="module")
def sample_files() -> list[Path]:
    files = sorted(Path("data/raw").glob("*.csv"))
    if len(files) < 2:
        pytest.skip("no fetched data; run `msdata fetch` to enable sampling tests")
    return files[:2]


def test_sampling_is_deterministic(sample_files):
    from market_engine.params import load_params

    spec = SampleSpec(target_count=20, min_bar_index=400)
    params = load_params()
    a = build_samples(sample_files, params, spec)
    b = build_samples(sample_files, params, spec)
    assert [e.example_id for e in a] == [e.example_id for e in b]


def test_sampling_respects_the_holdout_cutoff(sample_files):
    from market_engine.params import load_params

    spec = SampleSpec(target_count=30, cutoff="2020-01-01", min_bar_index=400)
    for example in build_samples(sample_files, load_params(), spec):
        assert example.as_of < "2020-01-01", "sampled from the reserved period"


def test_selected_bars_never_share_a_context_window(sample_files):
    """Adjacent bars share 59/60 candles; overlapping prompts are duplicates."""
    from collections import defaultdict

    from market_engine.params import load_params

    params = load_params()
    spec = SampleSpec(target_count=40, min_separation=60, min_bar_index=400)
    by_source: dict[str, list[int]] = defaultdict(list)
    for example in build_samples(sample_files, params, spec):
        by_source[example.source_file].append(example.bar_index)

    for source, indices in by_source.items():
        indices.sort()
        gaps = [b - a for a, b in zip(indices, indices[1:], strict=False)]
        assert all(gap >= 60 for gap in gaps), f"{source}: overlapping windows {gaps}"


def test_sampling_spreads_across_strata(sample_files):
    from market_engine.params import load_params

    examples = build_samples(
        sample_files, load_params(), SampleSpec(target_count=40, min_bar_index=400)
    )
    assert len({e.stratum for e in examples}) >= 5


def test_samples_round_trip_through_disk(tmp_path: Path, sample_files):
    from market_engine.params import load_params

    examples = build_samples(
        sample_files, load_params(), SampleSpec(target_count=5, min_bar_index=400)
    )
    path = tmp_path / "s.jsonl"
    write_samples(examples, path)
    restored = read_samples(path)
    assert [e.example_id for e in restored] == [e.example_id for e in examples]
    assert restored[0].context == examples[0].context


# ------------------------------------------------------------- batching


def test_checkpoint_skips_completed_work(tmp_path: Path):
    path = tmp_path / "ck.jsonl"
    calls: list[int] = []

    def work(item: int) -> dict:
        calls.append(item)
        return {"ok": True, "value": item}

    items = [1, 2, 3]
    run_batch(items, str, work, Checkpoint(path), concurrency=1, per_minute=6000)
    assert sorted(calls) == [1, 2, 3]

    calls.clear()
    progress = run_batch(items, str, work, Checkpoint(path), concurrency=1, per_minute=6000)
    assert calls == [], "already-completed work was re-run"
    assert progress.skipped == 3


def test_a_failing_item_is_recorded_not_raised(tmp_path: Path):
    path = tmp_path / "ck.jsonl"

    def work(item: int) -> dict:
        if item == 2:
            raise ValueError("boom")
        return {"ok": True}

    progress = run_batch([1, 2, 3], str, work, Checkpoint(path), concurrency=1, per_minute=6000)
    assert progress.failed == 1
    assert progress.completed == 3

    records = {r["_key"]: r for r in Checkpoint(path).records()}
    assert records["2"]["ok"] is False
    assert "boom" in records["2"]["error"]


def test_a_truncated_final_line_costs_one_item_not_the_run(tmp_path: Path):
    path = tmp_path / "ck.jsonl"
    path.write_text(
        json.dumps({"_key": "1", "ok": True}) + "\n" + '{"_key": "2", "ok"',
        encoding="utf-8",
    )
    checkpoint = Checkpoint(path)
    assert checkpoint.has("1")
    assert not checkpoint.has("2")
    assert len(checkpoint.records()) == 1


def test_checkpoint_appends_are_thread_safe(tmp_path: Path):
    path = tmp_path / "ck.jsonl"
    checkpoint = Checkpoint(path)

    def writer(start: int) -> None:
        for i in range(start, start + 50):
            checkpoint.append(str(i), {"ok": True, "i": i})

    threads = [threading.Thread(target=writer, args=(s,)) for s in (0, 100, 200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(checkpoint.records()) == 150


def test_rate_limiter_throttles():
    limiter = RateLimiter(per_minute=120)  # 2/s, bucket starts full
    for _ in range(120):
        limiter.acquire()
    started = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - started > 0.2


def test_pairs_are_interleaved_by_model():
    """An interrupted run should leave partial coverage of every model."""
    pairs = iter_pairs(["a", "b"], ["x", "y"])
    assert [m for m, _ in pairs] == ["a", "b", "a", "b"]


# ---------------------------------------------------------- aggregation


def _row(model: str, run: int = 0, **overrides) -> dict:
    row = {
        "model": model, "example_id": f"e{overrides.pop('eid', 0)}", "run": run,
        "ok": True, "error": None, "attempts": 1, "first_attempt_ok": True,
        "structured_mode": "JSON_SCHEMA", "prompt_tokens": 5000,
        "completion_tokens": 700, "cost": 0.002, "latency_seconds": 3.0,
        "agreement_points": 3, "agreement_comparable": 3,
        "structure_agrees": True, "htf_bias_agrees": True, "setup_agrees": True,
        "is_grounded": True, "hallucinated_numbers": 0, "numbers_in_prose": 0,
        "unresolvable_fields": 0, "contradictions": [], "contradiction_count": 0,
        "analysis": valid_payload(),
    }
    row.update(overrides)
    return row


def test_summary_rates_are_computed_over_primary_runs_only():
    rows = [_row("m", eid=i) for i in range(4)]
    rows += [_row("m", run=1, eid=0), _row("m", run=2, eid=0)]  # consistency repeats
    summary = summarise(rows)[0]
    assert summary.examples == 4, "repeat runs must not inflate the example count"
    assert summary.calls == 6
    assert summary.schema_compliance_rate == 1.0
    # Projected per-example cost must exclude the repeats.
    assert summary.mean_cost == pytest.approx(0.002)
    assert summary.total_cost == pytest.approx(0.012)


def test_failures_lower_compliance_and_are_grouped():
    rows = [_row("m", eid=0), _row("m", eid=1, ok=False, error="TransientError: down",
                                   first_attempt_ok=False)]
    summary = summarise(rows)[0]
    assert summary.schema_compliance_rate == 0.5
    assert summary.first_attempt_rate == 0.5
    assert summary.errors == {"TransientError": 1}


def test_repairs_show_up_as_a_lower_first_attempt_rate():
    rows = [_row("m", eid=0), _row("m", eid=1, attempts=2, first_attempt_ok=False)]
    summary = summarise(rows)[0]
    assert summary.schema_compliance_rate == 1.0
    assert summary.first_attempt_rate == 0.5
    assert summary.mean_attempts == 1.5


def test_contradictions_are_counted_and_broken_down():
    rows = [
        _row("m", eid=0, contradictions=["STATE_VS_STRUCTURE"], contradiction_count=1),
        _row("m", eid=1, contradictions=["STATE_VS_STRUCTURE", "LEVEL_SIDE_WRONG"],
             contradiction_count=2),
    ]
    summary = summarise(rows)[0]
    assert summary.contradictions_per_example == 1.5
    assert summary.contradiction_breakdown["STATE_VS_STRUCTURE"] == 2


def test_consistency_is_measured_across_repeat_runs():
    rows = [
        _row("m", run=0, eid=0),
        _row("m", run=1, eid=0, analysis=valid_payload(market_state="RANGING")),
    ]
    summary = summarise(rows)[0]
    assert 0.0 < summary.consistency < 1.0


def test_csv_has_a_row_per_model(tmp_path: Path):
    rows = [_row("a", eid=0), _row("b", eid=0)]
    path = tmp_path / "b.csv"
    write_csv(summarise(rows), path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # header + two models
    assert "model" in lines[0] and "mean_cost" in lines[0]


# ------------------------------------------------------- recommendation


def _summary(model: str, **overrides) -> ModelSummary:
    base = dict(
        examples=200, calls=200, schema_compliance_rate=1.0, first_attempt_rate=1.0,
        fact_agreement_rate=0.9, grounded_rate=0.95, contradictions_per_example=0.1,
        consistency=0.95, p50_latency=3.0, p95_latency=6.0, mean_cost=0.01,
    )
    base.update(overrides)
    return ModelSummary(model=model, **base)


def test_quality_weights_sum_to_one():
    from evaluation.recommend import WEIGHTS

    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_a_perfect_model_scores_one():
    perfect = _summary("p", schema_compliance_rate=1.0, grounded_rate=1.0,
                       fact_agreement_rate=1.0, contradictions_per_example=0.0,
                       consistency=1.0, first_attempt_rate=1.0)
    score, _ = quality_score(perfect)
    assert score == pytest.approx(1.0)


def test_gates_disqualify_regardless_of_other_strengths():
    """A weighted mean must not let a cheap model buy past a hallucination rate."""
    hallucinator = _summary("cheap", grounded_rate=0.40, mean_cost=0.0001)
    candidates = evaluate([hallucinator], dataset_size=5000)
    assert not candidates[0].eligible
    assert any("grounded" in reason for reason in candidates[0].disqualified_by)
    assert recommend(candidates) is None


def test_a_model_that_cannot_produce_the_schema_is_disqualified():
    candidates = evaluate([_summary("m", schema_compliance_rate=0.5)], dataset_size=100)
    assert not candidates[0].eligible


def test_the_biggest_model_does_not_win_automatically():
    """The rule that stops the answer being 'use the frontier model'.

    A marginally better model costing twenty times more is buying noise: the
    gap is inside the sampling error of a two-hundred example benchmark.
    """
    frontier = _summary("vendor/frontier", fact_agreement_rate=0.92, mean_cost=0.20)
    cheap = _summary("vendor/small", fact_agreement_rate=0.90, mean_cost=0.01)
    candidates = evaluate([frontier, cheap], dataset_size=5000)

    best = max(candidates, key=lambda c: c.quality)
    assert best.model == "vendor/frontier", "fixture should make the frontier score higher"
    assert recommend(candidates).model == "vendor/small"


def test_a_clearly_better_model_does_win():
    """The tolerance must not be so wide that quality stops mattering."""
    good = _summary("vendor/good", grounded_rate=0.99, fact_agreement_rate=0.95,
                    consistency=0.99, mean_cost=0.20)
    weak = _summary("vendor/weak", grounded_rate=0.82, fact_agreement_rate=0.70,
                    consistency=0.70, contradictions_per_example=0.9, mean_cost=0.01)
    assert recommend(evaluate([good, weak], dataset_size=5000)).model == "vendor/good"


def test_projected_cost_includes_a_retry_allowance():
    candidate = evaluate([_summary("m", mean_cost=0.01, schema_compliance_rate=0.95)],
                         dataset_size=1000)[0]
    assert candidate.projected_cost == pytest.approx(0.01 * 1000 / 0.95)


def test_unmeasured_consistency_is_neutral_not_zero():
    """Not taking a measurement is our omission, not the model's failure."""
    measured = quality_score(_summary("a", consistency=0.5))[0]
    unmeasured = quality_score(_summary("b", consistency=float("nan")))[0]
    assert measured == pytest.approx(unmeasured)


def test_report_states_the_recommendation_and_the_tradeoff():
    from evaluation.recommend import render_report

    frontier = _summary("vendor/frontier", fact_agreement_rate=0.92, mean_cost=0.20)
    cheap = _summary("vendor/small", fact_agreement_rate=0.90, mean_cost=0.01)
    report = render_report(evaluate([frontier, cheap], dataset_size=5000), 5000)

    assert "vendor/small" in report
    assert "Not the highest-quality model" in report
    assert "Projected cost" in report
    assert "credits" in report


def test_report_refuses_to_recommend_when_nothing_qualifies():
    from evaluation.recommend import render_report

    report = render_report(
        evaluate([_summary("m", grounded_rate=0.1)], dataset_size=5000), 5000
    )
    assert "No model cleared the gates" in report
    assert "Do not start dataset generation" in report


def test_gates_are_configurable():
    strict = Gates(min_schema_compliance=0.99)
    candidates = evaluate([_summary("m", schema_compliance_rate=0.95)],
                          dataset_size=100, gates=strict)
    assert not candidates[0].eligible


def test_consistency_subset_is_spread_not_the_head():
    """Taking the head would measure one symbol in one era."""
    from evaluation.benchmark import _consistency_subset
    from evaluation.sampling import Example

    examples = [
        Example(f"e{i}", f"SYM{i // 20}", "1d", "f.csv", i, "2020-01-01", "S|LOW", {})
        for i in range(100)
    ]
    subset = _consistency_subset(examples, 5)
    assert len(subset) == 5
    symbols = {e.symbol for e in examples if e.example_id in subset}
    assert len(symbols) > 1, "subset came from a single symbol"
    assert _consistency_subset(examples, 5) == subset, "must be deterministic"
    assert _consistency_subset(examples, 0) == set()


def test_sampling_is_balanced_across_strata(sample_files):
    """Stratum allocation must not depend on stratum *name*.

    Two bugs were caught here. The strata were iterated in sorted name order,
    and because the separation budget is shared per symbol, whichever strata
    were served first consumed it -- producing a monotonic decline from
    CONTRACTION (11-12 examples) to UPTREND (1-7) purely by alphabet. Then the
    inner loop covered strata x sources in a single pass, so one pass could
    take the entire target and the round-robin never rotated.

    Either defect under-samples whole market states, which surfaces much later
    as "the model is bad at uptrends" and gets blamed on the model.
    """
    from collections import Counter

    from market_engine.params import load_params

    examples = build_samples(
        sample_files, load_params(), SampleSpec(target_count=60, min_bar_index=300)
    )
    counts = Counter(e.stratum for e in examples)
    assert len(counts) >= 4, "fixture did not produce enough strata to test balance"

    lo, hi = min(counts.values()), max(counts.values())
    assert hi - lo <= 2, f"stratum allocation is lopsided: {dict(counts)}"

    # And the imbalance must not correlate with alphabetical position, which
    # is the specific signature of the original bug.
    ordered = [counts[name] for name in sorted(counts)]
    first_half = sum(ordered[: len(ordered) // 2])
    second_half = sum(ordered[len(ordered) // 2 :])
    assert abs(first_half - second_half) <= max(3, len(ordered)), (
        f"early-alphabet strata are favoured: {dict(counts)}"
    )
