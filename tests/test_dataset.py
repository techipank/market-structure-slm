"""The gate decides what the student learns from, so it is tested harder than
its size suggests. Every rule here corresponds to a defect measured on a real
benchmark run; the tests name the defect rather than the code path."""

from __future__ import annotations

import json
from pathlib import Path

from dataset.build import BuildSpec, assign_split, finalise, summarise, to_record
from dataset.quality import GateSpec, Reject, judge
from evaluation.sampling import Example

CONTEXT = {
    "schema_version": "1.1.0",
    "engine_version": "1.0.0",
    "params_fingerprint": "abc123",
    "symbol": "INFY_NS",
    "interval": "1d",
    "as_of": "2023-05-04T00:00:00+00:00",
    "bar_index": 900,
}

ANALYSIS = {"market_state": "TRENDING_UP", "confidence": 0.55}


def clean_row(**overrides) -> dict:
    row = {
        "model": "mimo",
        "example_id": "INFY_NS_1d#900",
        "ok": True,
        "error": None,
        "analysis": dict(ANALYSIS),
        "attempts": 1,
        "first_attempt_ok": True,
        "grounded_claims": 12,
        "total_claims": 12,
        "hallucinated_numbers": 0,
        "ungrounded_prices": 0,
        "value_mismatches": 0,
        "unresolvable_fields": 0,
        "numbers_in_prose": 0,
        "contradictions": [],
        "contradiction_count": 0,
        "resolved_model": "mimo-v2.5",
        "prompt_hash": "3d865cbca534b5d5",
        "structured_mode": "JSON_SCHEMA",
    }
    row.update(overrides)
    return row


def example(
    example_id: str = "INFY_NS_1d#900", as_of: str = "2023-05-04T00:00:00+00:00"
) -> Example:
    context = dict(CONTEXT, as_of=as_of)
    return Example(
        example_id=example_id,
        symbol="INFY_NS",
        as_of=as_of,
        bar_index=900,
        stratum="UPTREND|LOW",
        interval="1d",
        source_file="INFY_NS_1d.csv",
        context=context,
    )


# ------------------------------------------------------------------- gate


def test_a_clean_answer_is_accepted():
    assert judge(clean_row()).accepted


def test_an_invented_price_is_rejected():
    """The defect the whole project exists to avoid: a number with no source.
    One of these in the corpus is a lesson in fabrication."""
    verdict = judge(clean_row(ungrounded_prices=1, hallucinated_numbers=1))
    assert not verdict.accepted
    assert Reject.UNGROUNDED_NUMBER in verdict.reasons
    assert "invented" in verdict.detail[0]


def test_a_misquoted_value_is_rejected_the_same_way():
    """Citing a real field and stating the wrong value for it is not milder
    than inventing one - the student cannot tell the difference either."""
    verdict = judge(clean_row(value_mismatches=1, hallucinated_numbers=1))
    assert Reject.UNGROUNDED_NUMBER in verdict.reasons


def test_an_unresolvable_citation_is_rejected():
    verdict = judge(clean_row(unresolvable_fields=2))
    assert Reject.UNRESOLVABLE_CITATION in verdict.reasons


def test_digits_in_prose_are_rejected():
    verdict = judge(clean_row(numbers_in_prose=1))
    assert Reject.NUMBER_IN_PROSE in verdict.reasons


def test_a_contradiction_is_rejected_and_named():
    verdict = judge(clean_row(contradictions=["LEVEL_SIDE_WRONG"], contradiction_count=1))
    assert Reject.CONTRADICTION in verdict.reasons
    assert "LEVEL_SIDE_WRONG" in verdict.detail[0]


def test_a_failed_call_is_rejected_without_further_checks():
    verdict = judge(clean_row(ok=False, analysis=None, error="TransientError: down"))
    assert verdict.reasons == [Reject.CALL_FAILED]


def test_every_reason_is_reported_not_just_the_first():
    """The rejection log is meant to be counted. Stopping at the first defect
    would understate every mode after it."""
    verdict = judge(clean_row(numbers_in_prose=1, unresolvable_fields=1, contradiction_count=1,
                              contradictions=["STATE_VS_STRUCTURE"]))
    assert len(verdict.reasons) == 3


def test_repaired_answers_pass_by_default_and_fail_when_required():
    """Measured first-attempt rate was 100%, so this rule costs nothing today.
    It exists so a weaker teacher cannot quietly fill the corpus with its own
    corrections."""
    row = clean_row(first_attempt_ok=False, attempts=2)
    assert judge(row).accepted
    assert not judge(row, GateSpec(require_first_attempt=True)).accepted


# ------------------------------------------------------------------ split


def test_split_is_temporal():
    assert assign_split("2023-12-31T00:00:00+00:00", "2024-01-01") == "train"
    assert assign_split("2024-01-01T00:00:00+00:00", "2024-01-01") == "val"
    assert assign_split("2024-06-30T00:00:00+00:00", "2024-01-01") == "val"


# ------------------------------------------------------- record and writing


def test_a_row_records_enough_to_reproduce_or_distrust_it():
    record = to_record(example(), clean_row(), judge(clean_row()), BuildSpec(model="mimo"))
    lineage = record["lineage"]
    assert lineage["teacher_model"] == "mimo"
    assert lineage["resolved_model"] == "mimo-v2.5"
    assert lineage["prompt_hash"] == "3d865cbca534b5d5"
    assert lineage["engine_version"] == "1.0.0"
    assert lineage["context_schema_version"] == "1.1.0"
    assert lineage["params_fingerprint"] == "abc123"
    # The context travels with the analysis: a corpus row that cannot be
    # re-verified is a claim without evidence.
    assert record["context"]["bar_index"] == 900
    assert record["analysis"] == ANALYSIS


def test_no_rendered_prompt_is_stored():
    """Whether the student needs the candle window is an open question, and
    baking a prompt into the corpus would answer it by accident."""
    record = to_record(example(), clean_row(), judge(clean_row()), BuildSpec(model="mimo"))
    assert "prompt" not in record
    assert "messages" not in record


def test_finalise_separates_accepted_from_rejected(tmp_path: Path):
    examples = [example("a", "2023-01-02T00:00:00+00:00"),
                example("b", "2024-03-04T00:00:00+00:00")]
    rows = [
        clean_row(example_id="a"),
        clean_row(example_id="b", numbers_in_prose=1),
    ]
    summary = finalise(examples, rows, BuildSpec(model="mimo"), tmp_path)

    assert summary["accepted"] == 1 and summary["rejected"] == 1
    assert summary["reject_reasons"] == {"NUMBER_IN_PROSE": 1}

    corpus = [json.loads(x) for x in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    rejected = [json.loads(x) for x in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert [r["example_id"] for r in corpus] == ["a"]
    # Rejects are kept, with their reason, so the corpus can be interrogated.
    assert rejected[0]["quality"]["reasons"] == ["NUMBER_IN_PROSE"]


def test_results_without_a_matching_example_are_skipped(tmp_path: Path):
    """Re-gating after the example set changed must not resurrect orphans."""
    summary = finalise([example("a")], [clean_row(example_id="ghost")],
                       BuildSpec(model="mimo"), tmp_path)
    assert summary["generated"] == 0


def test_summary_counts_splits_and_intervals():
    spec = BuildSpec(model="mimo")
    accepted = [
        to_record(example("a", "2023-01-02T00:00:00+00:00"), clean_row(), judge(clean_row()), spec),
        to_record(example("b", "2024-03-04T00:00:00+00:00"), clean_row(), judge(clean_row()), spec),
    ]
    summary = summarise(accepted, [], spec)
    assert summary["splits"] == {"train": 1, "val": 1}
    assert summary["intervals"] == {"1d": 2}
    assert summary["accept_rate"] == 1.0


# ------------------------------------------------------------- rendering


def _corpus_row(window_bars: int = 100, cited_index: int | None = None) -> dict:
    # Timestamps must be unique across the window. An earlier version of this
    # fixture wrapped at 28 days, so a "deep" citation also matched a recent
    # bar and the test silently proved nothing.
    from datetime import UTC, datetime, timedelta

    start = datetime(2023, 1, 2, tzinfo=UTC)
    window = [
        {"t": (start + timedelta(days=i)).isoformat(), "o": 10.0 + i, "h": 11.0 + i,
         "l": 9.0 + i, "c": 10.5 + i, "v": 100.0}
        for i in range(window_bars)
    ]
    context = dict(CONTEXT, ohlcv_window=window, indicators={"rsi": 54.23})
    analysis = {
        "supporting_evidence": [
            {"statement": "momentum is positive", "context_field": "indicators.rsi",
             "value": "54.23"},
        ],
    }
    if cited_index is not None:
        analysis["supporting_evidence"].append({
            "statement": "the last bar closed higher",
            "context_field": f"ohlcv_window.{window[cited_index]['t']}.c",
            "value": str(window[cited_index]["c"]),
        })
    return {"example_id": "x", "split": "train", "context": context, "analysis": analysis,
            "lineage": {"teacher_model": "mimo"}}


def test_a_view_trims_only_the_candles():
    from dataset.render import VIEWS, render_context

    context = _corpus_row()["context"]
    tail = render_context(context, VIEWS["tail"])
    assert len(tail["ohlcv_window"]) == 20
    assert tail["ohlcv_window"] == context["ohlcv_window"][-20:]
    # Everything else is untouched, so views differ only in the candles.
    assert {k: v for k, v in tail.items() if k != "ohlcv_window"} == \
           {k: v for k, v in context.items() if k != "ohlcv_window"}
    assert "ohlcv_window" not in render_context(context, VIEWS["compact"])
    assert render_context(context, VIEWS["full"]) is context


def test_a_view_that_removes_cited_evidence_drops_the_row():
    """One row in five cites a candle. Showing the student a prompt without
    the evidence its target cites is a direct lesson in fabrication."""
    from dataset.render import VIEWS, render_pair
    from teacher.prompt import load_prompt

    template = load_prompt("v1")
    row = _corpus_row(cited_index=95)          # cited candle is near the end
    assert render_pair(row, VIEWS["full"], template) is not None
    assert render_pair(row, VIEWS["tail"], template) is not None
    assert render_pair(row, VIEWS["compact"], template) is None


def test_a_deep_candle_citation_survives_only_the_full_view():
    from dataset.render import VIEWS, render_pair
    from teacher.prompt import load_prompt

    template = load_prompt("v1")
    row = _corpus_row(cited_index=3)           # 96 bars back
    assert render_pair(row, VIEWS["full"], template) is not None
    assert render_pair(row, VIEWS["tail"], template) is None


def test_the_target_is_the_analysis_as_json():
    from dataset.render import VIEWS, render_pair
    from teacher.prompt import load_prompt

    pair = render_pair(_corpus_row(), VIEWS["tail"], load_prompt("v1"))
    assert [m["role"] for m in pair["messages"]] == ["system", "user", "assistant"]
    assert json.loads(pair["messages"][-1]["content"]) == _corpus_row()["analysis"]
