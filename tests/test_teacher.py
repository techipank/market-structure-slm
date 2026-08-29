"""Teacher tests. Every one runs offline: no network, no API key, no cost.

That is the point of the provider seam. A suite that needs credentials gets
skipped in CI, and a skipped test protects nothing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from teacher.grounding import Finding, resolve_path, verify
from teacher.jsonschema import (
    UNSUPPORTED_KEYWORDS,
    SchemaError,
    inline_refs,
    to_strict_schema,
)
from teacher.openrouter import OpenRouterConfig, OpenRouterProvider, _classify, _parse_usage
from teacher.prompt import PromptError, build_messages, load_prompt
from teacher.provider import (
    AuthError,
    Completion,
    RateLimited,
    StructuredMode,
    TransientError,
    UnsupportedFeature,
    Usage,
)
from teacher.runner import TeacherRunner, parse_analysis
from teacher.schema import TeacherAnalysis

# --------------------------------------------------------------- fixtures

CONTEXT: dict[str, Any] = {
    "schema_version": "1.0.0",
    "engine_version": "1.0.0",
    "params_fingerprint": "abc123",
    "symbol": "SPY",
    "interval": "1d",
    "as_of": "2026-08-21T00:00:00+00:00",
    "bar_index": 2925,
    "bars_available": 2926,
    "ohlcv_window": [{"t": "2026-08-21T00:00:00+00:00", "o": 765.0, "h": 770.0,
                      "l": 764.0, "c": 768.5, "v": 1000.0}],
    "indicators": {"ema": {"20": 763.7263}, "rsi": 54.23, "atr": 6.9373},
    "structure": {
        "trend": "EXPANSION",
        "bias": "BULLISH",
        "swings": [],
        "last_event": {"type": "BOS_BULLISH", "timestamp": "2026-08-13T00:00:00+00:00",
                       "level": 776.85, "close": 777.88, "bars_ago": 6},
        "active_swing_low": 729.1,
    },
    "levels": [{"price": 739.51, "side": "SUPPORT", "touches": 3,
                "distance_atr": -4.2, "last_index": 2893}],
    "volatility": {"atr": 6.9373, "regime": "LOW", "atr_percentile": 0.2222},
    "regime": {"trend_regime": "MIXED", "efficiency_ratio": 0.2841},
    "setups": [],
    "near_miss_setups": [],
}


def valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "higher_timeframe_bias": "BULLISH",
        "higher_timeframe_rationale": "The weekly structure remains constructive.",
        "market_structure": "EXPANSION",
        "market_structure_explanation": "A higher high sits above a lower low.",
        "market_state": "TRENDING_UP",
        "setup_type": "NONE",
        "supporting_evidence": [
            {"statement": "Momentum sits above the midpoint.",
             "context_field": "indicators.rsi", "value": "54.23"},
            {"statement": "The last structure break was upward.",
             "context_field": "structure.last_event.type", "value": "BOS_BULLISH"},
        ],
        "conflicting_evidence": [
            {"statement": "The trend regime is not decisively directional.",
             "context_field": "regime.trend_regime", "value": "MIXED"},
        ],
        "important_levels": [
            {"price": 739.51, "role": "SUPPORT",
             "context_field": "levels[0].price", "note": "A well-tested floor below price."},
        ],
        "scenarios": [
            {"name": "Continuation", "direction": "BULLISH",
             "condition": "Price holds above the recent break level.",
             "trigger_price": 776.85, "invalidation_price": 729.1,
             "expectation": "Structure would remain constructive."},
            {"name": "Failure", "direction": "BEARISH",
             "condition": "Price loses the active swing low.",
             "trigger_price": 729.1, "invalidation_price": None,
             "expectation": "Control would shift to sellers."},
        ],
        "confidence": 0.55,
        "reasoning_summary": "Structure is expanding while momentum stays neutral.",
    }
    payload.update(overrides)
    return payload


class StubProvider:
    """Returns canned responses in order, recording what it was asked."""

    def __init__(self, responses: list[str | Exception], model: str = "stub/model") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages, *, schema=None, schema_name="response",
                 mode=StructuredMode.JSON_SCHEMA) -> Completion:
        self.calls.append({"messages": messages, "schema": schema, "mode": mode})
        if not self._responses:
            raise AssertionError("StubProvider ran out of canned responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return Completion(
            text=nxt, model=self._model, structured_mode=mode,
            usage=Usage(prompt_tokens=100, completion_tokens=50, cost=0.001),
            latency_seconds=0.5,
        )


# ---------------------------------------------------------- strict schema


def test_strict_schema_has_no_refs_or_defs():
    schema = to_strict_schema(TeacherAnalysis, frozenset({"schema_version"}))
    text = json.dumps(schema)
    assert "$ref" not in text
    assert "$defs" not in text


def test_strict_schema_closes_objects_and_requires_everything():
    schema = to_strict_schema(TeacherAnalysis, frozenset({"schema_version"}))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


def test_strict_schema_strips_keywords_a_decoder_cannot_enforce():
    text = json.dumps(to_strict_schema(TeacherAnalysis, frozenset({"schema_version"})))
    for keyword in UNSUPPORTED_KEYWORDS:
        assert f'"{keyword}"' not in text


def test_excluded_field_is_not_requested_from_the_model():
    schema = to_strict_schema(TeacherAnalysis, frozenset({"schema_version"}))
    assert "schema_version" not in schema["properties"]
    assert "schema_version" not in schema["required"]


@pytest.mark.parametrize(
    "override",
    [
        {"confidence": 1.7},
        {"confidence": -0.1},
        {"supporting_evidence": []},
        {"conflicting_evidence": []},
        {"scenarios": []},
        {"important_levels": []},
    ],
)
def test_pydantic_still_enforces_what_the_schema_dropped(override):
    """The dropped keywords must survive as validation, not vanish.

    This is the seam between the three layers: the decoder cannot express
    these constraints, so if pydantic did not enforce them nothing would.
    """
    with pytest.raises(ValidationError):
        TeacherAnalysis.model_validate(valid_payload(**override))


def test_inline_refs_rejects_unresolvable_and_deep_schemas():
    with pytest.raises(SchemaError):
        inline_refs({"properties": {"a": {"$ref": "#/$defs/Missing"}}})


# ----------------------------------------------------------------- prompt


def test_prompt_loads_and_hash_is_stable():
    a, b = load_prompt("v1"), load_prompt("v1")
    assert a.content_hash == b.content_hash
    assert len(a.content_hash) == 16
    assert "copied from the context" in a.system


def test_unknown_prompt_version_names_what_is_available():
    with pytest.raises(PromptError, match="available"):
        load_prompt("v99")


def test_context_goes_in_the_user_message_not_the_system_message():
    """The system prompt must stay identical across examples so it can cache."""
    template = load_prompt("v1")
    a = build_messages(template, CONTEXT)
    b = build_messages(template, {**CONTEXT, "symbol": "QQQ", "as_of": "2020-01-01"})
    assert a[0]["content"] == b[0]["content"]
    assert a[1]["content"] != b[1]["content"]
    assert "SPY" in a[1]["content"]


def test_context_json_is_compact():
    """Whitespace in the payload is paid for on every single request."""
    rendered = load_prompt("v1").render_user(CONTEXT)
    assert '"symbol":"SPY"' in rendered  # no space after the colon


def test_schema_outline_is_sent_in_every_mode():
    """Strict decoding included: a provider can accept `response_format` and
    ignore it, leaving the prompt as the only statement of the field names."""
    runner = TeacherRunner(StubProvider([]))
    for mode in StructuredMode:
        system = runner.render_messages(CONTEXT, mode)[0]["content"]
        assert "Required output shape" in system, mode
        assert "market_structure" in system, mode


# ----------------------------------------------------------------- parsing


def test_parse_strips_markdown_fences():
    text = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert parse_analysis(text).market_state.value == "TRENDING_UP"


def test_parse_ignores_braces_in_leading_chain_of_thought():
    """Reasoning emitted into `content` with no delimiter, containing braces
    and a half-written draft of the answer. Observed on nemotron once the
    provider ignored `response_format`. Spanning first-brace to last-brace
    splices the thinking into the payload; the answer is the last balanced
    object that validates."""
    payload = valid_payload()
    text = (
        'We need to output {"market_state": ...} with fields {a, b, c}. '
        'Draft: {"market_state": "TRENDING_UP"} - no, incomplete. '
        + json.dumps(payload)
    )
    assert parse_analysis(text).confidence == payload["confidence"]


def test_parse_recovers_json_surrounded_by_chatter():
    text = "Sure! Here is the analysis:\n" + json.dumps(valid_payload()) + "\nHope that helps."
    assert parse_analysis(text).confidence == 0.55


def test_model_supplied_schema_version_is_ignored():
    payload = valid_payload()
    payload["schema_version"] = "9.9.9-fake"
    assert parse_analysis(json.dumps(payload)).schema_version != "9.9.9-fake"


# ------------------------------------------------------------ repair loop


def test_valid_first_response_needs_one_call():
    provider = StubProvider([json.dumps(valid_payload())])
    result = TeacherRunner(provider).analyse(CONTEXT)
    assert result.ok
    assert len(provider.calls) == 1
    assert len(result.attempts) == 1


def test_invalid_response_triggers_a_repair_that_includes_the_errors():
    bad = json.dumps(valid_payload(confidence=5.0))
    provider = StubProvider([bad, json.dumps(valid_payload())])
    result = TeacherRunner(provider).analyse(CONTEXT)

    assert result.ok
    assert len(provider.calls) == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].ok is False

    repair_turn = provider.calls[1]["messages"][-1]
    assert repair_turn["role"] == "user"
    assert "confidence" in repair_turn["content"]
    # The model must see its own failed answer to correct it.
    assert provider.calls[1]["messages"][-2]["role"] == "assistant"


def test_repairs_are_bounded_and_the_failure_is_reported():
    bad = json.dumps(valid_payload(confidence=5.0))
    provider = StubProvider([bad, bad, bad])
    result = TeacherRunner(provider, max_repairs=2).analyse(CONTEXT)
    assert not result.ok
    assert "validation failed" in result.error
    assert len(provider.calls) == 3


def test_malformed_json_is_repairable():
    provider = StubProvider(["{not json at all", json.dumps(valid_payload())])
    result = TeacherRunner(provider).analyse(CONTEXT)
    assert result.ok
    assert "not valid JSON" in (result.attempts[0].error or "")


# -------------------------------------------------------- mode fallback


def test_unsupported_structured_output_steps_down_a_mode():
    provider = StubProvider(
        [UnsupportedFeature("model does not support response_format"),
         json.dumps(valid_payload())]
    )
    result = TeacherRunner(provider).analyse(CONTEXT)
    assert result.ok
    assert result.structured_mode == StructuredMode.JSON_OBJECT.value
    assert provider.calls[0]["mode"] is StructuredMode.JSON_SCHEMA
    assert provider.calls[1]["mode"] is StructuredMode.JSON_OBJECT


def test_fallback_can_be_disabled_for_a_fair_comparison():
    provider = StubProvider([UnsupportedFeature("nope")])
    result = TeacherRunner(provider, allow_mode_fallback=False).analyse(CONTEXT)
    assert not result.ok
    assert len(provider.calls) == 1


# ------------------------------------------------------------- grounding


def test_resolve_path_handles_indices_and_missing_keys():
    assert resolve_path(CONTEXT, "indicators.rsi") == (True, 54.23)
    assert resolve_path(CONTEXT, "levels[0].price") == (True, 739.51)
    assert resolve_path(CONTEXT, "structure.last_event.level") == (True, 776.85)
    assert resolve_path(CONTEXT, "levels[9].price")[0] is False
    assert resolve_path(CONTEXT, "indicators.nope")[0] is False


def test_a_correct_analysis_is_fully_grounded():
    report = verify(TeacherAnalysis.model_validate(valid_payload()), CONTEXT)
    assert report.is_grounded, report.issues
    assert report.grounded_fraction == 1.0


def test_an_invented_price_is_caught():
    """The headline check: a plausible price that appears nowhere."""
    payload = valid_payload()
    payload["important_levels"][0]["price"] = 742.17  # never appears in CONTEXT
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Finding.UNGROUNDED_PRICE.value in report.counts()


def test_a_misquoted_indicator_is_caught():
    payload = valid_payload()
    payload["supporting_evidence"][0]["value"] = "68.9"  # rsi is really 54.23
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Finding.VALUE_MISMATCH.value in report.counts()


def test_a_fabricated_field_path_is_caught():
    payload = valid_payload()
    payload["supporting_evidence"][0]["context_field"] = "indicators.macd_histogram"
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Finding.UNRESOLVABLE_FIELD.value in report.counts()


def test_a_real_price_cited_at_the_wrong_path_is_a_milder_finding():
    payload = valid_payload()
    payload["important_levels"][0]["context_field"] = "levels[0].distance_atr"
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    counts = report.counts()
    assert Finding.UNRESOLVABLE_FIELD.value in counts
    assert Finding.UNGROUNDED_PRICE.value not in counts


def test_citation_missing_its_prefix_resolves_but_is_recorded():
    """Measured as the commonest citation defect on real runs. Accepting a
    unique suffix keeps the guarantee - it is still an exact lookup, and a
    wrong value still fails - but the concession is visible in the report."""
    payload = valid_payload()
    payload["supporting_evidence"][0] = {
        "statement": "The trend is expansionary",
        "context_field": "ema.20",          # short of its `indicators.` root
        "value": "763.7263",
    }
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert report.is_grounded
    assert Finding.FIELD_PATH_IMPRECISE.value in report.counts()
    assert Finding.UNRESOLVABLE_FIELD.value not in report.counts()


def test_an_ambiguous_suffix_is_not_a_citation():
    """Two matches is not an address. `last_event.level` exists under both
    `structure` and `higher_timeframe`, so it stays unresolvable."""
    context = {
        **CONTEXT,
        "higher_timeframe": {"last_event": {"level": 776.85}},
    }
    payload = valid_payload()
    payload["supporting_evidence"][0] = {
        "statement": "A break happened here",
        "context_field": "last_event.level",
        "value": "776.85",
    }
    report = verify(TeacherAnalysis.model_validate(payload), context)
    assert Finding.UNRESOLVABLE_FIELD.value in report.counts()
    assert not report.is_grounded


def test_a_wrong_value_still_fails_when_the_suffix_resolves():
    """The tolerance is about addressing, not about the number."""
    payload = valid_payload()
    payload["supporting_evidence"][0] = {
        "statement": "The trend is expansionary",
        "context_field": "ema.20",
        "value": "999.99",
    }
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Finding.VALUE_MISMATCH.value in report.counts()
    assert not report.is_grounded


def test_list_elements_can_be_cited_by_identity():
    """Positional citation into a list is the one thing models measurably
    cannot do; naming the element sidesteps counting entirely."""
    context = {
        **CONTEXT,
        "levels": [
            {"id": "L0", "price": 739.51, "side": "SUPPORT"},
            {"id": "L1", "price": 781.02, "side": "RESISTANCE"},
        ],
    }
    found, value = resolve_path(context, "levels.L1.price")
    assert (found, value) == (True, 781.02)
    # A candle is named by the timestamp it already carries.
    found, value = resolve_path(context, "ohlcv_window.2026-08-21T00:00:00+00:00.c")
    assert (found, value) == (True, 768.5)
    assert resolve_path(context, "levels.L7.price") == (False, None)


def test_positional_citation_still_works():
    """The identity lookup is additive: nothing that resolved before stops."""
    assert resolve_path(CONTEXT, "levels[0].price") == (True, 739.51)


def test_numbers_in_prose_are_rejected():
    payload = valid_payload()
    payload["reasoning_summary"] = "Price is holding above 763.73 with room to run."
    report = verify(TeacherAnalysis.model_validate(payload), CONTEXT)
    assert Finding.NUMBER_IN_PROSE.value in report.counts()


def test_prose_may_name_a_period_the_engine_actually_computed():
    """The exemption is derived from the context's own keys, so it tracks the
    engine's parameters instead of a hardcoded list. CONTEXT has a 20 EMA and
    no 50, and naming a series the engine never computed is still a claim
    about something the reader cannot check."""
    payload = valid_payload()
    payload["reasoning_summary"] = "Price is holding above the 20 EMA."
    assert Finding.NUMBER_IN_PROSE.value not in verify(
        TeacherAnalysis.model_validate(payload), CONTEXT
    ).counts()

    payload["reasoning_summary"] = "Price is holding above the 50 EMA."
    assert Finding.NUMBER_IN_PROSE.value in verify(
        TeacherAnalysis.model_validate(payload), CONTEXT
    ).counts()


def test_prose_exemption_requires_the_unit_word():
    """Otherwise the exemption would launder any number that happens to equal
    a period - "the RSI has rolled below 20" is a threshold claim, not a name."""
    payload = valid_payload()
    payload["reasoning_summary"] = "The RSI has rolled below 20."
    assert Finding.NUMBER_IN_PROSE.value in verify(
        TeacherAnalysis.model_validate(payload), CONTEXT
    ).counts()


def test_prose_allows_the_written_out_period_forms_models_actually_use():
    payload = valid_payload()
    for phrasing in ("the 20-day exponential moving average",
                     "the 20 period moving average",
                     "the EMA 20"):
        payload["reasoning_summary"] = f"Price is holding above {phrasing}."
        assert Finding.NUMBER_IN_PROSE.value not in verify(
            TeacherAnalysis.model_validate(payload), CONTEXT
        ).counts(), phrasing


def test_grounding_is_reported_even_when_it_fails():
    """A poorly grounded analysis is still returned - flagged, not discarded.

    Rejection is the dataset pipeline's job. Losing the evidence here would
    make it impossible to measure how often the teacher does this.
    """
    payload = valid_payload()
    payload["important_levels"][0]["price"] = 999.99
    provider = StubProvider([json.dumps(payload)])
    result = TeacherRunner(provider).analyse(CONTEXT)
    assert result.ok
    assert not result.grounding.is_grounded


# --------------------------------------------------------------- lineage


def test_result_record_carries_full_lineage():
    provider = StubProvider([json.dumps(valid_payload())])
    record = TeacherRunner(provider).analyse(CONTEXT).to_record()

    lineage = record["lineage"]
    for key in ("teacher_model", "prompt_version", "prompt_hash", "engine_version",
                "context_schema_version", "params_fingerprint", "analysis_schema_version",
                "structured_mode", "generated_at"):
        assert lineage[key], f"{key} is empty"
    assert lineage["engine_version"] == "1.0.0"
    assert lineage["params_fingerprint"] == "abc123"
    assert record["example"] == {"symbol": "SPY", "interval": "1d",
                                 "as_of": "2026-08-21T00:00:00+00:00", "bar_index": 2925}
    assert record["usage"]["prompt_tokens"] == 100
    assert record["usage"]["cost"] == 0.001
    assert "Not trading advice" in record["disclaimer"]
    json.dumps(record)  # must be serialisable


def test_usage_accumulates_across_repairs():
    bad = json.dumps(valid_payload(confidence=5.0))
    provider = StubProvider([bad, json.dumps(valid_payload())])
    result = TeacherRunner(provider).analyse(CONTEXT)
    assert result.prompt_tokens == 200
    assert result.cost == pytest.approx(0.002)


# ------------------------------------------------------- provider client


def test_api_key_never_appears_in_a_repr():
    config = OpenRouterConfig(model="x/y", api_key="sk-or-v1-SECRETVALUE")
    assert "SECRETVALUE" not in repr(config)
    assert "SECRETVALUE" not in str(config)


def test_missing_key_raises_a_helpful_auth_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("TEACHER_MODEL", "x/y")
    with pytest.raises(AuthError, match="OPENROUTER_API_KEY"):
        OpenRouterConfig.from_env()


def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv("TEACHER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(ValueError, match="TEACHER_MODEL"):
        OpenRouterConfig.from_env()


def test_extra_body_is_merged_and_wins():
    """How a hybrid reasoning model is told to stop thinking. Merged last so
    a per-model override beats anything the request builder chose."""
    config = OpenRouterConfig(
        model="m", api_key="k",
        extra_body={"reasoning": {"enabled": False}, "max_tokens": 2048},
    )
    body = OpenRouterProvider(config)._body([], None, "n", StructuredMode.NONE)
    assert body["reasoning"] == {"enabled": False}
    assert body["max_tokens"] == 2048


def test_request_body_shape():
    config = OpenRouterConfig(model="vendor/model", api_key="k", temperature=0.0)
    provider = OpenRouterProvider(config)
    body = provider._body([{"role": "user", "content": "hi"}], {"type": "object"},
                          "analysis", StructuredMode.JSON_SCHEMA)
    assert body["model"] == "vendor/model"
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "analysis"
    # require_parameters turns silent degradation into a clean routing failure.
    assert body["provider"]["require_parameters"] is True


def test_require_parameters_omitted_when_no_response_format_is_sent():
    provider = OpenRouterProvider(OpenRouterConfig(model="m", api_key="k"))
    body = provider._body([], None, "n", StructuredMode.NONE)
    assert "response_format" not in body
    assert "provider" not in body


def test_provider_pinning_disables_fallbacks():
    config = OpenRouterConfig(model="m", api_key="k", only_providers=("fireworks",),
                              allow_fallbacks=False)
    body = OpenRouterProvider(config)._body([], {"type": "object"}, "n",
                                            StructuredMode.JSON_SCHEMA)
    assert body["provider"]["only"] == ["fireworks"]
    assert body["provider"]["allow_fallbacks"] is False


def test_attribution_title_requires_a_referer():
    with_both = OpenRouterProvider(
        OpenRouterConfig(model="m", api_key="k", referer="https://x", title="T")
    )._headers()
    assert with_both["HTTP-Referer"] == "https://x"
    assert with_both["X-OpenRouter-Title"] == "T"

    title_only = OpenRouterProvider(
        OpenRouterConfig(model="m", api_key="k", title="T")
    )._headers()
    assert "X-OpenRouter-Title" not in title_only


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {"message": "no auth"}}, AuthError),
        (402, {"error": {"message": "out of credits"}}, AuthError),
        (429, {"error": {"message": "slow down"}}, RateLimited),
        (503, {"error": {"message": "No available model provider meets your "
                                    "routing requirements"}}, UnsupportedFeature),
        (400, {"error": {"message": "response_format is not supported"}}, UnsupportedFeature),
        (502, {"error": {"message": "upstream is down"}}, TransientError),
    ],
)
def test_error_classification(status, body, expected):
    assert isinstance(_classify(status, body, httpx.Headers()), expected)


def test_rate_limit_honours_retry_after():
    error = _classify(429, {"error": {"message": "x"}}, httpx.Headers({"Retry-After": "7"}))
    assert isinstance(error, RateLimited)
    assert error.retry_after == 7.0


def test_usage_parsing_reads_cost_and_cached_tokens():
    usage = _parse_usage({
        "prompt_tokens": 194, "completion_tokens": 2, "cost": 0.95,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 12},
    })
    assert usage.prompt_tokens == 194
    assert usage.cost == 0.95
    assert usage.cached_tokens == 100
    assert usage.reasoning_tokens == 12
    assert usage.total_tokens == 196


def test_missing_cost_is_none_not_zero():
    """Zero cost and unknown cost must never be confused in an accounting run."""
    assert _parse_usage({"prompt_tokens": 1}).cost is None


def test_error_inside_a_200_body_is_still_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 429, "message": "limited"}})

    provider = OpenRouterProvider(
        OpenRouterConfig(model="m", api_key="k", max_retries=1),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RateLimited):
        provider.complete([{"role": "user", "content": "x"}])


def test_successful_call_parses_into_a_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={
            "model": "vendor/model-2026", "provider": "Fireworks",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.0004},
        })

    provider = OpenRouterProvider(
        OpenRouterConfig(model="vendor/model", api_key="k"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    completion = provider.complete([{"role": "user", "content": "x"}])
    assert completion.text == "{}"
    assert completion.model == "vendor/model-2026"
    assert completion.served_by == "Fireworks"
    assert completion.usage.cost == 0.0004
    assert completion.latency_seconds >= 0


def test_auth_failure_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = OpenRouterProvider(
        OpenRouterConfig(model="m", api_key="bad", max_retries=4),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(AuthError):
        provider.complete([{"role": "user", "content": "x"}])
    assert calls["n"] == 1, "a bad key will still be bad on the next attempt"


def test_transient_failure_is_retried_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, json={"error": {"message": "upstream down"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    provider = OpenRouterProvider(
        OpenRouterConfig(model="m", api_key="k", max_retries=3),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.complete([{"role": "user", "content": "x"}]).text == "{}"
    assert calls["n"] == 2
