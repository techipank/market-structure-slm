"""Model registry and the generic OpenAI-compatible provider. All offline."""

from __future__ import annotations

import json

import httpx
import pytest

from teacher.openai_compat import EndpointConfig, OpenAICompatibleProvider
from teacher.provider import (
    AuthError,
    RateLimited,
    StructuredMode,
    TransientError,
    UnsupportedFeature,
)
from teacher.registry import RegistryError, load_registry

REGISTRY = "configs/models.yaml"


def _cfg(**kw) -> EndpointConfig:
    base = dict(name="test", model="m", base_url="http://x/v1/chat/completions", api_key="k")
    base.update(kw)
    return EndpointConfig(**base)


def _provider(handler, **kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        _cfg(**kw), client=httpx.Client(transport=httpx.MockTransport(handler))
    )


# ------------------------------------------------------------- registry


def test_shipped_registry_loads_and_resolves():
    registry = load_registry(REGISTRY)
    assert {"openrouter", "xiaomi"} <= set(registry.endpoints)
    for alias in registry.aliases():
        spec = registry.spec(alias)
        assert spec.endpoint in registry.endpoints, f"{alias} points at a missing endpoint"
        assert spec.model


def test_shipped_registry_has_no_placeholder_slugs():
    """A REPLACE-ME slug would 404 partway into a paid run."""
    registry = load_registry(REGISTRY)
    unfilled = [a for a in registry.aliases() if "REPLACE-ME" in registry.spec(a).model]
    assert not unfilled, f"placeholder slugs still in the registry: {unfilled}"


def test_unknown_alias_lists_the_known_ones():
    registry = load_registry(REGISTRY)
    with pytest.raises(RegistryError, match="Known aliases"):
        registry.spec("does-not-exist")


def test_a_typo_in_options_is_an_error_not_a_silent_default():
    """`temprature: 0.9` quietly ignored would make results inexplicable."""
    from teacher.registry import _filter

    with pytest.raises(RegistryError, match="unknown option"):
        _filter({"temprature": 0.9}, EndpointConfig)


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    registry = load_registry(REGISTRY)
    with pytest.raises(AuthError, match="OPENROUTER_API_KEY"):
        registry.build("haiku")


def test_missing_base_url_names_the_variable(monkeypatch):
    monkeypatch.delenv("XIAOMI_BASE_URL", raising=False)
    monkeypatch.setenv("XIAOMI_API_KEY", "k")
    registry = load_registry(REGISTRY)
    with pytest.raises(RegistryError, match="XIAOMI_BASE_URL"):
        registry.build("mimo")


def test_endpoint_options_flow_into_the_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("K", "secret")
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "endpoints:\n"
        "  e:\n"
        "    kind: openai_compatible\n"
        "    base_url: http://h/v1/chat/completions\n"
        "    api_key_env: K\n"
        "    supports_json_schema: false\n"
        "    options: {temperature: 0.25, max_tokens: 999}\n"
        "models:\n"
        "  a: {endpoint: e, model: some-model, options: {max_tokens: 111}}\n",
        encoding="utf-8",
    )
    provider = load_registry(cfg).build("a")
    assert provider.model == "some-model"
    assert provider.config.temperature == 0.25
    # Per-model options win over endpoint defaults.
    assert provider.config.max_tokens == 111
    assert provider.config.supports_json_schema is False


# ---------------------------------------------- openai-compatible client


def test_api_key_is_not_in_the_repr():
    assert "SECRET" not in repr(_cfg(api_key="SECRET-VALUE"))


def test_auth_header_is_configurable():
    p = _provider(lambda r: httpx.Response(200), auth_header="X-Api-Key", auth_prefix="")
    assert p.headers()["X-Api-Key"] == "k"
    assert "Authorization" not in p.headers()


def test_auth_omitted_entirely_when_no_key():
    """The normal case for a local server."""
    p = _provider(lambda r: httpx.Response(200), api_key="")
    assert "Authorization" not in p.headers()


def test_declared_unsupported_schema_fails_fast_without_a_call():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={})

    p = _provider(handler, supports_json_schema=False)
    with pytest.raises(UnsupportedFeature):
        p.complete([{"role": "user", "content": "x"}], schema={"type": "object"})
    assert calls["n"] == 0, "should not spend a call to learn what config already says"


def test_json_schema_body_shape():
    p = _provider(lambda r: httpx.Response(200))
    body = p.body([], {"type": "object"}, "analysis", StructuredMode.JSON_SCHEMA)
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "analysis"


def test_extra_body_is_merged():
    """The escape hatch for vendor-specific knobs without a code change."""
    p = _provider(lambda r: httpx.Response(200), extra_body={"top_p": 0.8, "seed": 7})
    body = p.body([], None, "n", StructuredMode.NONE)
    assert body["top_p"] == 0.8 and body["seed"] == 7


def test_successful_completion_parses():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={
            "model": "served-model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })

    c = _provider(handler).complete([{"role": "user", "content": "x"}], mode=StructuredMode.NONE)
    assert c.text == "{}"
    assert c.model == "served-model"
    assert c.served_by == "test"
    assert c.usage.prompt_tokens == 12


def test_unreported_cost_is_none_not_zero():
    """Free and price-unknown are different facts, and the report says so."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    c = _provider(handler).complete([{"role": "user", "content": "x"}], mode=StructuredMode.NONE)
    assert c.usage.cost is None


def test_content_returned_as_parts_is_joined():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": [{"text": "{\"a\":"}, {"text": "1}"}]}}],
            "usage": {},
        })

    c = _provider(handler).complete([{"role": "user", "content": "x"}], mode=StructuredMode.NONE)
    assert c.text == '{"a":1}'


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": {"message": "bad key"}}, AuthError),
        (429, {"error": {"message": "slow down"}}, RateLimited),
        (500, {"error": {"message": "boom"}}, TransientError),
        (400, {"error": {"message": "response_format not supported"}}, UnsupportedFeature),
        (400, {"error": "guided_json is unsupported"}, UnsupportedFeature),
    ],
)
def test_error_classification(status, body, expected):
    p = _provider(lambda r: httpx.Response(200))
    assert isinstance(p.classify(status, body, httpx.Headers()), expected)


def test_auth_failure_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "nope"}})

    with pytest.raises(AuthError):
        _provider(handler, max_retries=4).complete(
            [{"role": "user", "content": "x"}], mode=StructuredMode.NONE
        )
    assert calls["n"] == 1


# ------------------------------------------------- failure modes seen live
#
# Every case below was found by the first real call against stealth/ox-alpha
# rather than imagined. They are regression tests for bugs that shipped.


def _openrouter(handler, **kw):
    from teacher.openrouter import OpenRouterConfig, OpenRouterProvider

    base = dict(model="m", api_key="k", max_retries=2)
    base.update(kw)
    return OpenRouterProvider(
        OpenRouterConfig(**base), client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_routing_rejection_is_recognised_and_relaxed():
    """`require_parameters` filters on DECLARED capability, not real capability.

    Observed live: OpenRouter refuses to route a strict json_schema request to
    stealth/ox-alpha because its provider does not advertise structured
    outputs, yet the identical request succeeds once the constraint is
    dropped. The 404 wording also differs from the documented 503 phrasing,
    which is why the first version did not catch it.
    """
    seen: list[dict] = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("provider", {}).get("require_parameters"):
            return httpx.Response(404, json={"error": {"message":
                "No endpoints found that can handle the requested parameters."}})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    completion = _openrouter(handler).complete(
        [{"role": "user", "content": "x"}], schema={"type": "object"}
    )
    assert completion.text == "{}"
    assert len(seen) == 2, "should retry once without require_parameters"
    assert "require_parameters" not in seen[1].get("provider", {})
    # The concession must be recorded, not silently made.
    assert completion.degraded == ("routing: require_parameters relaxed",)


def test_truncation_names_the_remedy():
    """A reasoning model can spend the whole budget thinking and answer nothing.

    Observed live: finish_reason "length", 0 characters of content, 12,032
    characters of reasoning. Retrying or stepping down the structured mode
    would both be useless - the fix is a bigger max_tokens, so the error says
    so.
    """
    from teacher.provider import Truncated

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": "", "reasoning": "thinking " * 500},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 8752, "completion_tokens": 4096},
        })

    with pytest.raises(Truncated, match="max_tokens"):
        _openrouter(handler).complete([{"role": "user", "content": "x"}],
                                      mode=StructuredMode.NONE)


def test_empty_structured_response_steps_down_instead_of_retrying():
    """json_object returned "" on ox-alpha while json_schema worked fine.

    Treating that as transient retried the same doomed request four times and
    never reached a mode that works.
    """
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {},
        })

    with pytest.raises(UnsupportedFeature):
        _openrouter(handler).complete([{"role": "user", "content": "x"}],
                                      mode=StructuredMode.JSON_OBJECT)


def test_empty_unstructured_response_is_transient():
    """With nothing requested, an empty body really is just a blip."""
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {},
        })

    with pytest.raises(TransientError):
        _openrouter(handler).complete([{"role": "user", "content": "x"}],
                                      mode=StructuredMode.NONE)


def test_timeouts_are_not_retried():
    """Retrying a request that was too slow just spends the deadline again.

    Four attempts at a 120s timeout is eight wasted minutes before the mode
    fallback even begins - which is exactly how one smoke call consumed
    fifteen minutes and produced nothing.
    """
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(TransientError, match="timed out"):
        _openrouter(handler, max_retries=4).complete(
            [{"role": "user", "content": "x"}], mode=StructuredMode.NONE
        )
    assert calls["n"] == 1


def test_reasoning_model_config_has_headroom():
    """Guards the settings that stopped alpha hanging."""
    registry = load_registry(REGISTRY)
    options = registry.spec("alpha").options
    assert options["max_tokens"] >= 16000, "reasoning needs budget beyond the answer"
    assert options["timeout_seconds"] >= 600, "25 tok/s over a long response needs time"
