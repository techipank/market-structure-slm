"""OpenRouter client. The only module that knows about HTTP or API keys.

Deliberate choices
------------------
``provider: {"require_parameters": true}``
    Without it, a model that cannot honour `response_format` may be routed to
    silently and answer with unconstrained text - the request "succeeds" and
    the degradation is invisible until the parse fails. With it, routing fails
    cleanly (documented as a 503) and we can fall back on purpose, recording
    that we did. Turning an invisible failure into a loud one is worth a
    slightly higher failure rate.

Usage accounting
    OpenRouter returns `usage` including `cost` on every response; the older
    ``usage: {include: true}`` request flag is deprecated and does nothing, so
    it is not sent. `cost` is denominated in credits, not dollars, and that is
    recorded verbatim rather than converted with a guessed exchange rate.

Retries
    Only on genuinely retryable failures, with exponential backoff and full
    jitter, honouring `Retry-After` when present. A 401 is never retried: a bad
    key will still be bad in four seconds, and hammering it wastes the run.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from teacher.provider import (
    AuthError,
    Completion,
    ProviderError,
    RateLimited,
    StructuredMode,
    TransientError,
    Truncated,
    UnsupportedFeature,
    Usage,
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Substrings that identify "this model cannot do structured outputs" inside an
#: otherwise generic 400/503. OpenRouter does not document a specific code for
#: it, so this is a heuristic and is treated as one: a wrong guess costs one
#: wasted fallback attempt, never a silent wrong answer.
_UNSUPPORTED_HINTS = (
    "response_format",
    "structured output",
    "json_schema",
    "does not support",
    # Both wordings observed in the wild. The documented one is the first;
    # the second is what a `require_parameters` rejection actually returns,
    # as a 404 rather than the documented 503.
    "no available model provider meets your routing requirements",
    "no endpoints found that can handle the requested parameters",
)


@dataclass
class OpenRouterConfig:
    """Configuration. The key is read from the environment, never from YAML.

    `__repr__` is suppressed on the key field so it cannot reach a log line,
    a traceback, or a checkpoint file by accident. A test asserts this.
    """

    model: str
    api_key: str = field(repr=False, default="")
    #: Overridable so the endpoint can come from the environment like any
    #: other vendor. Accepts a base URL or a full chat-completions URL.
    base_url: str = API_URL
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    max_retries: int = 4
    #: Restrict routing to providers that support every parameter sent.
    require_parameters: bool = True
    #: Pin to specific provider slugs for reproducibility. Empty = any.
    only_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = True
    #: Merged into every request body, last, so it can override anything
    #: built above it. The escape hatch for OpenRouter-wide knobs that vary
    #: per model - notably `reasoning`, which is how a hybrid model is told
    #: to think less, or not at all.
    extra_body: dict[str, Any] = field(default_factory=dict)
    #: Optional attribution headers. Both optional; the title needs the
    #: referer to have any effect.
    referer: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("teacher model is not configured")
        from teacher.openai_compat import chat_completions_url

        self.base_url = chat_completions_url(self.base_url)

    @classmethod
    def from_env(cls, model: str | None = None, **overrides: Any) -> OpenRouterConfig:
        resolved = model or os.environ.get("TEACHER_MODEL", "")
        if not resolved:
            raise ValueError(
                "no teacher model configured: set TEACHER_MODEL or pass --model"
            )
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise AuthError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in; "
                ".env is gitignored."
            )
        return cls(model=resolved, api_key=key, **overrides)


class OpenRouterProvider:
    def __init__(self, config: OpenRouterConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def model(self) -> str:
        return self.config.model

    # ------------------------------------------------------------ request

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.referer:
            headers["HTTP-Referer"] = self.config.referer
            if self.config.title:
                headers["X-OpenRouter-Title"] = self.config.title
        return headers

    def _body(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None,
        schema_name: str,
        mode: StructuredMode,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if mode is StructuredMode.JSON_SCHEMA and schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        elif mode is StructuredMode.JSON_OBJECT:
            body["response_format"] = {"type": "json_object"}

        routing: dict[str, Any] = {}
        # require_parameters only means anything when we actually sent one of
        # the parameters whose support varies.
        if self.config.require_parameters and "response_format" in body:
            routing["require_parameters"] = True
        if self.config.only_providers:
            routing["only"] = list(self.config.only_providers)
            routing["allow_fallbacks"] = self.config.allow_fallbacks
        if routing:
            body["provider"] = routing
        body.update(self.config.extra_body)
        return body

    # ----------------------------------------------------------- complete

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        mode: StructuredMode = StructuredMode.JSON_SCHEMA,
    ) -> Completion:
        body = self._body(messages, schema, schema_name, mode)
        started = time.monotonic()
        degraded: tuple[str, ...] = ()
        try:
            payload = self._post_with_retries(body)
        except UnsupportedFeature:
            # `require_parameters` filters on a provider's *declared*
            # capabilities, not its real ones. Observed on stealth/ox-alpha:
            # routing refuses a strict json_schema request because the
            # provider does not advertise structured_outputs, yet the same
            # request succeeds and returns valid schema-conforming JSON once
            # the constraint is dropped.
            #
            # Retrying without it is safe here in a way it would not be in
            # general: `response_format` is still sent, and if the provider
            # silently ignores it the response fails pydantic validation and
            # the repair loop catches it. The gate was protecting against a
            # failure mode we already detect downstream, at the cost of
            # rejecting models that work.
            if not (self.config.require_parameters and "provider" in body):
                raise
            relaxed = dict(body)
            routing = {k: v for k, v in relaxed["provider"].items()
                       if k != "require_parameters"}
            if routing:
                relaxed["provider"] = routing
            else:
                relaxed.pop("provider")
            payload = self._post_with_retries(relaxed)
            degraded = ("routing: require_parameters relaxed",)
        elapsed = time.monotonic() - started

        choices = payload.get("choices") or []
        if not choices:
            raise TransientError(f"response contained no choices: {_trim(payload)}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if not text.strip():
            # Distinguish the three ways an empty body happens, because the
            # remedies are completely different.
            reasoning = message.get("reasoning") or ""
            if choices[0].get("finish_reason") == "length":
                raise Truncated(
                    f"output truncated at max_tokens={self.config.max_tokens} with no "
                    f"content produced"
                    + (
                        f"; the model emitted {len(reasoning)} characters of reasoning "
                        "first, so raise max_tokens for this reasoning model"
                        if reasoning
                        else ""
                    )
                )
            # An empty body in response to a structured-output request is a
            # capability failure, not a blip: observed on stealth/ox-alpha,
            # which returns "" for every `json_object` request while handling
            # `json_schema` correctly. Classifying it as transient would retry
            # the identical request four times and then give up, never trying
            # the mode that works.
            if mode is not StructuredMode.NONE:
                raise UnsupportedFeature(
                    f"model returned an empty message under {mode.value}"
                )
            raise TransientError("model returned an empty message")

        return Completion(
            text=text,
            model=payload.get("model", self.config.model),
            structured_mode=mode,
            usage=_parse_usage(payload.get("usage") or {}),
            latency_seconds=elapsed,
            finish_reason=choices[0].get("finish_reason"),
            served_by=payload.get("provider"),
            degraded=degraded,
            raw=payload,
        )

    # ------------------------------------------------------------ retries

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        last: ProviderError | None = None
        for attempt in range(self.config.max_retries):
            try:
                return self._post(body)
            except (RateLimited, TransientError) as exc:
                last = exc
                # A timeout on a slow reasoning model is not bad luck, it is
                # the request being too big for the deadline. Retrying the
                # identical request just spends the deadline again: four
                # attempts at a 120s timeout is eight wasted minutes before
                # the mode fallback even starts. Fail fast and let the
                # operator raise timeout_seconds.
                if "timed out" in str(exc).lower():
                    break
                if attempt == self.config.max_retries - 1:
                    break
                delay = getattr(exc, "retry_after", None)
                if delay is None:
                    # Exponential backoff with full jitter. Jitter matters when
                    # many workers hit the same limit at once, which is exactly
                    # what happens during dataset generation.
                    delay = random.uniform(0, min(30.0, 2.0**attempt))
                time.sleep(delay)
        raise last or TransientError("request failed with no recorded error")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(
                self.config.base_url, headers=self._headers(), json=body
            )
        except httpx.TimeoutException as exc:
            raise TransientError(f"request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"transport error: {exc}") from exc

        if response.status_code == 200:
            payload = response.json()
            # Errors can also arrive inside a 200 body.
            if isinstance(payload, dict) and payload.get("error"):
                raise _classify(200, payload, response.headers)
            return payload

        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}
        raise _classify(response.status_code, payload, response.headers)


# ---------------------------------------------------------------- helpers


def _classify(status: int, payload: dict[str, Any], headers: Any) -> ProviderError:
    error = payload.get("error") or {}
    # OpenRouter can return an error inside a 200 body; the real status lives
    # in error.code. Trusting the HTTP status there would classify a rate
    # limit as a generic failure and skip the retry it deserves.
    if isinstance(error.get("code"), int):
        status = int(error["code"])
    message = str(error.get("message") or payload)
    metadata = error.get("metadata") or {}
    error_type = str(metadata.get("error_type") or "")
    combined = f"{message} {error_type}".lower()

    if status in (401, 403) or error_type == "authentication":
        return AuthError(f"{status}: {message}")
    if status == 402 or error_type == "payment_required":
        return AuthError(f"{status}: out of credits: {message}")
    if status == 429 or error_type == "rate_limit_exceeded":
        return RateLimited(f"{status}: {message}", retry_after=_retry_after(headers))
    if any(hint in combined for hint in _UNSUPPORTED_HINTS):
        return UnsupportedFeature(f"{status}: {message}")
    if status >= 500 or error_type in {"provider_overloaded", "timeout"}:
        return TransientError(f"{status}: {message}")
    return ProviderError(f"{status}: {message}")


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_usage(usage: dict[str, Any]) -> Usage:
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cost=_maybe_float(usage.get("cost")),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
    )


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trim(payload: Any, limit: int = 300) -> str:
    return str(payload)[:limit]
