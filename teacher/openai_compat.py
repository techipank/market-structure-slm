"""Generic OpenAI-compatible chat-completions client.

Almost every vendor and every local serving stack (vLLM, SGLang, Ollama,
LM Studio, llama.cpp's server) exposes `POST /v1/chat/completions` with
OpenAI's request and response shape. This module talks to any of them, so
adding a vendor is normally a config entry rather than code.

What it deliberately does *not* assume
--------------------------------------
* **That cost is reported.** OpenRouter returns `usage.cost`; most endpoints
  do not. `Usage.cost` stays `None` rather than becoming `0.0`, because "this
  was free" and "nobody told us the price" are different facts and the
  benchmark report distinguishes them.
* **That JSON Schema mode works.** Support varies wildly, and several servers
  accept `response_format` and then ignore it. `supports_json_schema=False`
  makes the runner skip straight to a weaker mode instead of burning an
  attempt discovering it.
* **That the auth header is `Authorization: Bearer`.** Some endpoints use a
  bespoke header name, so both the header and the value prefix are
  configurable, and auth is omitted entirely when no key is set (the normal
  case for a local server).
"""

from __future__ import annotations

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

#: Substrings that mean "this endpoint cannot do what you asked for" rather
#: than "something went wrong". Matching is a heuristic: a wrong guess costs
#: one wasted fallback attempt, never a silent wrong answer.
UNSUPPORTED_HINTS: tuple[str, ...] = (
    "response_format",
    "structured output",
    "json_schema",
    "does not support",
    "not supported",
    "unsupported",
    "guided_json",
    "no available model provider meets your routing requirements",
)


@dataclass
class EndpointConfig:
    """Everything needed to talk to one chat-completions endpoint.

    The key is never stored in YAML - only the *name* of the environment
    variable holding it. `api_key` is populated at construction and suppressed
    from `repr` so it cannot reach a log line or a traceback.
    """

    name: str
    model: str
    base_url: str
    api_key: str = field(repr=False, default="")
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 4
    supports_json_schema: bool = True
    #: Merged into every request body. An escape hatch for vendor-specific
    #: knobs (reasoning effort, top_p, a served model alias) without needing a
    #: code change per vendor.
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError(f"endpoint {self.name!r}: no model configured")
        if not self.base_url:
            raise ValueError(f"endpoint {self.name!r}: no base_url configured")
        self.base_url = chat_completions_url(self.base_url)


def chat_completions_url(base_url: str) -> str:
    """Accept either a base URL or a full endpoint URL.

    Vendors document these inconsistently and the OpenAI SDK convention is to
    configure the *base* (`https://host/v1`), so both spellings turn up in a
    `.env` written by a human. Guessing wrong produces a 404 that reads like a
    bad key, which is a miserable thing to debug on your first live call.
    """
    trimmed = base_url.strip().rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    return f"{trimmed}/chat/completions"


class OpenAICompatibleProvider:
    """Implements `LLMProvider` against any OpenAI-shaped endpoint."""

    def __init__(self, config: EndpointConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def model(self) -> str:
        return self.config.model

    # ------------------------------------------------------------ request

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        if self.config.api_key:
            headers[self.config.auth_header] = (
                f"{self.config.auth_prefix}{self.config.api_key}"
            )
        return headers

    def body(
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
        if (
            mode is StructuredMode.JSON_SCHEMA
            and schema is not None
            and self.config.supports_json_schema
        ):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        elif mode is StructuredMode.JSON_OBJECT:
            body["response_format"] = {"type": "json_object"}
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
        if mode is StructuredMode.JSON_SCHEMA and not self.config.supports_json_schema:
            # Declared unsupported in config. Say so immediately rather than
            # spending a call to be told, and let the runner step down.
            raise UnsupportedFeature(
                f"{self.config.name} is configured as not supporting json_schema"
            )

        body = self.body(messages, schema, schema_name, mode)
        started = time.monotonic()
        payload = self.post_with_retries(body)
        elapsed = time.monotonic() - started
        return self.to_completion(payload, mode, elapsed)

    def to_completion(
        self, payload: dict[str, Any], mode: StructuredMode, elapsed: float
    ) -> Completion:
        choices = payload.get("choices") or []
        if not choices:
            raise TransientError(f"response contained no choices: {str(payload)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str):
            # Some servers return content as a list of parts.
            text = "".join(
                part.get("text", "") for part in text if isinstance(part, dict)
            )
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
            # Empty output in response to a structured-output request is a
            # capability failure, not a blip. Treating it as transient would
            # retry the same doomed request instead of stepping down to a
            # mode that works.
            if mode is not StructuredMode.NONE:
                raise UnsupportedFeature(
                    f"model returned an empty message under {mode.value}"
                )
            raise TransientError("model returned an empty message")

        return Completion(
            text=text,
            model=payload.get("model", self.config.model),
            structured_mode=mode,
            usage=self.parse_usage(payload.get("usage") or {}),
            latency_seconds=elapsed,
            finish_reason=choices[0].get("finish_reason"),
            served_by=payload.get("provider") or self.config.name,
            raw=payload,
        )

    @staticmethod
    def parse_usage(usage: dict[str, Any]) -> Usage:
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return Usage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost=_maybe_float(usage.get("cost")),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )

    # ------------------------------------------------------------ retries

    def post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        last: ProviderError | None = None
        for attempt in range(self.config.max_retries):
            try:
                return self.post(body)
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
                    # Full jitter: many workers hitting one limit together is
                    # exactly what dataset generation does.
                    delay = random.uniform(0, min(30.0, 2.0**attempt))
                time.sleep(delay)
        raise last or TransientError("request failed with no recorded error")

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(
                self.config.base_url, headers=self.headers(), json=body
            )
        except httpx.TimeoutException as exc:
            raise TransientError(f"request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"transport error: {exc}") from exc

        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise self.classify(200, payload, response.headers)
            return payload

        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text[:500]}}
        raise self.classify(response.status_code, payload, response.headers)

    def classify(self, status: int, payload: dict[str, Any], headers: Any) -> ProviderError:
        error = payload.get("error")
        if isinstance(error, str):
            error = {"message": error}
        error = error or {}
        message = str(error.get("message") or payload)
        combined = f"{message} {error.get('type', '')} {error.get('code', '')}".lower()

        if status in (401, 403):
            return AuthError(f"{status}: {message}")
        if status == 402:
            return AuthError(f"{status}: out of credits: {message}")
        if status == 429:
            return RateLimited(f"{status}: {message}", retry_after=retry_after(headers))
        if any(hint in combined for hint in UNSUPPORTED_HINTS):
            return UnsupportedFeature(f"{status}: {message}")
        if status >= 500:
            return TransientError(f"{status}: {message}")
        return ProviderError(f"{status}: {message}")


def retry_after(headers: Any) -> float | None:
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


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
