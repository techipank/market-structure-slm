"""The boundary between domain logic and model-vendor logic.

Everything above this line (prompts, schema, grounding, the repair loop) works
in terms of `Completion` and `Usage`. Everything below it (`openrouter.py`)
knows about HTTP, API keys, and one vendor's request shape.

The value is not hypothetical portability - it is that the repair loop, the
grounding checker and their tests can run with a `StubProvider` and no network,
no key, and no money. A test suite that cannot run offline gets skipped, and a
skipped test protects nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class StructuredMode(StrEnum):
    """How the shape of the response was enforced, in decreasing strength.

    Recorded on every result: a model that only managed JSON_OBJECT is not
    competing on equal terms with one that honoured JSON_SCHEMA, and comparing
    their schema-compliance rates without noting that would be misleading.
    """

    JSON_SCHEMA = "JSON_SCHEMA"  # decoder constrained to our schema
    JSON_OBJECT = "JSON_OBJECT"  # decoder constrained to valid JSON only
    NONE = "NONE"  # nothing enforced; the prompt asked nicely


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Reported by the provider, in its own units (OpenRouter reports credits).
    #: None when the provider did not return a cost.
    cost: float | None = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    structured_mode: StructuredMode
    usage: Usage
    latency_seconds: float
    finish_reason: str | None = None
    #: Provider slug that actually served the request, when reported. Two runs
    #: of the "same model" served by different providers, or at different
    #: quantizations, are not the same experiment.
    served_by: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ProviderError(RuntimeError):
    """Base class for anything that stopped a completion from happening."""

    retryable = False


class AuthError(ProviderError):
    """Missing or rejected credentials. Never retried - it will never work."""


class RateLimited(ProviderError):
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """Timeouts, 5xx, provider overload. Worth another attempt."""

    retryable = True


class UnsupportedFeature(ProviderError):
    """The model or its provider cannot honour a requested parameter.

    Distinct from `TransientError` because the correct response is to retry
    with a weaker `StructuredMode`, not to retry the same request.
    """


class LLMProvider(Protocol):
    """The only thing domain code is allowed to assume about a vendor."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        mode: StructuredMode = StructuredMode.JSON_SCHEMA,
    ) -> Completion: ...

    @property
    def model(self) -> str: ...
