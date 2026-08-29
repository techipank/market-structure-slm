"""Resolve a short model name into a configured provider.

Why a registry rather than passing raw slugs around
---------------------------------------------------
Once more than one vendor is in play, `--model` has to answer two questions at
once: *which endpoint* and *which model on it*. Encoding both in one string
(`vendor:model`) collides with real slugs — OpenRouter's free variants already
use a colon, as in `qwen/qwen-2.5-72b-instruct:free`.

So endpoints and models live in `configs/models.yaml` under short aliases, and
the CLI takes the alias. Three further benefits:

* **Secrets stay out of the file.** YAML holds the *name* of the environment
  variable, never the key.
* **The benchmark records what it ran.** An alias resolves to an exact
  endpoint, model id and options, all of which land in the result lineage.
* **Swapping a model is a config edit**, which is what "make teacher models
  replaceable" is supposed to mean in practice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from teacher.openai_compat import EndpointConfig, OpenAICompatibleProvider
from teacher.openrouter import OpenRouterConfig, OpenRouterProvider
from teacher.provider import AuthError, LLMProvider

DEFAULT_REGISTRY = Path("configs/models.yaml")


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    kind: str  # "openrouter" | "openai_compatible"
    base_url: str | None = None
    #: One name or a list of names, tried in order. A list is not
    #: over-engineering here: the same endpoint gets configured under
    #: different variable names by different people, and a forgiving lookup
    #: turns "why is my key not picked up" into a non-event.
    base_url_env: str | list[str] | None = None
    api_key_env: str | list[str] | None = None
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    supports_json_schema: bool = True
    #: True when the endpoint charges nothing per token (a local or
    #: self-hosted server). Distinguishes "free" from "price unreported".
    self_hosted: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def resolved_base_url(self) -> str:
        value = first_env(self.base_url_env)
        if value:
            return value
        if self.base_url_env and not self.base_url:
            raise RegistryError(
                f"endpoint {self.name!r} needs one of "
                f"{names(self.base_url_env)} set in the environment (or .env)"
            )
        if not self.base_url:
            raise RegistryError(f"endpoint {self.name!r} has neither base_url nor base_url_env")
        return self.base_url

    def resolved_key(self) -> str:
        if not self.api_key_env:
            return ""
        key = first_env(self.api_key_env)
        if not key and not self.self_hosted:
            raise AuthError(
                f"endpoint {self.name!r} needs one of {names(self.api_key_env)} set. "
                "Copy .env.example to .env and fill it in; .env is gitignored."
            )
        return key


def names(spec: str | list[str] | None) -> str:
    if not spec:
        return "<none>"
    return " / ".join([spec] if isinstance(spec, str) else spec)


def first_env(spec: str | list[str] | None) -> str:
    """First non-empty environment variable among the candidates."""
    if not spec:
        return ""
    for name in [spec] if isinstance(spec, str) else spec:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    endpoint: str
    model: str = ""
    #: Read the model id from the environment instead of pinning it here.
    #: Convenient, but it weakens reproducibility - the committed config no
    #: longer records which model ran - so the resolved id is written into
    #: every result's lineage to compensate.
    model_env: str | list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def resolved_model(self) -> str:
        value = first_env(self.model_env) or self.model
        if not value:
            raise RegistryError(
                f"model {self.alias!r} has no model id: set one of "
                f"{names(self.model_env)} or give it a literal `model:`"
            )
        return value


@dataclass(frozen=True)
class Registry:
    endpoints: dict[str, EndpointSpec]
    models: dict[str, ModelSpec]

    def aliases(self) -> list[str]:
        return sorted(self.models)

    def spec(self, alias: str) -> ModelSpec:
        if alias in self.models:
            return self.models[alias]
        raise RegistryError(
            f"unknown model alias {alias!r}. Known aliases: {', '.join(self.aliases()) or 'none'}. "
            "Add it to configs/models.yaml."
        )

    def build(self, alias: str, **overrides: Any) -> LLMProvider:
        spec = self.spec(alias)
        endpoint = self.endpoints.get(spec.endpoint)
        if endpoint is None:
            raise RegistryError(
                f"model {alias!r} points at unknown endpoint {spec.endpoint!r}"
            )

        # Endpoint defaults, then per-model options, then explicit overrides.
        options = {**endpoint.options, **spec.options, **overrides}

        if endpoint.kind == "openrouter":
            filtered = _filter(options, OpenRouterConfig)
            base = first_env(endpoint.base_url_env) or endpoint.base_url
            if base:
                filtered.setdefault("base_url", base)
            config = OpenRouterConfig(
                model=spec.resolved_model(),
                api_key=endpoint.resolved_key(),
                **filtered,
            )
            return OpenRouterProvider(config)

        if endpoint.kind == "openai_compatible":
            config = EndpointConfig(
                name=endpoint.name,
                model=spec.resolved_model(),
                base_url=endpoint.resolved_base_url(),
                api_key=endpoint.resolved_key(),
                auth_header=endpoint.auth_header,
                auth_prefix=endpoint.auth_prefix,
                supports_json_schema=endpoint.supports_json_schema,
                extra_headers=dict(endpoint.extra_headers),
                extra_body={**endpoint.extra_body, **options.pop("extra_body", {})},
                **_filter(options, EndpointConfig),
            )
            return OpenAICompatibleProvider(config)

        raise RegistryError(
            f"endpoint {endpoint.name!r} has unknown kind {endpoint.kind!r}; "
            "expected 'openrouter' or 'openai_compatible'"
        )


def _filter(options: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep only keys the dataclass actually accepts.

    A typo in the YAML becomes a loud error rather than a silently ignored
    setting, which is the failure mode that matters here: a benchmark run with
    `temprature: 0.9` quietly at default would produce numbers nobody could
    explain later.
    """
    allowed = set(getattr(cls, "__dataclass_fields__", {}))
    unknown = set(options) - allowed - {"extra_body"}
    if unknown:
        raise RegistryError(
            f"unknown option(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid options: {sorted(allowed - {'model', 'api_key', 'name'})}"
        )
    reserved = {"model", "api_key", "name"}
    return {k: v for k, v in options.items() if k in allowed and k not in reserved}


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> Registry:
    p = Path(path)
    if not p.exists():
        raise RegistryError(f"model registry not found at {p}")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    endpoints: dict[str, EndpointSpec] = {}
    for name, raw in (doc.get("endpoints") or {}).items():
        endpoints[name] = EndpointSpec(name=name, **raw)

    models: dict[str, ModelSpec] = {}
    for alias, raw in (doc.get("models") or {}).items():
        models[alias] = ModelSpec(alias=alias, **raw)

    return Registry(endpoints=endpoints, models=models)
