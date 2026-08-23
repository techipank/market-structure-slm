"""Orchestration: prompt -> call -> parse -> validate -> repair -> verify.

The repair loop
---------------
Constrained decoding guarantees shape but not the constraints it cannot
express (at least two evidence items, confidence in range). When pydantic
rejects a response, the errors are handed back to the model as another turn.
This converts most hard failures into successes at the cost of one extra call,
and the attempt count is recorded because "valid after two repairs" and "valid
first time" are different models wearing the same score.

Structured-mode fallback
------------------------
If the provider rejects `response_format` outright, the runner steps down
JSON_SCHEMA -> JSON_OBJECT -> NONE, appending the schema outline to the prompt
once the decoder stops enforcing it. Which mode succeeded is recorded: a model
that needed the fallback is not comparable to one that did not.

Nothing here trusts the teacher. The returned result carries the analysis
*and* its grounding report, and callers are expected to look at both.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from teacher import ANALYSIS_DISCLAIMER, TEACHER_SCHEMA_VERSION
from teacher.grounding import GroundingReport, verify
from teacher.jsonschema import describe_schema, to_strict_schema
from teacher.prompt import PromptTemplate, build_messages, load_prompt
from teacher.provider import (
    Completion,
    LLMProvider,
    StructuredMode,
    UnsupportedFeature,
    Usage,
)
from teacher.schema import TeacherAnalysis

#: Order the runner steps down through when a provider refuses a mode.
MODE_FALLBACK: tuple[StructuredMode, ...] = (
    StructuredMode.JSON_SCHEMA,
    StructuredMode.JSON_OBJECT,
    StructuredMode.NONE,
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class TeacherError(RuntimeError):
    pass


@dataclass
class Attempt:
    index: int
    mode: str
    ok: bool
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    latency_seconds: float = 0.0


@dataclass
class TeacherResult:
    """One analysis plus everything needed to reproduce or distrust it."""

    analysis: TeacherAnalysis | None
    grounding: GroundingReport | None
    ok: bool
    error: str | None

    # --- lineage -------------------------------------------------------
    teacher_model: str
    resolved_model: str | None
    served_by: str | None
    structured_mode: str
    prompt_version: str
    prompt_hash: str
    analysis_schema_version: str
    engine_version: str
    context_schema_version: str
    params_fingerprint: str
    symbol: str
    interval: str
    as_of: str
    bar_index: int
    generated_at: str

    # --- accounting ----------------------------------------------------
    attempts: list[Attempt] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    latency_seconds: float = 0.0

    disclaimer: str = ANALYSIS_DISCLAIMER

    def to_record(self) -> dict[str, Any]:
        """Flat JSON record for a dataset row."""
        record: dict[str, Any] = {
            "ok": self.ok,
            "error": self.error,
            "analysis": self.analysis.model_dump(mode="json") if self.analysis else None,
            "grounding": {
                "is_grounded": self.grounding.is_grounded if self.grounding else None,
                "grounded_claims": self.grounding.grounded_claims if self.grounding else 0,
                "total_claims": self.grounding.total_claims if self.grounding else 0,
                "issues": [asdict(i) for i in self.grounding.issues] if self.grounding else [],
            },
            "lineage": {
                "teacher_model": self.teacher_model,
                "resolved_model": self.resolved_model,
                "served_by": self.served_by,
                "structured_mode": self.structured_mode,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
                "analysis_schema_version": self.analysis_schema_version,
                "engine_version": self.engine_version,
                "context_schema_version": self.context_schema_version,
                "params_fingerprint": self.params_fingerprint,
                "generated_at": self.generated_at,
            },
            "example": {
                "symbol": self.symbol,
                "interval": self.interval,
                "as_of": self.as_of,
                "bar_index": self.bar_index,
            },
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost": self.cost,
                "latency_seconds": round(self.latency_seconds, 3),
                "attempts": [asdict(a) for a in self.attempts],
            },
            "disclaimer": self.disclaimer,
        }
        # Grounding findings are enum members; asdict leaves them as enums.
        for issue in record["grounding"]["issues"]:
            issue["finding"] = str(issue["finding"])
        return record


class TeacherRunner:
    def __init__(
        self,
        provider: LLMProvider,
        prompt_version: str = "v1",
        max_repairs: int = 2,
        allow_mode_fallback: bool = True,
    ) -> None:
        self.provider = provider
        self.template: PromptTemplate = load_prompt(prompt_version)
        self.max_repairs = max_repairs
        self.allow_mode_fallback = allow_mode_fallback
        self.schema = to_strict_schema(
            TeacherAnalysis, exclude=frozenset({"schema_version"})
        )
        self.schema_outline = describe_schema(self.schema)

    # ------------------------------------------------------------- public

    def render_messages(
        self, context: dict[str, Any], mode: StructuredMode = StructuredMode.JSON_SCHEMA
    ) -> list[dict[str, str]]:
        """Exposed so the prompt can be inspected, and costed, without calling."""
        outline = None if mode is StructuredMode.JSON_SCHEMA else self.schema_outline
        return build_messages(self.template, context, outline)

    def analyse(self, context: dict[str, Any]) -> TeacherResult:
        result = self._blank_result(context)
        modes = MODE_FALLBACK if self.allow_mode_fallback else (MODE_FALLBACK[0],)

        for mode in modes:
            messages = self.render_messages(context, mode)
            try:
                self._attempt_with_repairs(result, context, messages, mode)
            except UnsupportedFeature as exc:
                result.attempts.append(
                    Attempt(len(result.attempts), mode.value, ok=False, error=str(exc))
                )
                continue  # step down to a weaker mode
            except Exception as exc:  # provider or transport failure
                result.error = f"{type(exc).__name__}: {exc}"
                result.attempts.append(
                    Attempt(len(result.attempts), mode.value, ok=False, error=str(exc))
                )
                return result

            result.structured_mode = mode.value
            return result

        result.error = result.error or "no structured mode was accepted by the provider"
        return result

    # ------------------------------------------------------------ private

    def _attempt_with_repairs(
        self,
        result: TeacherResult,
        context: dict[str, Any],
        messages: list[dict[str, str]],
        mode: StructuredMode,
    ) -> None:
        conversation = list(messages)

        for repair in range(self.max_repairs + 1):
            completion = self.provider.complete(
                conversation,
                schema=self.schema if mode is StructuredMode.JSON_SCHEMA else None,
                schema_name="market_structure_analysis",
                mode=mode,
            )
            self._account(result, completion)

            try:
                analysis = parse_analysis(completion.text)
            except (json.JSONDecodeError, ValidationError) as exc:
                message = _format_errors(exc)
                result.attempts.append(
                    Attempt(
                        len(result.attempts), mode.value, ok=False, error=message,
                        prompt_tokens=completion.usage.prompt_tokens,
                        completion_tokens=completion.usage.completion_tokens,
                        cost=completion.usage.cost,
                        latency_seconds=completion.latency_seconds,
                    )
                )
                if repair == self.max_repairs:
                    result.error = f"schema validation failed after repairs: {message}"
                    return
                # The failed answer stays in the conversation on purpose: the
                # model needs to see what it produced to correct it.
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": completion.text},
                    {"role": "user", "content": self.template.render_repair(message)},
                ]
                continue

            result.attempts.append(
                Attempt(
                    len(result.attempts), mode.value, ok=True,
                    prompt_tokens=completion.usage.prompt_tokens,
                    completion_tokens=completion.usage.completion_tokens,
                    cost=completion.usage.cost,
                    latency_seconds=completion.latency_seconds,
                )
            )
            result.analysis = analysis
            result.grounding = verify(analysis, context)
            result.resolved_model = completion.model
            result.served_by = completion.served_by
            result.ok = True
            return

    def _account(self, result: TeacherResult, completion: Completion) -> None:
        usage: Usage = completion.usage
        result.prompt_tokens += usage.prompt_tokens
        result.completion_tokens += usage.completion_tokens
        result.latency_seconds += completion.latency_seconds
        if usage.cost is not None:
            result.cost = (result.cost or 0.0) + usage.cost

    def _blank_result(self, context: dict[str, Any]) -> TeacherResult:
        return TeacherResult(
            analysis=None,
            grounding=None,
            ok=False,
            error=None,
            teacher_model=self.provider.model,
            resolved_model=None,
            served_by=None,
            structured_mode="",
            prompt_version=self.template.version,
            prompt_hash=self.template.content_hash,
            analysis_schema_version=TEACHER_SCHEMA_VERSION,
            engine_version=str(context.get("engine_version", "")),
            context_schema_version=str(context.get("schema_version", "")),
            params_fingerprint=str(context.get("params_fingerprint", "")),
            symbol=str(context.get("symbol", "")),
            interval=str(context.get("interval", "")),
            as_of=str(context.get("as_of", "")),
            bar_index=int(context.get("bar_index", -1)),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


# ---------------------------------------------------------------- parsing


def parse_analysis(text: str) -> TeacherAnalysis:
    """Strip fences, parse, validate.

    Models told "return only JSON" still wrap it in ```json fences often
    enough that not handling it would inflate the failure rate with something
    that is not really a failure of the model's analysis.
    """
    cleaned = _FENCE.sub("", text).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    payload.pop("schema_version", None)  # ours to stamp, not the model's
    return TeacherAnalysis.model_validate(payload)


def _format_errors(exc: Exception) -> str:
    """Turn a validation failure into something a model can act on.

    Pydantic's default repr is long and full of URLs; the model needs the
    field path and what was wrong with it, nothing else.
    """
    if isinstance(exc, json.JSONDecodeError):
        return f"the response was not valid JSON: {exc.msg} at position {exc.pos}"
    lines = []
    for error in exc.errors()[:12]:
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)
