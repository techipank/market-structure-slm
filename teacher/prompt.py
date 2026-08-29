"""Prompt loading, versioning, and rendering.

Prompts are files under `teacher/prompts/<version>/`, not string literals in
code. Three reasons:

* A prompt change alters the meaning of every example generated afterwards.
  Versioning it as a directory makes "which prompt produced this row" a fact
  rather than an archaeology exercise through git history.
* The content hash goes on every result. Two rows claiming prompt `v1` with
  different hashes is a detectable bug; an edited string literal is not.
* Prompts are edited far more often than code, by people reading them as prose.

Message layout is deliberate. The system message is identical for every
example in a run, so providers that cache prompt prefixes can reuse it; the
per-example context goes in the user message. Putting the context in the
system message would defeat that and cost real money at dataset scale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_VERSION = "v1"


class PromptError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user: str
    repair: str

    @property
    def content_hash(self) -> str:
        """Hash of all three files together.

        One hash for the set, not three: they are edited as a unit and a
        result only needs to answer "was this the same prompt?".
        """
        digest = hashlib.sha256()
        for part in (self.system, self.user, self.repair):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:16]

    def render_user(self, context: dict[str, Any]) -> str:
        return self.user.format(
            symbol=context.get("symbol", "UNKNOWN"),
            interval=context.get("interval", "UNKNOWN"),
            as_of=context.get("as_of", "UNKNOWN"),
            # Compact, not pretty-printed. Indenting this context costs 1,285
            # tokens per example - 37% of the payload - to transmit whitespace.
            # Across a five-thousand example run that is over six million
            # tokens spent on indentation. Models parse compact JSON fine.
            context_json=json.dumps(context, separators=(",", ":"), sort_keys=False),
        )

    def render_repair(self, errors: str) -> str:
        return self.repair.format(errors=errors)


def load_prompt(version: str = DEFAULT_PROMPT_VERSION) -> PromptTemplate:
    directory = PROMPTS_DIR / version
    if not directory.is_dir():
        available = sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_dir())
        raise PromptError(f"unknown prompt version {version!r}; available: {available}")

    parts: dict[str, str] = {}
    for name in ("system", "user", "repair"):
        path = directory / f"{name}.md"
        if not path.exists():
            raise PromptError(f"prompt version {version!r} is missing {name}.md")
        parts[name] = path.read_text(encoding="utf-8")

    return PromptTemplate(version=version, **parts)


def build_messages(
    template: PromptTemplate, context: dict[str, Any], schema_outline: str | None = None
) -> list[dict[str, str]]:
    """Two messages: a stable system prompt and the per-example context.

    `schema_outline` is appended to the system message whenever it is given.
    It costs a few hundred prompt tokens, and it is worth them even when
    `response_format` is also sent: a provider that does not support
    structured outputs accepts the request and ignores the constraint, and a
    model that was never told the field names then invents its own. Observed
    on nvidia/nemotron-3.5-lightning, which wrote a well-formed object made
    entirely of keys we do not have.
    """
    system = template.system
    if schema_outline:
        system = (
            f"{system}\n\n## Required output shape\n\n"
            "Emit exactly these fields, and only these fields.\n\n"
            f"{schema_outline}\n"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": template.render_user(context)},
    ]
