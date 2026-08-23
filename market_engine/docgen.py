"""Generate the market-context data dictionary from the pydantic models.

Written rather than hand-maintained so the docs cannot drift from the schema.
A hand-written field table is accurate on the day it is written and misleading
three commits later, and a stale definition is worse than none once a model is
being trained to produce these fields.
"""

from __future__ import annotations

from pydantic import BaseModel

from market_engine import CONTEXT_SCHEMA_VERSION, DISCLAIMER, ENGINE_VERSION
from market_engine.schema import (
    Candle,
    HigherTimeframeSnapshot,
    IndicatorSnapshot,
    LevelRef,
    MarketContext,
    RegimeSnapshot,
    SetupRef,
    StructureEventRef,
    StructureSnapshot,
    SwingRef,
    VolatilitySnapshot,
)

MODELS: tuple[type[BaseModel], ...] = (
    MarketContext,
    Candle,
    IndicatorSnapshot,
    StructureSnapshot,
    SwingRef,
    StructureEventRef,
    LevelRef,
    VolatilitySnapshot,
    RegimeSnapshot,
    HigherTimeframeSnapshot,
    SetupRef,
)


def iter_fields(model: type[BaseModel]):
    for name, info in model.model_fields.items():
        yield name, _type_name(info.annotation), (info.description or "").strip()


def _type_name(annotation) -> str:
    text = str(annotation)
    text = text.replace("typing.", "").replace("market_engine.schema.", "")
    text = text.replace("<class '", "").replace("'>", "")
    return text.replace(" | None", " (optional)")


def render_data_dictionary() -> str:
    lines = [
        "# Market context — data dictionary",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Schema version `{CONTEXT_SCHEMA_VERSION}` · engine version `{ENGINE_VERSION}`",
        "",
        "**Generated from `market_engine/schema.py` — do not edit by hand.**",
        "Regenerate with `msengine schema`.",
        "",
        "Every value below is computed deterministically from candles at or before",
        "the bar being described. No field may depend on a later bar.",
        "",
    ]
    for model in MODELS:
        lines.append(f"## `{model.__name__}`")
        doc = (model.__doc__ or "").strip().splitlines()
        if doc:
            lines += ["", doc[0].strip(), ""]
        else:
            lines.append("")
        lines += ["| field | type | definition |", "|---|---|---|"]
        for name, type_name, description in iter_fields(model):
            lines.append(f"| `{name}` | `{type_name}` | {description} |")
        lines.append("")
    return "\n".join(lines)
