"""Turn a pydantic model into a JSON Schema a constrained decoder will accept.

Why this module exists
----------------------
`model_json_schema()` produces a *validation* schema. Strict structured-output
modes want a *generation* schema, and the two differ in ways that cause
confusing runtime rejections:

1. Every object must set ``additionalProperties: false``.
2. Every property must appear in ``required``. Optionality is expressed by
   allowing ``null`` in the type, not by omitting the key. Pydantic already
   emits ``anyOf: [..., {"type": "null"}]`` for ``X | None``, so the shape is
   right; only the ``required`` list needs fixing.
3. Validation keywords the decoder cannot enforce - ``minItems``,
   ``minLength``, ``pattern``, ``minimum``, ``format``, ``default`` - are
   rejected outright by strict mode rather than ignored.
4. ``$defs``/``$ref`` support is **not documented** by OpenRouter, and support
   is per-provider rather than per-model. Pydantic emits a ``$ref`` for every
   nested model, so the schema is dereferenced into a self-contained tree
   before it is sent. Our models have no recursive types, so inlining always
   terminates; a guard raises rather than looping if that ever changes.

Point 3 has a consequence worth internalising: **the schema sent to the model
is deliberately weaker than the schema we validate against.** The decoder
guarantees shape; pydantic guarantees the rest (at least two supporting
evidence items, confidence within 0-1); and the repair loop closes the gap by
handing pydantic's complaints back to the model. Three layers, each doing what
it is actually capable of, instead of one layer pretending to do all three.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

#: Keywords a strict decoder rejects. Stripped from the generation schema and
#: enforced by pydantic after parsing instead.
UNSUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "default",
        "examples",
        "uniqueItems",
    }
)


def to_strict_schema(
    model: type[BaseModel], exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Generation-ready JSON Schema for `model`.

    `exclude` drops top-level properties the model should not be asked to
    produce - `schema_version` is ours to stamp, not the teacher's to guess.
    """
    schema = copy.deepcopy(model.model_json_schema())

    for name in exclude:
        schema.get("properties", {}).pop(name, None)
        if name in schema.get("required", []):
            schema["required"].remove(name)

    schema = inline_refs(schema)
    _tighten(schema)
    return schema


class SchemaError(RuntimeError):
    pass


def inline_refs(schema: dict[str, Any], max_depth: int = 12) -> dict[str, Any]:
    """Replace every ``$ref`` with its definition and drop ``$defs``.

    Produces a larger but self-contained schema, which is the safer thing to
    put on the wire when reference support is undocumented and varies by
    provider. Cost is a few hundred extra tokens once per request; the benefit
    is not discovering the limitation halfway through a paid dataset run.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, depth: int) -> Any:
        if depth > max_depth:
            raise SchemaError(
                "schema nesting exceeded the inlining depth limit; "
                "a recursive model type would do this"
            )
        if isinstance(node, list):
            return [resolve(item, depth) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name not in defs:
                raise SchemaError(f"unresolvable $ref: {node['$ref']}")
            target = resolve(copy.deepcopy(defs[name]), depth + 1)
            # Keywords alongside the $ref (a description, usually) win.
            extras = {k: v for k, v in node.items() if k != "$ref"}
            return {**target, **extras}
        return {k: resolve(v, depth) for k, v in node.items()}

    out = resolve({k: v for k, v in schema.items() if k != "$defs"}, 0)
    return out


def _tighten(node: Any) -> None:
    """Recursively strip unsupported keywords and close every object."""
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    for keyword in list(node):
        if keyword in UNSUPPORTED_KEYWORDS:
            del node[keyword]

    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        properties = node.get("properties", {})
        # Every property required; nullability already lives in the type.
        node["required"] = list(properties.keys())

    for value in node.values():
        _tighten(value)


def describe_schema(schema: dict[str, Any]) -> str:
    """Compact field listing for embedding in a prompt.

    Sending the raw JSON Schema in the prompt as well as in `response_format`
    would roughly double the fixed prompt cost for no benefit. A terse outline
    is enough for models that do not support strict mode and still need to know
    the shape.
    """
    lines: list[str] = []
    defs = schema.get("$defs", {})  # empty once inlined; kept for raw schemas

    def render(node: dict[str, Any], prefix: str, depth: int) -> None:
        if depth > 3:
            return
        for name, field in node.get("properties", {}).items():
            resolved = _resolve(field, defs)
            kind = _kind(resolved, defs)
            description = (field.get("description") or resolved.get("description") or "").strip()
            suffix = f": {description}" if description else ""
            lines.append(f"{'  ' * depth}- {prefix}{name} ({kind}){suffix}")
            child = _item_object(resolved, defs)
            if child:
                render(child, "", depth + 1)

    render(schema, "", 0)
    return "\n".join(lines)


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if ref:
        return defs.get(ref.rsplit("/", 1)[-1], {})
    for option in node.get("anyOf", []):
        if option.get("type") != "null":
            return _resolve(option, defs)
    return node


def _kind(node: dict[str, Any], defs: dict[str, Any]) -> str:
    if "enum" in node:
        return " | ".join(str(v) for v in node["enum"])
    if node.get("type") == "array":
        inner = _resolve(node.get("items", {}), defs)
        return f"list of {inner.get('title') or inner.get('type', 'object')}"
    return str(node.get("type", node.get("title", "object")))


def _item_object(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any] | None:
    if node.get("type") == "array":
        inner = _resolve(node.get("items", {}), defs)
        return inner if "properties" in inner else None
    return node if "properties" in node else None
