"""``msteacher`` - run the teacher over a market context.

Subcommands
-----------
``dry-run``  render the prompt and estimate its size. No API call, no cost.
``analyse``  call the model, validate, verify grounding, print or save.
``schema``   dump the strict JSON Schema actually sent to the provider.

`dry-run` exists because the most expensive way to discover a prompt bug is
during a paid dataset run. Rendering and inspecting the exact bytes that would
be sent costs nothing and catches most mistakes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from data_pipeline.loaders import load_ohlcv_csv
from data_pipeline.report import Verdict
from data_pipeline.validate import validate_file
from market_engine.context import MarketEngine
from market_engine.params import load_params
from teacher import ANALYSIS_DISCLAIMER
from teacher.jsonschema import to_strict_schema
from teacher.openrouter import OpenRouterConfig, OpenRouterProvider
from teacher.prompt import DEFAULT_PROMPT_VERSION
from teacher.runner import TeacherRunner
from teacher.schema import TeacherAnalysis

#: Rough characters-per-token. Every model tokenises differently and this is
#: not a substitute for the provider's own count - it is a sanity check on
#: prompt size before spending anything.
CHARS_PER_TOKEN = 4


def _load_env() -> None:
    """Read .env if present. Never overrides an already-set variable."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _context(path: Path, bar: int, allow_failed: bool) -> dict:
    report = validate_file(path)
    if report.verdict is Verdict.FAIL and not allow_failed:
        raise SystemExit(
            f"{path.name} failed validation ({report.error_count} errors); refusing to run."
        )
    engine = MarketEngine(load_params())
    symbol, _, interval = path.stem.rpartition("_")
    computed = engine.compute(load_ohlcv_csv(path).frame, symbol or path.stem, interval or "1d")
    index = bar if bar >= 0 else computed.n + bar
    return engine.context_at(computed, index).model_dump(mode="json", exclude_none=True)


def _cmd_dry_run(args: argparse.Namespace) -> int:
    context = _context(Path(args.path), args.bar, args.allow_failed)
    runner = _offline_runner(args.prompt_version)
    messages = runner.render_messages(context)

    total = 0
    for message in messages:
        chars = len(message["content"])
        total += chars
        print(f"--- {message['role']}: {chars} chars (~{chars // CHARS_PER_TOKEN} tokens)")
    schema_chars = len(json.dumps(runner.schema))
    print(f"--- response_format schema: {schema_chars} chars "
          f"(~{schema_chars // CHARS_PER_TOKEN} tokens)")
    grand = total + schema_chars
    print(f"\nestimated input: ~{grand // CHARS_PER_TOKEN} tokens per example")
    print("(estimate only: every model tokenises differently)")

    if args.show:
        print("\n" + "=" * 70)
        for message in messages:
            print(f"\n########## {message['role'].upper()} ##########\n")
            print(message["content"])
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    _load_env()
    context = _context(Path(args.path), args.bar, args.allow_failed)

    config = OpenRouterConfig.from_env(
        model=args.model,
        temperature=args.temperature,
        only_providers=tuple(args.provider or ()),
        allow_fallbacks=not args.pin_provider,
    )
    runner = TeacherRunner(
        OpenRouterProvider(config),
        prompt_version=args.prompt_version,
        max_repairs=args.max_repairs,
    )
    result = runner.analyse(context)
    record = result.to_record()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        _print_summary(result)
    return 0 if result.ok else 1


def _print_summary(result) -> None:
    print(f"model        : {result.teacher_model} -> {result.resolved_model or '?'}")
    print(f"served by    : {result.served_by or 'unreported'}")
    print(f"mode         : {result.structured_mode}")
    print(f"prompt       : {result.prompt_version} ({result.prompt_hash})")
    print(f"attempts     : {len(result.attempts)}")
    print(
        f"tokens       : {result.prompt_tokens} in / {result.completion_tokens} out"
        f"   cost: {result.cost if result.cost is not None else 'unreported'}"
    )
    print(f"latency      : {result.latency_seconds:.2f}s")

    if not result.ok:
        print(f"\nFAILED: {result.error}")
        return

    a = result.analysis
    print(f"\nHTF bias     : {a.higher_timeframe_bias}")
    print(f"structure    : {a.market_structure}")
    print(f"market state : {a.market_state}")
    print(f"setup        : {a.setup_type}")
    print(f"confidence   : {a.confidence}")
    print(f"\nreasoning    : {a.reasoning_summary}")

    g = result.grounding
    print(f"\ngrounding    : {g.grounded_claims}/{g.total_claims} claims verified")
    if g.issues:
        print(f"  {len(g.issues)} issue(s):")
        for issue in g.issues[:10]:
            print(f"    {issue.finding.value:<20} {issue.location}: {issue.detail}")
    else:
        print("  no issues")
    print(f"\n{ANALYSIS_DISCLAIMER}")


def _cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(to_strict_schema(TeacherAnalysis, frozenset({"schema_version"})), indent=2))
    return 0


def _offline_runner(prompt_version: str) -> TeacherRunner:
    """A runner with a provider that refuses to be called.

    dry-run must never make a network request, and constructing a real
    provider would demand an API key the user may not have set yet.
    """

    class _NoCall:
        model = "<dry-run>"

        def complete(self, *args, **kwargs):  # pragma: no cover - never called
            raise RuntimeError("dry-run must not call a provider")

    return TeacherRunner(_NoCall(), prompt_version=prompt_version)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msteacher", description=ANALYSIS_DISCLAIMER)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler, needs_model in (
        ("dry-run", _cmd_dry_run, False),
        ("analyse", _cmd_analyse, True),
    ):
        p = sub.add_parser(name, help=handler.__doc__ or name)
        p.add_argument("path", help="validated OHLCV CSV")
        p.add_argument("--bar", type=int, default=-1, help="bar index; negative from the end")
        p.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
        p.add_argument("--allow-failed", action="store_true")
        if needs_model:
            p.add_argument("--model", help="OpenRouter model id (default: $TEACHER_MODEL)")
            p.add_argument("--temperature", type=float, default=0.0)
            p.add_argument("--max-repairs", type=int, default=2)
            p.add_argument("--provider", action="append", help="pin to a provider slug")
            p.add_argument("--pin-provider", action="store_true", help="disable fallbacks")
            p.add_argument("--json", action="store_true", help="print the full record")
            p.add_argument("--out", help="write the record to this path")
        else:
            p.add_argument("--show", action="store_true", help="print the rendered prompt")
        p.set_defaults(func=handler)

    p_schema = sub.add_parser("schema", help="print the strict JSON Schema sent to the provider")
    p_schema.set_defaults(func=_cmd_schema)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except Exception as exc:  # keep secrets out of tracebacks
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
