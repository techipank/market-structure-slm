"""``msbench`` - sample, smoke-test, run, and report the teacher benchmark.

Intended order, cheapest first:

    msbench sample            free   build and inspect the frozen example set
    msbench smoke --model X   ~1 call  prove one model works end to end
    msbench run               paid   fan out across every candidate
    msbench report            free   aggregate, score, recommend

`smoke` exists because the expensive way to discover a broken request shape is
eight hundred calls into a fan-out.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from data_pipeline.config import load_config
from evaluation.benchmark import BenchmarkSpec, run_benchmark, summarise, write_csv
from evaluation.recommend import Gates, evaluate, render_report
from evaluation.sampling import (
    SampleSpec,
    build_samples,
    describe,
    read_samples,
    write_samples,
)
from market_engine.params import load_params
from teacher.batch import BatchProgress, Checkpoint
from teacher.cli import _load_env
from teacher.registry import first_env, load_registry, names
from teacher.runner import TeacherRunner

DEFAULT_SAMPLES = Path("data/benchmark/samples.jsonl")
DEFAULT_CHECKPOINT = Path("data/benchmark/results.jsonl")
DEFAULT_CSV = Path("data/benchmark/teacher_benchmark.csv")
DEFAULT_REPORT = Path("data/benchmark/teacher_benchmark.md")


def _cmd_sample(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    files = sorted(cfg.raw_dir.glob("*.csv"))
    if not files:
        print(f"no CSV files in {cfg.raw_dir}; run `msdata fetch` first", file=sys.stderr)
        return 2

    params = load_params(args.params)
    if args.window:
        # Override the candle window so the same bars can be re-rendered at a
        # different context size. Separation is pinned rather than left to
        # default to the window, otherwise changing the window would change
        # which bars get selected and the comparison would not be controlled.
        params = replace(params, ohlcv_window_bars=args.window)
        if args.min_separation is None:
            args.min_separation = SampleSpec().min_separation or 60

    spec = SampleSpec(
        target_count=args.count,
        cutoff=args.cutoff,
        min_separation=args.min_separation,
        seed=args.seed,
    )
    examples = build_samples(files, params, spec)
    write_samples(examples, Path(args.out))
    print(describe(examples))
    print(f"\nwrote {len(examples)} examples -> {args.out}")
    print(
        f"cutoff {spec.cutoff}: everything at or after this date is reserved "
        "for validation and test and was not sampled."
    )
    if len(examples) < args.count:
        print(
            f"\nNOTE: asked for {args.count}, got {len(examples)}. The separation "
            "rule or the cutoff exhausted the pool; lower --min-separation or "
            "add symbols.",
            file=sys.stderr,
        )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    _load_env()
    examples = read_samples(Path(args.samples))
    if not examples:
        print("no samples; run `msbench sample` first", file=sys.stderr)
        return 2
    example = examples[args.index]

    registry = load_registry(args.registry)
    spec = registry.spec(args.model)
    runner = TeacherRunner(
        registry.build(args.model),
        max_repairs=args.max_repairs,
        allow_mode_fallback=not args.strict_only,
    )
    print(
        f"calling {args.model} ({spec.endpoint} -> {spec.model}) "
        f"on {example.example_id} ({example.stratum})...\n"
    )
    result = runner.analyse(example.context)
    record = result.to_record()

    print(json.dumps(record, indent=2)[: args.max_print])
    print(
        f"\nok={result.ok} attempts={len(result.attempts)} mode={result.structured_mode} "
        f"tokens={result.prompt_tokens}/{result.completion_tokens} "
        f"cost={result.cost} latency={result.latency_seconds:.2f}s"
    )
    return 0 if result.ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    _load_env()
    examples = read_samples(Path(args.samples))
    if not examples:
        print("no samples; run `msbench sample` first", file=sys.stderr)
        return 2
    if args.limit:
        examples = examples[: args.limit]

    spec = BenchmarkSpec(
        models=tuple(args.model),
        concurrency=args.concurrency,
        requests_per_minute=args.rpm,
        consistency_repeats=args.consistency_repeats,
        consistency_examples=args.consistency_examples,
        max_repairs=args.max_repairs,
        allow_mode_fallback=args.allow_fallback,
    )

    calls = len(examples) * len(spec.models) + (
        min(spec.consistency_examples, len(examples))
        * len(spec.models)
        * (spec.consistency_repeats - 1)
    )
    registry = load_registry(args.registry)
    for alias in spec.models:
        registry.spec(alias)  # fail fast on a typo, before spending anything

    print(f"{len(examples)} examples x {len(spec.models)} models = ~{calls} calls")
    for alias in spec.models:
        s_ = registry.spec(alias)
        print(f"  {alias:12s} {s_.endpoint:12s} {s_.model}")
    print(f"checkpoint: {args.checkpoint} (re-running skips completed work)\n")
    if not args.yes:
        reply = input("this will spend money. proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("aborted")
            return 1

    last = {"line": ""}

    def on_progress(progress: BatchProgress) -> None:
        line = progress.line()
        if line != last["line"]:
            last["line"] = line
            print(f"\r{line}", end="", flush=True)

    run_benchmark(
        examples, spec, Path(args.checkpoint),
        on_progress=on_progress, registry=load_registry(args.registry),
    )
    print("\ndone. run `msbench report` to aggregate.")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    rows = Checkpoint(Path(args.checkpoint)).records()
    if not rows:
        print(f"no results in {args.checkpoint}", file=sys.stderr)
        return 2

    summaries = summarise(rows)
    write_csv(summaries, Path(args.csv))

    candidates = evaluate(
        summaries,
        dataset_size=args.dataset_size,
        gates=Gates(
            min_schema_compliance=args.min_schema_compliance,
            min_grounded_rate=args.min_grounded,
        ),
    )
    report = render_report(candidates, args.dataset_size, tolerance=args.tolerance)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")

    print(report)
    print(f"\nwrote {args.csv}")
    print(f"wrote {args.report}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    _load_env()
    registry = load_registry(args.registry)
    print(f"{'alias':10s} {'endpoint':12s} {'model':36s} status")
    for alias in registry.aliases():
        spec = registry.spec(alias)
        endpoint = registry.endpoints[spec.endpoint]
        problems = []
        try:
            model_id = spec.resolved_model()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            model_id, _ = "<unset>", problems.append(str(exc))
        # Only openai_compatible endpoints need a base URL; the OpenRouter
        # client carries its own default.
        if (
            endpoint.kind == "openai_compatible"
            and not first_env(endpoint.base_url_env)
            and not endpoint.base_url
        ):
            problems.append(f"{names(endpoint.base_url_env)} unset")
        if endpoint.api_key_env and not first_env(endpoint.api_key_env):
            problems.append(f"{names(endpoint.api_key_env)} unset")
        status = "READY" if not problems else "; ".join(problems)
        print(f"{alias:10s} {spec.endpoint:12s} {model_id:36s} {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msbench", description="Teacher model benchmark. Research use only."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="build the frozen example set (free)")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--params", default="configs/engine.yaml")
    p.add_argument("--count", type=int, default=200)
    p.add_argument(
        "--cutoff",
        default="2025-01-01",
        help="exclude bars at or after this date; reserves the holdout era",
    )
    p.add_argument("--min-separation", type=int, default=None)
    p.add_argument(
        "--window",
        type=int,
        help=(
            "override ohlcv_window_bars. Pins separation to 60 so the same bars "
            "are selected, making window size a controlled variable"
        ),
    )
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--out", default=str(DEFAULT_SAMPLES))
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("smoke", help="one call against one model, to prove it works")
    p.add_argument("--model", required=True, help="alias from configs/models.yaml")
    p.add_argument("--registry", default="configs/models.yaml")
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument(
        "--strict-only",
        action="store_true",
        help="fail rather than falling back to a weaker structured mode",
    )
    p.add_argument("--max-print", type=int, default=6000)
    p.set_defaults(func=_cmd_smoke)

    p = sub.add_parser("run", help="fan out across models (spends money)")
    p.add_argument(
        "--model", action="append", required=True,
        help="alias from configs/models.yaml; repeatable",
    )
    p.add_argument("--registry", default="configs/models.yaml")
    p.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--limit", type=int, help="use only the first N examples")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rpm", type=float, default=60.0)
    p.add_argument("--consistency-repeats", type=int, default=3)
    p.add_argument("--consistency-examples", type=int, default=15)
    p.add_argument("--max-repairs", type=int, default=2)
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="let models drop to a weaker structured mode (off: a fairer comparison)",
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("models", help="list configured model aliases (free)")
    p.add_argument("--registry", default="configs/models.yaml")
    p.set_defaults(func=_cmd_models)

    p = sub.add_parser("report", help="aggregate, score, recommend (free)")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--dataset-size", type=int, default=5000)
    p.add_argument("--tolerance", type=float, default=0.03)
    p.add_argument("--min-schema-compliance", type=float, default=0.90)
    p.add_argument("--min-grounded", type=float, default=0.80)
    p.set_defaults(func=_cmd_report)
    return parser


def _use_utf8_console() -> None:
    """Stop a non-ASCII character from killing a run that already succeeded.

    The Windows console defaults to cp1252, and printing the report's warning
    glyph raised UnicodeEncodeError *after* the CSV and Markdown had been
    written - a crash exit code on completed work, which is the worst kind of
    lie a CLI can tell. Reconfiguring is preferred over dropping the glyph:
    the report is read on other platforms too.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
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
