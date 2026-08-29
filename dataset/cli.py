"""``msdataset`` - sample, generate, and inspect the distillation corpus.

Intended order:

    msdataset sample              free   choose and freeze the example set
    msdataset generate --dry-run  free   what it will cost, before it costs it
    msdataset generate            spends the teacher's budget
    msdataset regate              free   re-apply the gate to existing results
    msdataset describe            free   what the corpus contains

`regate` exists because the gate will be tightened at least once. Re-judging
answers already paid for costs nothing, and rebuilding the corpus from the
checkpoint is preferable to a second generation run that would produce
different answers to the same questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from data_pipeline.config import load_config
from dataset.build import BuildSpec, build, finalise, render
from dataset.quality import GateSpec
from dataset.render import VIEWS, estimate_tokens, render_pair
from evaluation.sampling import SampleSpec, build_samples, describe, read_samples, write_samples
from market_engine.params import load_params
from teacher.batch import Checkpoint
from teacher.cli import _load_env
from teacher.jsonschema import describe_schema, to_strict_schema
from teacher.prompt import load_prompt
from teacher.registry import load_registry
from teacher.runner import TeacherRunner
from teacher.schema import TeacherAnalysis

DEFAULT_DIR = Path("data/dataset")
DEFAULT_EXAMPLES = DEFAULT_DIR / "examples.jsonl"


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _cmd_sample(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    files = sorted(cfg.raw_dir.glob("*.csv"))
    if not files:
        print(f"no CSV files in {cfg.raw_dir}; run `msdata fetch` first", file=sys.stderr)
        return 2

    spec = SampleSpec(target_count=args.count, cutoff=args.cutoff, seed=args.seed)
    examples = build_samples(files, load_params(args.params), spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_samples(examples, out)
    print(describe(examples))
    print(f"\nwrote {len(examples)} examples -> {out}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    _load_env()
    examples = read_samples(Path(args.examples))
    if not examples:
        print(f"no examples in {args.examples}; run `msdataset sample` first", file=sys.stderr)
        return 2
    if args.limit:
        examples = examples[: args.limit]

    spec = _spec(args)
    out_dir = Path(args.out)

    done = 0
    checkpoint_path = out_dir / "results.jsonl"
    if checkpoint_path.exists():
        done = len(Checkpoint(checkpoint_path).records())
    remaining = max(0, len(examples) - done)

    if args.dry_run:
        registry = load_registry(args.registry)
        runner = TeacherRunner(registry.build(spec.model), prompt_version=spec.prompt_version)
        messages = runner.render_messages(examples[0].context)
        chars = sum(len(m["content"]) for m in messages)
        print(
            f"{len(examples)} examples, {done} already done, {remaining} calls to make\n"
            f"prompt is ~{chars:,} characters; measured billing on this prompt family runs\n"
            f"~35% above chars/4, so budget ~{int(chars / 4 * 1.35):,} prompt tokens per call\n"
            f"total ~{int(chars / 4 * 1.35) * remaining:,} prompt tokens"
        )
        return 0

    print(f"{len(examples)} examples ({done} already in the checkpoint) -> {out_dir}")
    if not args.yes and remaining:
        reply = input(f"make {remaining} calls to {spec.model}? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("aborted")
            return 1

    def on_progress(progress) -> None:
        # BatchProgress renders itself. Formatting it here again is how the
        # first version of this invented two field names that do not exist.
        print("\r" + progress.line(), end="", flush=True)

    summary = build(examples, spec, out_dir, on_progress=on_progress)
    print("\n")
    print(render(summary, args.cutoff))
    _write_summary(out_dir, summary)
    return 0


def _cmd_regate(args: argparse.Namespace) -> int:
    """Re-judge existing results without calling anything."""
    examples = read_samples(Path(args.examples))
    out_dir = Path(args.out)
    rows = Checkpoint(out_dir / "results.jsonl").records()
    if not rows:
        print(f"no results in {out_dir}; run `msdataset generate` first", file=sys.stderr)
        return 2

    summary = finalise(examples, rows, _spec(args), out_dir)
    print(render(summary, args.cutoff))
    _write_summary(out_dir, summary)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    path = Path(args.out) / "corpus.jsonl"
    if not path.exists():
        print(f"{path} does not exist; run `msdataset generate` first", file=sys.stderr)
        return 2
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        print("corpus is empty", file=sys.stderr)
        return 2

    splits = Counter(r["split"] for r in rows)
    print(f"corpus            {len(rows)} rows -> {path}")
    print("split             " + ", ".join(f"{k}={v}" for k, v in sorted(splits.items())))
    print("interval          " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(str(r["interval"]) for r in rows).items())))
    print(f"symbols           {len({r['symbol'] for r in rows})}")
    print(f"strata            {len({r['stratum'] for r in rows})} populated")
    print(f"date range        {min(r['as_of'] for r in rows)[:10]} .. "
          f"{max(r['as_of'] for r in rows)[:10]}")

    teachers = Counter(r["lineage"]["resolved_model"] for r in rows)
    prompts = Counter(r["lineage"]["prompt_hash"] for r in rows)
    print(f"teacher           {', '.join(f'{k} x{v}' for k, v in teachers.items())}")
    if len(prompts) > 1:
        # Rows produced by different prompts are not one corpus; say so loudly.
        print(f"WARNING: {len(prompts)} distinct prompt hashes in one corpus: {dict(prompts)}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Render the corpus into training pairs. Free - no calls, no network."""
    corpus = Path(args.out) / "corpus.jsonl"
    if not corpus.exists():
        print(f"{corpus} does not exist; run `msdataset generate` first", file=sys.stderr)
        return 2
    rows = [json.loads(x) for x in corpus.read_text(encoding="utf-8").splitlines() if x.strip()]

    template = load_prompt(args.prompt_version)
    outline = None
    if args.include_schema_outline:
        # Parity with what the teacher was sent when the decoder could not be
        # trusted to enforce the shape. Off by default: a fine-tuned student
        # learns the shape from the targets, and the outline is ~900 tokens on
        # every single training example.
        outline = describe_schema(to_strict_schema(TeacherAnalysis,
                                                   exclude=frozenset({"schema_version"})))

    views = [VIEWS[name] for name in (args.view or list(VIEWS))]
    dest = Path(args.dest)
    for view in views:
        pairs, dropped = [], []
        for row in rows:
            pair = render_pair(row, view, template, outline)
            (pairs if pair else dropped).append(pair or row["example_id"])

        by_split = Counter(p["split"] for p in pairs)
        view_dir = dest / view.name
        view_dir.mkdir(parents=True, exist_ok=True)
        for split in sorted(by_split):
            path = view_dir / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for pair in pairs:
                    if pair["split"] == split:
                        handle.write(json.dumps(pair, separators=(",", ":")) + "\n")

        # The whole sequence, prompt *and* target: that is what a training run
        # has to fit, and it runs above the teacher's billed prompt_tokens by
        # roughly the length of the answer.
        lengths = sorted(
            estimate_tokens("".join(m["content"] for m in p["messages"])) for p in pairs
        )
        print(f"{view.name:8} {view.description}")
        print(f"         {len(pairs)} pairs ("
              + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())) + ")"
              + (f", {len(dropped)} dropped" if dropped else ""))
        if lengths:
            print(f"         sequence tokens: median {lengths[len(lengths) // 2]:,}, "
                  f"p95 {lengths[int(len(lengths) * 0.95)]:,}, max {lengths[-1]:,}")
        if dropped:
            # A view that cannot show a row's evidence is a fact about the
            # view, and it must be visible rather than silently absorbed.
            print(f"         dropped because the view removed cited evidence: "
                  f"{len(dropped)} rows")
        print(f"         -> {view_dir}")
    return 0


def _spec(args: argparse.Namespace) -> BuildSpec:
    gate = GateSpec(require_first_attempt=args.require_first_attempt)
    return BuildSpec(
        model=args.model,
        concurrency=args.concurrency,
        requests_per_minute=args.rpm,
        max_repairs=args.max_repairs,
        val_start=args.val_start,
        gate=gate,
    )


def _write_summary(out_dir: Path, summary: dict) -> None:
    path = Path(out_dir) / "corpus_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msdataset",
        description="Build the distillation corpus. Research use only.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="choose and freeze the example set (free)")
    p_sample.add_argument("--config", default="configs/data.yaml")
    p_sample.add_argument("--params", default="configs/engine.yaml")
    p_sample.add_argument("--count", type=int, default=1200)
    p_sample.add_argument("--cutoff", default=SampleSpec().cutoff)
    p_sample.add_argument("--seed", type=int, default=SampleSpec().seed)
    p_sample.add_argument("--out", default=str(DEFAULT_EXAMPLES))
    p_sample.set_defaults(func=_cmd_sample)

    for name, help_text, func in (
        ("generate", "call the teacher and build the corpus", _cmd_generate),
        ("regate", "re-apply the gate to existing results (free)", _cmd_regate),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--model", default="mimo", help="alias from configs/models.yaml")
        p.add_argument("--registry", default="configs/models.yaml")
        p.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
        p.add_argument("--out", default=str(DEFAULT_DIR))
        p.add_argument("--limit", type=int, help="use only the first N examples")
        p.add_argument("--concurrency", type=int, default=8)
        p.add_argument("--rpm", type=float, default=90.0)
        p.add_argument("--max-repairs", type=int, default=2)
        p.add_argument("--cutoff", default=SampleSpec().cutoff)
        p.add_argument("--val-start", default=BuildSpec(model="x").val_start,
                       help="examples on or after this date go to the validation split")
        p.add_argument("--require-first-attempt", action="store_true",
                       help="reject answers that were only valid after a repair")
        p.add_argument("--dry-run", action="store_true", help="cost the run without making it")
        p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
        p.set_defaults(func=func)

    p_export = sub.add_parser("export", help="render the corpus into training pairs (free)")
    p_export.add_argument("--out", default=str(DEFAULT_DIR), help="corpus directory")
    p_export.add_argument("--dest", default=str(DEFAULT_DIR / "training"))
    p_export.add_argument("--view", action="append", choices=sorted(VIEWS),
                          help="repeatable; default is every view")
    p_export.add_argument("--prompt-version", default="v1")
    p_export.add_argument(
        "--include-schema-outline", action="store_true",
        help="append the schema outline to the system prompt (~900 tokens per example)",
    )
    p_export.set_defaults(func=_cmd_export)

    p_desc = sub.add_parser("describe", help="what the corpus contains (free)")
    p_desc.add_argument("--out", default=str(DEFAULT_DIR))
    p_desc.set_defaults(func=_cmd_describe)
    return parser


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
