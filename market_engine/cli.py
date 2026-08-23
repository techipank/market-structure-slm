"""``msengine`` - build market contexts from validated OHLCV files.

Subcommands
-----------
``context``  emit one context, or a sampled set, as JSON
``summary``  human-readable description of what the engine found in a file
``schema``   dump the JSON Schema and regenerate the data dictionary

The engine refuses to run on a file whose validation verdict is FAIL. Features
computed from impossible candles are worse than no features: they look
plausible, propagate into the teacher's analysis, and are undetectable by the
time anyone notices the model is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_pipeline.config import load_config
from data_pipeline.loaders import load_ohlcv_csv
from data_pipeline.report import Verdict
from data_pipeline.validate import validate_file
from market_engine import DISCLAIMER
from market_engine.context import MarketEngine
from market_engine.docgen import render_data_dictionary
from market_engine.params import DEFAULT_CONFIG, load_params
from market_engine.schema import MarketContext


def _load(path: Path, allow_failed: bool) -> tuple[object, str, str]:
    report = validate_file(path)
    if report.verdict is Verdict.FAIL and not allow_failed:
        raise SystemExit(
            f"{path.name} failed validation ({report.error_count} errors). "
            "Fix the data or re-run with --allow-failed if you know what you are doing."
        )
    loaded = load_ohlcv_csv(path)
    symbol, _, interval = path.stem.rpartition("_")
    return loaded.frame, symbol or path.stem, interval or "1d"


def _cmd_context(args: argparse.Namespace) -> int:
    params = load_params(args.params)
    engine = MarketEngine(params)
    frame, symbol, interval = _load(Path(args.path), args.allow_failed)
    computed = engine.compute(frame, symbol, interval)

    if args.bar is not None:
        indices = [args.bar if args.bar >= 0 else computed.n + args.bar]
    else:
        # Evenly spaced samples across the usable range, skipping the warm-up
        # where the long EMAs do not yet exist.
        start = min(params.max_ema_period, computed.n - 1)
        span = max(1, computed.n - start)
        step = max(1, span // max(1, args.count))
        indices = list(range(start, computed.n, step))[: args.count]

    contexts = [engine.context_at(computed, i) for i in indices]
    payload = [c.model_dump(mode="json", exclude_none=True) for c in contexts]
    text = json.dumps(payload if len(payload) > 1 else payload[0], indent=2)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {len(payload)} context(s) -> {args.out}")
    else:
        print(text)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    params = load_params(args.params)
    engine = MarketEngine(params)
    cfg = load_config(args.config)
    targets = (
        [Path(args.path)]
        if args.path
        else sorted(cfg.raw_dir.glob("*.csv"))
    )
    if not targets:
        print("no CSV files found", file=sys.stderr)
        return 2

    for path in targets:
        frame, symbol, interval = _load(path, args.allow_failed)
        computed = engine.compute(frame, symbol, interval)
        ctx = engine.context_at(computed, computed.n - 1)
        events = computed.structure.events
        print(f"\n=== {symbol} {interval}  ({computed.n} bars, as of {ctx.as_of[:10]}) ===")
        print(f"  swings confirmed : {len(computed.swings)}")
        print(f"  structure events : {len(events)}")
        print(f"  trend / bias     : {ctx.structure.trend} / {ctx.structure.bias}")
        print(
            f"  volatility       : {ctx.volatility.regime} "
            f"(pct={ctx.volatility.atr_percentile})"
        )
        print(
            f"  regime           : {ctx.regime.trend_regime} "
            f"(ER={ctx.regime.efficiency_ratio})"
        )
        print(f"  levels           : {len(ctx.levels)}")
        htf = ctx.higher_timeframe
        print(
            f"  higher timeframe : {htf.interval} {htf.trend}/{htf.bias} as of {htf.as_of[:10]}"
            if htf
            else "  higher timeframe : unavailable"
        )
        print(f"  setups           : {[s.name for s in ctx.setups] or 'none'}")
    print(f"\n{DISCLAIMER}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    schema = MarketContext.model_json_schema()
    if args.json:
        print(json.dumps(schema, indent=2))
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_data_dictionary(), encoding="utf-8")
    print(f"wrote data dictionary -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msengine", description=DISCLAIMER)
    parser.add_argument("--params", default=str(DEFAULT_CONFIG), help="engine config YAML")
    parser.add_argument("--config", default="configs/data.yaml", help="data config YAML")
    parser.add_argument(
        "--allow-failed", action="store_true", help="run even on files that failed validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ctx = sub.add_parser("context", help="emit MarketContext JSON")
    p_ctx.add_argument("path", help="path to a validated OHLCV CSV")
    p_ctx.add_argument("--bar", type=int, help="bar index; negative counts from the end")
    p_ctx.add_argument("--count", type=int, default=1, help="number of sampled bars")
    p_ctx.add_argument("--out", help="write to this file instead of stdout")
    p_ctx.set_defaults(func=_cmd_context)

    p_sum = sub.add_parser("summary", help="human-readable engine summary")
    p_sum.add_argument("path", nargs="?", help="CSV file (default: every file in raw_dir)")
    p_sum.set_defaults(func=_cmd_summary)

    p_schema = sub.add_parser("schema", help="dump JSON Schema / regenerate the data dictionary")
    p_schema.add_argument("--json", action="store_true", help="print raw JSON Schema")
    p_schema.add_argument("--out", default="docs/market_context.md")
    p_schema.set_defaults(func=_cmd_schema)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
