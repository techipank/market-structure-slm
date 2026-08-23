"""Command line entry point: ``msdata fetch`` and ``msdata validate``.

Exit codes are part of the contract so this can sit in CI:
    0 -> PASS or PASS_WITH_WARNINGS
    1 -> at least one ERROR-severity finding
    2 -> the command itself failed (bad path, network error)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_pipeline.config import DEFAULT_CONFIG, load_config
from data_pipeline.ingest import IngestError, fetch
from data_pipeline.report import Verdict
from data_pipeline.validate import validate_file, write_report

DISCLAIMER = "Educational/research use only. Not trading advice."


def _cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    failures = 0
    for spec in cfg.symbols:
        try:
            csv_path, sidecar = fetch(spec, cfg.raw_dir, force=args.force)
        except IngestError as exc:
            print(f"[SKIP] {spec.symbol} {spec.interval}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"[OK]   {spec.symbol} {spec.interval} -> {csv_path}")
        print(f"       lineage -> {sidecar}")
    return 2 if failures and failures == len(cfg.symbols) else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    targets: list[Path]
    if args.path:
        p = Path(args.path)
        targets = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    else:
        targets = sorted(cfg.raw_dir.glob("*.csv"))

    if not targets:
        print("no CSV files to validate", file=sys.stderr)
        return 2

    worst_is_fail = False
    for path in targets:
        report = validate_file(path, args.interval, cfg.thresholds)
        json_path, md_path = write_report(report, cfg.report_dir, path.stem)
        verdict = report.verdict
        worst_is_fail |= verdict is Verdict.FAIL
        print(
            f"{verdict.value:<19} {path.name}  "
            f"rows={report.dataset.rows} errors={report.error_count} "
            f"warnings={report.warning_count}"
        )
        print(f"                    report -> {json_path}")
        print(f"                    report -> {md_path}")
        if args.verbose:
            for check in report.checks:
                for issue in check.issues:
                    print(f"    {issue.severity.value:<7} {issue.code:<22} {issue.message}")

    print(f"\n{DISCLAIMER}")
    return 1 if worst_is_fail else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msdata",
        description="OHLCV ingestion and validation. " + DISCLAIMER,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to data config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download OHLCV from yfinance into data/raw")
    p_fetch.add_argument("--force", action="store_true", help="overwrite existing raw files")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_val = sub.add_parser("validate", help="run the check suite and write reports")
    p_val.add_argument("path", nargs="?", help="CSV file or directory (default: config raw_dir)")
    p_val.add_argument("--interval", help="override the declared interval, e.g. 1d")
    p_val.add_argument("-v", "--verbose", action="store_true", help="print every finding")
    p_val.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
