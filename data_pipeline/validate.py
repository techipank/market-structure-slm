"""Runs the check suite over a loaded file and assembles the report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.loaders import LoadedFrame, load_ohlcv_csv
from data_pipeline.report import DatasetFacts, ValidationReport
from data_pipeline.schema import DATETIME_COL
from data_pipeline.validators import (
    ALL_CHECKS,
    CheckContext,
    Thresholds,
    infer_interval_seconds,
)


def _facts(loaded: LoadedFrame, declared_interval: str | None) -> DatasetFacts:
    f = loaded.frame
    first = last = None
    inferred = None
    if DATETIME_COL in f.columns and pd.api.types.is_datetime64_any_dtype(f[DATETIME_COL]):
        ts = f[DATETIME_COL]
        # min/max (which skip NaT), not iloc[0]/iloc[-1]: an out-of-order file
        # must still report its true span so split boundaries stay honest.
        # This reads the column; it never alters it.
        if ts.notna().any():
            first, last = str(ts.min()), str(ts.max())
        inferred = infer_interval_seconds(f[DATETIME_COL])
    return DatasetFacts(
        source_path=str(loaded.path),
        sha256=loaded.sha256,
        rows=int(len(f)),
        columns=[str(c) for c in f.columns],
        first_timestamp=first,
        last_timestamp=last,
        inferred_interval_seconds=inferred,
        declared_interval=declared_interval,
        timezone=loaded.original_tz or "naive",
    )


def validate_frame(
    loaded: LoadedFrame,
    declared_interval: str | None = None,
    thresholds: Thresholds | None = None,
) -> ValidationReport:
    ctx = CheckContext(
        frame=loaded.frame,
        declared_interval=declared_interval,
        original_tz=loaded.original_tz,
        mixed_timezones=loaded.mixed_timezones,
        thresholds=thresholds or Thresholds(),
    )
    report = ValidationReport(dataset=_facts(loaded, declared_interval))
    for check in ALL_CHECKS:
        report.checks.append(check(ctx))
    return report


def validate_file(
    path: Path,
    declared_interval: str | None = None,
    thresholds: Thresholds | None = None,
) -> ValidationReport:
    path = Path(path)
    if declared_interval is None:
        declared_interval = _interval_from_lineage(path)
    return validate_frame(load_ohlcv_csv(path), declared_interval, thresholds)


def _interval_from_lineage(csv_path: Path) -> str | None:
    """Read the declared interval from the sidecar written at ingest time.

    Provenance beats inference: if we know the feed was asked for 5m bars, an
    inferred 15m spacing is itself a finding rather than the ground truth.
    """
    sidecar = csv_path.with_suffix(".lineage.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")).get("interval")
    except (json.JSONDecodeError, OSError):
        return None


def write_report(report: ValidationReport, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.validation.json"
    md_path = out_dir / f"{stem}.validation.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return json_path, md_path
