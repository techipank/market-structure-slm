"""Config loading for the data pipeline.

Config is a *recorded input*, not ambient state: the thresholds a validation
run used are written into the report so a report from six months ago can be
interpreted correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from data_pipeline.ingest import FetchSpec
from data_pipeline.validators import Thresholds

DEFAULT_CONFIG = Path("configs/data.yaml")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    report_dir: Path
    thresholds: Thresholds
    symbols: list[FetchSpec]


def load_config(path: Path | str = DEFAULT_CONFIG) -> DataConfig:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    paths = doc.get("paths", {})
    thr = doc.get("thresholds", {})
    defaults = doc.get("defaults", {})

    specs: list[FetchSpec] = []
    for entry in doc.get("symbols", []):
        specs.append(
            FetchSpec(
                symbol=entry["symbol"],
                interval=entry.get("interval", defaults.get("interval", "1d")),
                start=entry.get("start", defaults.get("start")),
                end=entry.get("end", defaults.get("end")),
                period=entry.get("period", defaults.get("period")),
            )
        )

    return DataConfig(
        raw_dir=Path(paths.get("raw_dir", "data/raw")),
        report_dir=Path(paths.get("report_dir", "data/reports")),
        thresholds=Thresholds(
            min_gap_pct=float(thr.get("min_gap_pct", 0.02)),
            gap_mad_multiplier=float(thr.get("gap_mad_multiplier", 8.0)),
            max_bar_range_pct=float(thr.get("max_bar_range_pct", 0.25)),
            max_zero_volume_run=int(thr.get("max_zero_volume_run", 5)),
            max_holiday_business_days=int(thr.get("max_holiday_business_days", 3)),
        ),
        symbols=specs,
    )
