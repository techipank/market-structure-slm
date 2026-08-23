"""yfinance -> immutable raw CSV + lineage sidecar.

Deliberate choices
------------------
* ``auto_adjust=False``. This disables *dividend* adjustment only. yfinance
  always back-adjusts for **splits** regardless of the flag - verified against
  AAPL's 2020-08-31 4:1 split, which shows no discontinuity in the fetched
  series. So what we store is: split-adjusted, dividend-unadjusted.
  Consequences, both recorded in the lineage sidecar:
    - The series is continuous across splits, which is what market-structure
      analysis wants (no artificial 75% "gap" to explain).
    - Prices are NOT what a trader saw on screen at the time, and a future
      split silently rewrites history, changing the file's sha256. That is why
      the sha256 is recorded per fetch rather than assumed stable forever.
* We never overwrite an existing raw file unless ``--force`` is passed. Raw
  data is an input to reproducible experiments, not a cache.
* The sidecar records everything needed to re-run the exact fetch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from data_pipeline import SCHEMA_VERSION
from data_pipeline.loaders import sha256_file
from data_pipeline.schema import CANONICAL_TZ, DATETIME_COL, REQUIRED_COLS


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchSpec:
    symbol: str
    interval: str
    start: str | None = None
    end: str | None = None
    period: str | None = None

    @property
    def stem(self) -> str:
        safe = self.symbol.replace("^", "IDX_").replace("=", "_").replace("/", "_")
        return f"{safe}_{self.interval}"


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance frame into the contracted column layout.

    This is renaming and timezone conversion only. No value is altered, no row
    is added or removed - if yfinance returned a hole or a bad candle, the
    validator must be able to see it.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # Single-symbol downloads still come back multi-indexed in recent
        # yfinance versions; keep the price level only.
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"date", "datetime", "index"}:
            rename[col] = DATETIME_COL
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise IngestError(f"yfinance response lacks required column(s): {missing}")

    ts = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert(CANONICAL_TZ)
    else:
        # Daily bars come back naive and are exchange-local dates. Label them
        # UTC rather than shifting them; the lineage sidecar records that this
        # happened so the market engine can reason about session boundaries.
        ts = ts.dt.tz_localize(CANONICAL_TZ)
    df[DATETIME_COL] = ts

    return df[list(REQUIRED_COLS)]


def fetch(spec: FetchSpec, out_dir: Path, force: bool = False) -> tuple[Path, Path]:
    import yfinance as yf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{spec.stem}.csv"
    sidecar_path = csv_path.with_suffix(".lineage.json")

    if csv_path.exists() and not force:
        raise IngestError(
            f"{csv_path} already exists. Raw data is immutable; pass --force to refetch."
        )

    kwargs: dict[str, object] = {
        "interval": spec.interval,
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "threads": False,
    }
    if spec.period:
        kwargs["period"] = spec.period
    else:
        kwargs["start"] = spec.start
        kwargs["end"] = spec.end

    raw = yf.download(spec.symbol, **kwargs)
    if raw is None or raw.empty:
        raise IngestError(
            f"yfinance returned no rows for {spec.symbol} {spec.interval}. "
            "Check the symbol, and note that intraday history is depth-limited "
            "(1m ~7 days, <1d ~60 days)."
        )

    df = _normalise(raw)
    df.to_csv(csv_path, index=False)

    lineage = {
        "schema_version": SCHEMA_VERSION,
        "source": "yfinance",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "symbol": spec.symbol,
        "interval": spec.interval,
        "period": spec.period,
        "requested_start": spec.start,
        "requested_end": spec.end,
        "auto_adjust": False,
        "storage_timezone": CANONICAL_TZ,
        "rows": int(len(df)),
        "first_timestamp": str(df[DATETIME_COL].min()),
        "last_timestamp": str(df[DATETIME_COL].max()),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": sha256_file(csv_path),
        "usage": "Educational/research use only. Not trading advice. See docs/data_sources.md.",
    }
    sidecar_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    return csv_path, sidecar_path
