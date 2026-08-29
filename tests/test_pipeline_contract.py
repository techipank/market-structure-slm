"""Guardrail tests for the properties that make the pipeline trustworthy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from data_pipeline.cli import main
from data_pipeline.ingest import FetchSpec, _normalise
from data_pipeline.loaders import load_ohlcv_csv, sha256_file
from data_pipeline.report import Verdict
from data_pipeline.schema import DATETIME_COL, HIGH_COL, LOW_COL, REQUIRED_COLS
from data_pipeline.validate import validate_file
from tests.conftest import make_clean_daily

PKG = Path(__file__).resolve().parents[1] / "data_pipeline"

#: Any of these appearing in the load/validate path means data is being
#: silently altered before the checks can see it.
FORBIDDEN = (
    "fillna",
    "dropna",
    "drop_duplicates",
    "sort_values",
    "ffill",
    "bfill",
    "interpolate",
    "clip",
)

#: Modules that must never repair. `ingest` is exempt only for column renaming,
#: and is covered by its own assertions below.
NO_REPAIR_MODULES = ("loaders.py", "validate.py")


def test_load_and_validate_never_repair():
    for name in NO_REPAIR_MODULES:
        src = (PKG / name).read_text(encoding="utf-8")
        # Strip comments and docstrings so prose mentioning fillna is allowed.
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        code = re.sub(r"#.*", "", code)
        for token in FORBIDDEN:
            assert token not in code, f"{name} calls {token}() in the validation path"


def test_validators_only_use_fillna_on_boolean_masks():
    """`.fillna(False)` on a comparison mask is not data repair.

    It only decides how NaN-vs-number comparisons are treated when building an
    index of offending rows. Assert nothing else is filled.
    """
    src = (PKG / "validators.py").read_text(encoding="utf-8")
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = re.sub(r"#.*", "", code)
    for match in re.findall(r"fillna\(([^)]*)\)", code):
        assert match.strip() == "False", f"unexpected fillna({match})"
    for token in ("dropna", "drop_duplicates", "ffill", "bfill", "interpolate"):
        # dropna on a *derived statistic* series is fine; on the frame it is not.
        assert f"frame.{token}" not in code
        assert f"ctx.frame.{token}" not in code


def test_roundtrip_csv_preserves_values(tmp_path: Path):
    frame = make_clean_daily(n=30)
    csv = tmp_path / "round.csv"
    frame.to_csv(csv, index=False)
    loaded = load_ohlcv_csv(csv)
    assert list(loaded.frame.columns) == list(REQUIRED_COLS)
    assert len(loaded.frame) == 30
    assert str(loaded.original_tz) == "UTC"
    pd.testing.assert_series_equal(
        loaded.frame[HIGH_COL].round(8),
        frame[HIGH_COL].round(8),
        check_names=False,
    )


def test_sha256_is_stable(tmp_path: Path):
    csv = tmp_path / "hash.csv"
    make_clean_daily(n=10).to_csv(csv, index=False)
    assert sha256_file(csv) == sha256_file(csv)
    assert len(sha256_file(csv)) == 64


def test_normalise_flattens_multiindex_and_renames():
    """Mirrors the shape recent yfinance returns for a single symbol."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            ("Open", "SPY"): [1.0, 2.0, 3.0],
            ("High", "SPY"): [2.0, 3.0, 4.0],
            ("Low", "SPY"): [0.5, 1.5, 2.5],
            ("Close", "SPY"): [1.5, 2.5, 3.5],
            ("Volume", "SPY"): [10, 20, 30],
        },
        index=idx,
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    raw.index.name = "Date"

    out = _normalise(raw)
    assert list(out.columns) == list(REQUIRED_COLS)
    assert str(out[DATETIME_COL].dt.tz) == "UTC"
    # Naive daily dates must be labelled, not shifted.
    assert out[DATETIME_COL].iloc[0].strftime("%Y-%m-%d") == "2024-01-01"


def test_fetch_spec_stem_is_filesystem_safe():
    assert FetchSpec("^VIX", "1d").stem == "IDX_VIX_1d"
    assert FetchSpec("EURUSD=X", "1h").stem == "EURUSD_X_1h"
    # NSE tickers: a dot would confuse suffix handling, an ampersand is a
    # shell metacharacter.
    assert FetchSpec("RELIANCE.NS", "1d").stem == "RELIANCE_NS_1d"
    assert FetchSpec("M&M.NS", "1d").stem == "M_M_NS_1d"


# ------------------------------------------------------------- CLI contract


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "data.yaml"
    cfg.write_text(
        "paths:\n"
        f"  raw_dir: {(tmp_path / 'raw').as_posix()}\n"
        f"  report_dir: {(tmp_path / 'reports').as_posix()}\n"
        "thresholds: {}\n"
        "symbols: []\n",
        encoding="utf-8",
    )
    return cfg


def test_cli_exit_zero_on_clean(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_clean_daily(n=60).to_csv(raw / "GOOD_1d.csv", index=False)
    cfg = _write_config(tmp_path)

    rc = main(["--config", str(cfg), "validate", "--interval", "1d"])
    assert rc == 0
    assert "Not trading advice" in capsys.readouterr().out
    assert (tmp_path / "reports" / "GOOD_1d.validation.json").exists()
    assert (tmp_path / "reports" / "GOOD_1d.validation.md").exists()


def test_cli_exit_one_on_error(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = make_clean_daily(n=60)
    df.loc[5, HIGH_COL] = df.loc[5, LOW_COL] - 1.0
    df.to_csv(raw / "BAD_1d.csv", index=False)
    cfg = _write_config(tmp_path)

    assert main(["--config", str(cfg), "validate", "--interval", "1d"]) == 1

    payload = json.loads(
        (tmp_path / "reports" / "BAD_1d.validation.json").read_text(encoding="utf-8")
    )
    assert payload["verdict"] == "FAIL"
    assert payload["error_count"] >= 1
    assert payload["dataset"]["sha256"]


def test_report_never_leaks_prices(tmp_path: Path):
    """Reports are committable; the data is not. They must not carry prices."""
    raw = tmp_path / "raw"
    raw.mkdir()
    df = make_clean_daily(n=60)
    csv = raw / "LEAK_1d.csv"
    df.to_csv(csv, index=False)
    report = validate_file(csv, "1d")
    text = report.to_json() + report.to_markdown()
    for value in df[HIGH_COL].head(20):
        assert f"{value:.6f}" not in text


def test_declared_interval_read_from_lineage_sidecar(tmp_path: Path):
    csv = tmp_path / "SPY_1d.csv"
    make_clean_daily(n=40).to_csv(csv, index=False)
    csv.with_suffix(".lineage.json").write_text(
        json.dumps({"interval": "1d", "symbol": "SPY"}), encoding="utf-8"
    )
    report = validate_file(csv)
    assert report.dataset.declared_interval == "1d"
    assert report.verdict in (Verdict.PASS, Verdict.PASS_WITH_WARNINGS)


def test_unclosed_trailing_bar_is_dropped_and_recorded():
    """Fetching during market hours returns today's bar as nulls. Archiving it
    fails the missing-values check on every file - a gate crying wolf."""
    from data_pipeline.ingest import _drop_unclosed_tail

    # Shaped on what the source actually returns: the forming bar has traded,
    # so open, high, low and volume are populated and only `close` is absent.
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-08-21", "2026-08-24"], utc=True),
        "open": [10.0, 10.6], "high": [11.0, 10.9],
        "low": [9.0, 10.2], "close": [10.5, float("nan")],
        "volume": [100.0, 83251.0],
    })
    kept, dropped = _drop_unclosed_tail(frame)
    assert len(kept) == 1
    assert dropped is not None and dropped.startswith("2026-08-24")


def test_a_hole_in_the_middle_is_not_dropped():
    """Only the last row, and only when every price is null. A null mid-series
    is a genuine data hole and must still fail validation."""
    from data_pipeline.ingest import _drop_unclosed_tail

    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24"], utc=True),
        "open": [10.0, float("nan"), 12.0], "high": [11.0, float("nan"), 13.0],
        "low": [9.0, float("nan"), 11.0], "close": [10.5, float("nan"), 12.5],
        "volume": [100.0, 0.0, 90.0],
    })
    kept, dropped = _drop_unclosed_tail(frame)
    assert len(kept) == 3 and dropped is None
