"""One test per failure mode, plus a clean-baseline test.

The clean-baseline test is the most important one here: a validator suite that
fires on healthy data is worse than no suite at all, because people learn to
ignore it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline.report import Severity, Verdict
from data_pipeline.schema import (
    CLOSE_COL,
    DATETIME_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    VOLUME_COL,
)
from data_pipeline.validate import validate_frame
from tests.conftest import as_loaded, make_clean_daily


def codes(report, severity: Severity | None = None) -> set[str]:
    return {
        i.code
        for c in report.checks
        for i in c.issues
        if severity is None or i.severity is severity
    }


# ------------------------------------------------------------------ baseline


def test_clean_data_passes(clean_daily):
    report = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    assert report.error_count == 0, codes(report, Severity.ERROR)
    assert report.verdict is Verdict.PASS, codes(report)


def test_clean_data_facts(clean_daily):
    report = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    assert report.dataset.rows == len(clean_daily)
    assert report.dataset.declared_interval == "1d"
    assert report.dataset.first_timestamp is not None
    assert report.dataset.last_timestamp is not None


# --------------------------------------------------------------- each defect


def test_missing_column(clean_daily):
    report = validate_frame(as_loaded(clean_daily.drop(columns=[VOLUME_COL])))
    assert "MISSING_COLUMN" in codes(report, Severity.ERROR)
    assert report.verdict is Verdict.FAIL


def test_non_numeric_price_becomes_null_error(clean_daily):
    df = clean_daily.copy()
    df.loc[5, CLOSE_COL] = np.nan  # what a text cell becomes after parsing
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "NULL_VALUE" in codes(report, Severity.ERROR)


def test_null_volume_is_only_a_warning(clean_daily):
    df = clean_daily.copy()
    df.loc[7, VOLUME_COL] = np.nan
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "NULL_VALUE" in codes(report, Severity.WARNING)
    assert report.error_count == 0


def test_out_of_order(clean_daily):
    df = clean_daily.copy()
    df.iloc[[10, 11]] = df.iloc[[11, 10]].values
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "OUT_OF_ORDER" in codes(report, Severity.ERROR)


def test_duplicate_timestamp(clean_daily):
    df = clean_daily.copy()
    df.loc[20, DATETIME_COL] = df.loc[19, DATETIME_COL]
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "DUPLICATE_TIMESTAMP" in codes(report, Severity.ERROR)


def test_duplicate_row(clean_daily):
    df = pd.concat([clean_daily, clean_daily.iloc[[30]]], ignore_index=True)
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "DUPLICATE_ROW" in codes(report, Severity.ERROR)


def test_mixed_timezone(clean_daily):
    report = validate_frame(as_loaded(clean_daily, original_tz="MIXED", mixed=True))
    assert "MIXED_TIMEZONE" in codes(report, Severity.ERROR)


def test_naive_timezone_warns(clean_daily):
    df = clean_daily.copy()
    df[DATETIME_COL] = df[DATETIME_COL].dt.tz_localize(None)
    report = validate_frame(as_loaded(df, original_tz=None), declared_interval="1d")
    assert "NAIVE_TIMEZONE" in codes(report, Severity.WARNING)
    assert report.error_count == 0


@pytest.mark.parametrize(
    ("target", "reference", "delta", "expected"),
    [
        (HIGH_COL, LOW_COL, -1.0, "HIGH_BELOW_LOW"),
        (HIGH_COL, OPEN_COL, -1.0, "HIGH_BELOW_OPEN"),
        (LOW_COL, CLOSE_COL, +1.0, "LOW_ABOVE_CLOSE"),
    ],
)
def test_ohlc_geometry(clean_daily, target, reference, delta, expected):
    df = clean_daily.copy()
    df.loc[3, target] = df.loc[3, reference] + delta
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert expected in codes(report, Severity.ERROR)


def test_non_positive_price(clean_daily):
    df = clean_daily.copy()
    df.loc[9, [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]] = 0.0
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "NON_POSITIVE_PRICE" in codes(report, Severity.ERROR)


def test_negative_volume(clean_daily):
    df = clean_daily.copy()
    df.loc[9, VOLUME_COL] = -1.0
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "NEGATIVE_VOLUME" in codes(report, Severity.ERROR)


def test_zero_volume_run_warns(clean_daily):
    df = clean_daily.copy()
    df.loc[10:25, VOLUME_COL] = 0.0
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "ZERO_VOLUME_RUN" in codes(report, Severity.WARNING)
    assert report.error_count == 0


def test_missing_candles_daily(clean_daily):
    # Remove a run of business days in the middle -> a hole a weekend cannot explain.
    df = clean_daily.drop(index=range(40, 48)).reset_index(drop=True)
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "MISSING_CANDLES" in codes(report, Severity.WARNING)


def test_single_day_holiday_is_info_not_warning(clean_daily):
    """Every US market holiday must not become a warning.

    Real data has ~9 of these a year; warning on them makes the check noise
    and trains the reader to ignore the whole report.
    """
    df = clean_daily.drop(index=[45]).reset_index(drop=True)
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "HOLIDAY_BREAK" in codes(report, Severity.INFO)
    assert "MISSING_CANDLES" not in codes(report)


def test_weekends_are_not_reported_as_missing(clean_daily):
    report = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    assert "MISSING_CANDLES" not in codes(report)


def test_missing_candles_intraday():
    idx = pd.date_range("2024-03-04 14:30", periods=60, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            DATETIME_COL: idx,
            OPEN_COL: 100.0,
            HIGH_COL: 101.0,
            LOW_COL: 99.0,
            CLOSE_COL: 100.5,
            VOLUME_COL: 1000.0,
        }
    )
    holed = frame.drop(index=range(20, 26)).reset_index(drop=True)
    report = validate_frame(as_loaded(holed), declared_interval="5m")
    assert "MISSING_CANDLES" in codes(report, Severity.WARNING)


def test_interval_mismatch_is_reported():
    frame = make_clean_daily(n=40)
    report = validate_frame(as_loaded(frame), declared_interval="1h")
    assert "INTERVAL_MISMATCH" in codes(report, Severity.WARNING)


def test_abnormal_gap(clean_daily):
    df = clean_daily.copy()
    # A 2:1 split printed unadjusted: everything from row 60 halves.
    df.loc[60:, [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]] /= 2.0
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "ABNORMAL_GAP" in codes(report, Severity.WARNING)
    assert report.error_count == 0


def test_abnormal_bar_range(clean_daily):
    df = clean_daily.copy()
    df.loc[50, HIGH_COL] = df.loc[50, CLOSE_COL] * 1.6
    report = validate_frame(as_loaded(df), declared_interval="1d")
    assert "ABNORMAL_BAR_RANGE" in codes(report, Severity.WARNING)


# ------------------------------------------------------------- report shape


def test_report_is_deterministic_apart_from_generated_at(clean_daily):
    a = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    b = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    da.pop("generated_at")
    db.pop("generated_at")
    assert da == db


def test_all_ten_checks_run(clean_daily):
    report = validate_frame(as_loaded(clean_daily), declared_interval="1d")
    assert len(report.checks) == 10
    assert len({c.check for c in report.checks}) == 10


def test_markdown_renders(clean_daily):
    df = clean_daily.copy()
    df.loc[3, HIGH_COL] = df.loc[3, LOW_COL] - 1
    md = validate_frame(as_loaded(df), declared_interval="1d").to_markdown()
    assert "FAIL" in md
    assert "HIGH_BELOW_LOW" in md
    assert "Not trading advice" in md
