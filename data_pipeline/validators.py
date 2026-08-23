"""The ten validation checks.

Contract for every validator in this module:

* signature ``(ctx: CheckContext) -> CheckResult``
* pure: it must never mutate ``ctx.frame`` and never write to disk
* it must degrade gracefully when an earlier check already failed (e.g. the
  OHLC invariant check simply reports nothing if the columns are absent)

Severity policy is documented on ``Severity``. Short version: physically
impossible -> ERROR, merely unusual -> WARNING.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_pipeline.report import CheckResult, Issue, Severity
from data_pipeline.schema import (
    CALENDAR_INTERVALS,
    CLOSE_COL,
    DATETIME_COL,
    HIGH_COL,
    INTERVAL_SECONDS,
    LOW_COL,
    OPEN_COL,
    PRICE_COLS,
    REQUIRED_COLS,
    VOLUME_COL,
)

SAMPLE_LIMIT = 10


@dataclass(frozen=True)
class Thresholds:
    """Tunable knobs for the two heuristic checks.

    They are heuristics on purpose: there is no universal definition of an
    "abnormal" gap across an illiquid small cap and an index future. The values
    live in ``configs/data.yaml`` so an analysis run records what it used.
    """

    #: A gap must exceed this fraction of the previous close to be considered
    #: at all, regardless of what the robust z-score says. Stops us flagging
    #: 0.2% gaps on a series whose typical gap is 0.01%.
    min_gap_pct: float = 0.02
    #: Robust z-score multiplier (median + k * 1.4826 * MAD).
    gap_mad_multiplier: float = 8.0
    #: Single-bar (high-low)/close above this is reported.
    max_bar_range_pct: float = 0.25
    #: Consecutive zero-volume bars above this count are reported.
    max_zero_volume_run: int = 5
    #: On calendar intervals, a gap spanning at most this many business days is
    #: treated as an exchange holiday (INFO) rather than a hole (WARNING).
    #: Empirically, 11 years of US equity data contains 103 two-business-day
    #: gaps (single-day holidays) and none longer than three, so a lower value
    #: makes the check fire on every Thanksgiving and become noise.
    max_holiday_business_days: int = 3


@dataclass
class CheckContext:
    frame: pd.DataFrame
    declared_interval: str | None = None
    original_tz: str | None = None
    mixed_timezones: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def has_datetime(self) -> bool:
        return DATETIME_COL in self.frame.columns and pd.api.types.is_datetime64_any_dtype(
            self.frame[DATETIME_COL]
        )


# ----------------------------------------------------------------- helpers


def _issue(
    code: str,
    severity: Severity,
    message: str,
    idx,
    ctx: CheckContext,
) -> Issue:
    positions = [int(i) for i in list(idx)]
    sample = positions[:SAMPLE_LIMIT]
    stamps: list[str] = []
    if ctx.has_datetime:
        series = ctx.frame[DATETIME_COL]
        for p in sample:
            try:
                stamps.append(str(series.iloc[p]))
            except (IndexError, KeyError):  # pragma: no cover - defensive
                stamps.append("<out-of-range>")
    return Issue(
        code=code,
        severity=severity,
        message=message,
        row_indices=sample,
        row_timestamps=stamps,
        count=len(positions),
    )


def _result(check: str, description: str, issues: list[Issue]) -> CheckResult:
    fatal = any(i.severity is Severity.ERROR for i in issues)
    return CheckResult(check=check, description=description, passed=not fatal, issues=issues)


def infer_interval_seconds(ts: pd.Series) -> int | None:
    """Modal positive spacing between consecutive timestamps.

    We use the *mode*, not the mean or min: real series contain weekend gaps
    (which would blow up the mean) and occasional duplicate/near-duplicate
    stamps (which would collapse the min). The mode is the bar size the feed
    actually intends.
    """
    clean = ts.dropna().sort_values()
    if len(clean) < 3:
        return None
    deltas = clean.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return int(deltas.mode().iloc[0].total_seconds())


# ------------------------------------------------------------------ checks


def check_required_columns(ctx: CheckContext) -> CheckResult:
    missing = [c for c in REQUIRED_COLS if c not in ctx.frame.columns]
    issues: list[Issue] = []
    if missing:
        issues.append(
            Issue(
                code="MISSING_COLUMN",
                severity=Severity.ERROR,
                message="required column(s) absent: " + ", ".join(missing),
                count=len(missing),
            )
        )
    extra = [str(c) for c in ctx.frame.columns if c not in REQUIRED_COLS]
    if extra:
        issues.append(
            Issue(
                code="EXTRA_COLUMN",
                severity=Severity.INFO,
                message="columns outside the contract, ignored downstream: " + ", ".join(extra),
                count=len(extra),
            )
        )
    return _result(
        "required_columns",
        "every column named in the data contract is present",
        issues,
    )


def check_dtypes(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    if DATETIME_COL in ctx.frame.columns and not ctx.has_datetime:
        issues.append(
            Issue(
                code="DATETIME_UNPARSEABLE",
                severity=Severity.ERROR,
                message="the datetime column could not be parsed as timestamps",
                count=1,
            )
        )
    for col in (*PRICE_COLS, VOLUME_COL):
        if col in ctx.frame.columns and not pd.api.types.is_numeric_dtype(ctx.frame[col]):
            issues.append(
                Issue(
                    code="NON_NUMERIC_COLUMN",
                    severity=Severity.ERROR,
                    message=f"column '{col}' is not numeric after parsing",
                    count=1,
                )
            )
    return _result("dtypes", "columns parse to their contracted types", issues)


def check_missing_values(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    for col in REQUIRED_COLS:
        if col not in ctx.frame.columns:
            continue
        mask = ctx.frame[col].isna()
        if mask.any():
            severity = Severity.WARNING if col == VOLUME_COL else Severity.ERROR
            issues.append(
                _issue(
                    "NULL_VALUE",
                    severity,
                    f"column '{col}' has null/unparseable values",
                    np.flatnonzero(mask.to_numpy()),
                    ctx,
                )
            )
    return _result("missing_values", "no nulls in contracted columns", issues)


def check_chronological_order(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    if ctx.has_datetime:
        ts = ctx.frame[DATETIME_COL]
        # A row is out of order when it is strictly earlier than its predecessor.
        backwards = (ts < ts.shift(1)).fillna(False)
        if backwards.any():
            issues.append(
                _issue(
                    "OUT_OF_ORDER",
                    Severity.ERROR,
                    "timestamps are not monotonically increasing",
                    np.flatnonzero(backwards.to_numpy()),
                    ctx,
                )
            )
    return _result(
        "chronological_order",
        "rows are ordered oldest -> newest (required to avoid lookahead bugs)",
        issues,
    )


def check_duplicates(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    if ctx.has_datetime:
        ts = ctx.frame[DATETIME_COL]
        dup_ts = ts.duplicated(keep="first") & ts.notna()
        if dup_ts.any():
            issues.append(
                _issue(
                    "DUPLICATE_TIMESTAMP",
                    Severity.ERROR,
                    "the same timestamp appears more than once",
                    np.flatnonzero(dup_ts.to_numpy()),
                    ctx,
                )
            )
    full_dup = ctx.frame.duplicated(keep="first")
    if full_dup.any():
        issues.append(
            _issue(
                "DUPLICATE_ROW",
                Severity.ERROR,
                "byte-identical duplicate rows",
                np.flatnonzero(full_dup.to_numpy()),
                ctx,
            )
        )
    return _result("duplicates", "no duplicate timestamps or rows", issues)


def check_timezone_consistency(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    if ctx.mixed_timezones:
        issues.append(
            Issue(
                code="MIXED_TIMEZONE",
                severity=Severity.ERROR,
                message=(
                    "the datetime column mixes offsets or naive and aware values; "
                    "this silently corrupts ordering across DST boundaries"
                ),
                count=1,
            )
        )
    elif ctx.original_tz is None and DATETIME_COL in ctx.frame.columns:
        issues.append(
            Issue(
                code="NAIVE_TIMEZONE",
                severity=Severity.WARNING,
                message=(
                    "timestamps carry no timezone; they were assumed to already be UTC. "
                    "Record the exchange timezone in the lineage sidecar."
                ),
                count=1,
            )
        )
    return _result(
        "timezone_consistency",
        "all timestamps share one unambiguous timezone",
        issues,
    )


def check_ohlc_relationships(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    if not all(c in ctx.frame.columns for c in PRICE_COLS):
        return _result("ohlc_relationships", "high/low bracket open and close", issues)

    f = ctx.frame
    o, h, low, c = f[OPEN_COL], f[HIGH_COL], f[LOW_COL], f[CLOSE_COL]
    valid = o.notna() & h.notna() & low.notna() & c.notna()

    rules = {
        "HIGH_BELOW_LOW": (h < low, "high < low"),
        "HIGH_BELOW_OPEN": (h < o, "high < open"),
        "HIGH_BELOW_CLOSE": (h < c, "high < close"),
        "LOW_ABOVE_OPEN": (low > o, "low > open"),
        "LOW_ABOVE_CLOSE": (low > c, "low > close"),
    }
    for code, (raw_mask, human) in rules.items():
        mask = raw_mask & valid
        if mask.any():
            issues.append(
                _issue(
                    code,
                    Severity.ERROR,
                    "impossible candle geometry: " + human,
                    np.flatnonzero(mask.to_numpy()),
                    ctx,
                )
            )
    return _result(
        "ohlc_relationships",
        "high >= max(open, close) and low <= min(open, close)",
        issues,
    )


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def check_value_domains(ctx: CheckContext) -> CheckResult:
    issues: list[Issue] = []
    f = ctx.frame
    for col in PRICE_COLS:
        if col not in f.columns:
            continue
        mask = (f[col] <= 0) & f[col].notna()
        if mask.any():
            issues.append(
                _issue(
                    "NON_POSITIVE_PRICE",
                    Severity.ERROR,
                    f"column '{col}' contains values <= 0",
                    np.flatnonzero(mask.to_numpy()),
                    ctx,
                )
            )
    if VOLUME_COL in f.columns:
        neg = (f[VOLUME_COL] < 0) & f[VOLUME_COL].notna()
        if neg.any():
            issues.append(
                _issue(
                    "NEGATIVE_VOLUME",
                    Severity.ERROR,
                    "volume is negative",
                    np.flatnonzero(neg.to_numpy()),
                    ctx,
                )
            )
        zero = (f[VOLUME_COL] == 0).fillna(False)
        run = _longest_run(zero.to_numpy())
        if run > ctx.thresholds.max_zero_volume_run:
            issues.append(
                Issue(
                    code="ZERO_VOLUME_RUN",
                    severity=Severity.WARNING,
                    message=(
                        f"{run} consecutive zero-volume bars "
                        f"(threshold {ctx.thresholds.max_zero_volume_run}); "
                        "often a padded or synthetic session"
                    ),
                    count=run,
                )
            )
    return _result("value_domains", "prices positive, volume non-negative", issues)


def check_missing_candles(ctx: CheckContext) -> CheckResult:
    """Detect holes in the timestamp grid.

    Why this is heuristic: we do not carry an exchange trading calendar in
    here. So we infer the bar size, then split every larger-than-expected
    spacing into two buckets - breaks that look like a session boundary
    (weekend, or a jump across a UTC date) and breaks inside a single session.
    Only the latter is suspicious enough to warn about.
    """
    issues: list[Issue] = []
    description = "timestamp grid has no unexplained holes"
    if not ctx.has_datetime:
        return _result("missing_candles", description, issues)

    ts = ctx.frame[DATETIME_COL]
    declared = INTERVAL_SECONDS.get(ctx.declared_interval or "")
    inferred = infer_interval_seconds(ts)
    step = declared or inferred
    if step is None:
        return _result("missing_candles", description, issues)

    if inferred and declared and inferred != declared:
        issues.append(
            Issue(
                code="INTERVAL_MISMATCH",
                severity=Severity.WARNING,
                message=(
                    f"declared interval is {ctx.declared_interval} ({declared}s) but the modal "
                    f"spacing in the file is {inferred}s"
                ),
                count=1,
            )
        )

    deltas = ts.diff().dt.total_seconds()
    oversized = (deltas > step * 1.5).fillna(False).to_numpy()
    if not oversized.any():
        return _result("missing_candles", description, issues)

    calendar_mode = (ctx.declared_interval in CALENDAR_INTERVALS) or step >= 86_400
    intra_session: list[int] = []
    session_breaks: list[int] = []
    holiday_breaks: list[int] = []
    for pos in np.flatnonzero(oversized):
        prev, cur = ts.iloc[pos - 1], ts.iloc[pos]
        if pd.isna(prev) or pd.isna(cur):
            continue
        if calendar_mode:
            # Three buckets, because without an exchange calendar we can only
            # rank plausibility: one business day apart is a weekend, a few
            # days is a holiday, a week or more is a real hole in the feed.
            business_days = int(np.busday_count(prev.date(), cur.date()))
            if business_days <= 1:
                session_breaks.append(int(pos))
            elif business_days <= ctx.thresholds.max_holiday_business_days:
                holiday_breaks.append(int(pos))
            else:
                intra_session.append(int(pos))
        else:
            if prev.date() == cur.date():
                intra_session.append(int(pos))
            else:
                session_breaks.append(int(pos))

    if intra_session:
        missing_est = int(sum(max(0, round(deltas.iloc[p] / step) - 1) for p in intra_session))
        issues.append(
            _issue(
                "MISSING_CANDLES",
                Severity.WARNING,
                f"{len(intra_session)} unexplained hole(s) in the grid, "
                f"~{missing_est} bar(s) absent",
                intra_session,
                ctx,
            )
        )
    if holiday_breaks:
        issues.append(
            _issue(
                "HOLIDAY_BREAK",
                Severity.INFO,
                f"{len(holiday_breaks)} gap(s) of 2-"
                f"{ctx.thresholds.max_holiday_business_days} business days, "
                "consistent with an exchange holiday",
                holiday_breaks,
                ctx,
            )
        )
    if session_breaks:
        issues.append(
            _issue(
                "SESSION_BREAK",
                Severity.INFO,
                f"{len(session_breaks)} spacing jump(s) consistent with a session or weekend break",
                session_breaks,
                ctx,
            )
        )
    return _result("missing_candles", description, issues)


def check_abnormal_gaps(ctx: CheckContext) -> CheckResult:
    """Statistical outlier detection on overnight gaps and bar ranges.

    Robust statistics (median + MAD) rather than mean + stdev, because a single
    50% gap would inflate the stdev enough to hide itself. 1.4826 is the
    constant that makes MAD a consistent estimator of sigma for normal data.

    These are WARNINGs by construction: a 20% gap can be a real earnings move
    or an unadjusted split. The gate flags it; a human decides.
    """
    issues: list[Issue] = []
    description = "no statistical outliers in gaps or bar ranges"
    f = ctx.frame
    if not {OPEN_COL, CLOSE_COL}.issubset(f.columns):
        return _result("abnormal_gaps", description, issues)

    prev_close = f[CLOSE_COL].shift(1)
    gap_pct = ((f[OPEN_COL] - prev_close).abs() / prev_close.abs()).replace(
        [np.inf, -np.inf], np.nan
    )
    clean = gap_pct.dropna()
    if len(clean) >= 10:
        med = float(clean.median())
        mad = float((clean - med).abs().median())
        cutoff = max(
            ctx.thresholds.min_gap_pct,
            med + ctx.thresholds.gap_mad_multiplier * 1.4826 * mad,
        )
        mask = (gap_pct > cutoff).fillna(False)
        if mask.any():
            issues.append(
                _issue(
                    "ABNORMAL_GAP",
                    Severity.WARNING,
                    f"open gapped more than {cutoff:.2%} from the previous close "
                    "(possible unadjusted corporate action or bad tick)",
                    np.flatnonzero(mask.to_numpy()),
                    ctx,
                )
            )

    if {HIGH_COL, LOW_COL}.issubset(f.columns):
        rng = ((f[HIGH_COL] - f[LOW_COL]) / f[CLOSE_COL].abs()).replace([np.inf, -np.inf], np.nan)
        mask = (rng > ctx.thresholds.max_bar_range_pct).fillna(False)
        if mask.any():
            issues.append(
                _issue(
                    "ABNORMAL_BAR_RANGE",
                    Severity.WARNING,
                    f"single-bar range exceeds {ctx.thresholds.max_bar_range_pct:.0%} of close",
                    np.flatnonzero(mask.to_numpy()),
                    ctx,
                )
            )
    return _result("abnormal_gaps", description, issues)


#: Execution order matters only for readability of the report; the checks are
#: independent by design.
ALL_CHECKS = (
    check_required_columns,
    check_dtypes,
    check_missing_values,
    check_chronological_order,
    check_duplicates,
    check_timezone_consistency,
    check_ohlc_relationships,
    check_value_domains,
    check_missing_candles,
    check_abnormal_gaps,
)
