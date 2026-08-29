"""Screen a candidate universe on activity and volatility, before fetching it.

Why this is a command and not a one-off script
----------------------------------------------
The instrument universe is a modelling decision: it decides which market
regimes the teacher ever sees, and therefore what the student can learn. A set
chosen by reputation ("the big liquid names") cannot be defended later, and
cannot be re-derived when the data changes. So the screen is code, its output
is a report, and `docs/data_sources.md` cites the numbers it produced.

What is measured, and why these three
-------------------------------------
* **Median turnover** (close x volume) - "is it actually traded". The median
  rather than the mean, because a single frenzied session should not qualify
  an otherwise thin name.
* **Median ATR%** - "does it move". Again the median: a robust measure of
  typical range, not of the worst day.
* **Annualised stdev of returns** - deliberately *not* used for ranking. It is
  computed only so it can be divided by the ATR-based figure.

That ratio is the interesting one. For a clean series the two volatility
measures track each other at roughly 11:1. An unadjusted corporate action
leaves one enormous return in the series, which inflates the stdev while the
median ATR shrugs it off, so the ratio spikes: measured at 16 for a 3:2 split
and 28 for a demerger. It is a corporate-action detector built out of
measurements already being taken, and on this data source that matters,
because yfinance does not reliably adjust NSE splits and demergers.

A direct jump detector runs alongside it, since the ratio is a summary and a
summary can hide a second, smaller discontinuity behind a large one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.schema import CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL
from market_engine.indicators import atr

#: One crore, the unit Indian market commentary quotes turnover in. Reports
#: read as nonsense to anyone familiar with this market in any other unit.
CRORE = 1e7

#: Ratio of annualised-stdev volatility to ATR-derived volatility above which
#: a series is assumed to contain an unadjusted corporate action. Clean names
#: measured ~11; the two known-bad names measured 16 and 28. 14 sits in the
#: gap, and every flagged name is reported rather than silently dropped.
SUSPECT_VOL_RATIO = 14.0

#: Single-bar close-to-close move beyond which a bar is called a discontinuity
#: rather than a rally. No Indian large-cap moves 35% in a session without a
#: corporate action behind it; the daily price band is 20% for most names.
SUSPECT_JUMP = 0.35

TRADING_DAYS = 252.0


@dataclass
class ScreenResult:
    symbol: str
    bars: int
    median_turnover_cr: float
    median_atr_pct: float
    stdev_vol_annual: float
    vol_ratio: float
    max_jump_pct: float
    max_jump_date: str
    #: Populated when the series looks like it contains an unadjusted
    #: corporate action. Never a reason to drop silently - only to report.
    suspect: str = ""

    @property
    def score(self) -> float:
        """Rank on activity and volatility together.

        Turnover spans orders of magnitude and ATR% does not, so turnover is
        compressed logarithmically before the two are combined. Without that,
        the ranking is a turnover ranking with a rounding error attached.
        """
        if self.median_turnover_cr <= 0 or self.median_atr_pct <= 0:
            return 0.0
        return math.log10(self.median_turnover_cr) * self.median_atr_pct


def screen_frame(symbol: str, frame: pd.DataFrame) -> ScreenResult | None:
    """Measure one candidate. Returns None when there is not enough data."""
    df = frame.dropna(subset=[CLOSE_COL])
    if len(df) < 60:
        return None

    close = df[CLOSE_COL].astype(float)
    turnover = (close * df[VOLUME_COL].astype(float)).median() / CRORE

    atr_series = atr(df[HIGH_COL].astype(float), df[LOW_COL].astype(float), close)
    atr_pct = float((atr_series / close).median() * 100.0)

    returns = close.pct_change().dropna()
    stdev_vol = float(returns.std() * math.sqrt(TRADING_DAYS) * 100.0)

    ratio = stdev_vol / atr_pct if atr_pct > 0 else 0.0

    jumps = returns.abs()
    position = int(jumps.values.argmax()) if len(jumps) else 0
    max_jump = float(jumps.iloc[position] * 100.0) if len(jumps) else 0.0
    jump_date = str(df.iloc[position + 1, 0])[:10] if len(jumps) else ""

    reasons = []
    if ratio > SUSPECT_VOL_RATIO:
        reasons.append(f"vol ratio {ratio:.0f}")
    if max_jump > SUSPECT_JUMP * 100:
        reasons.append(f"{max_jump:.0f}% jump on {jump_date}")

    return ScreenResult(
        symbol=symbol,
        bars=len(df),
        median_turnover_cr=round(float(turnover), 1),
        median_atr_pct=round(atr_pct, 2),
        stdev_vol_annual=round(stdev_vol, 1),
        vol_ratio=round(ratio, 1),
        max_jump_pct=round(max_jump, 1),
        max_jump_date=jump_date,
        suspect="; ".join(reasons),
    )


def to_frame(results: list[ScreenResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [asdict(r) | {"score": round(r.score, 3)} for r in results]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def render(frame: pd.DataFrame, min_turnover: float, top: int) -> str:
    """A report a human can act on, not a data dump."""
    if frame.empty:
        return "no candidate returned usable data"

    clean = frame[(frame["suspect"] == "") & (frame["median_turnover_cr"] >= min_turnover)]
    flagged = frame[frame["suspect"] != ""]
    thin = frame[(frame["suspect"] == "") & (frame["median_turnover_cr"] < min_turnover)]

    lines = [
        f"screened {len(frame)} candidates: {len(clean)} clean, "
        f"{len(flagged)} flagged for a corporate action, "
        f"{len(thin)} below the turnover floor of Rs.{min_turnover:.0f}cr",
        "",
        f"{'symbol':<18}{'turnover':>10}{'ATR%':>8}{'score':>8}  notes",
    ]
    for _, row in clean.head(top).iterrows():
        lines.append(
            f"{row['symbol']:<18}{row['median_turnover_cr']:>9.0f}cr"
            f"{row['median_atr_pct']:>8.2f}{row['score']:>8.2f}"
        )

    if len(flagged):
        lines += ["", "flagged - inspect before including, or set a later start:"]
        for _, row in flagged.iterrows():
            lines.append(f"  {row['symbol']:<16} {row['suspect']}")

    return "\n".join(lines)


# --------------------------------------------------------------- gathering


def load_candidates(path: Path) -> dict[str, list[str]]:
    """Read the sector -> tickers mapping."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    candidates = raw.get("candidates", raw)
    return {
        str(sector): [str(sym) for sym in symbols or []]
        for sector, symbols in candidates.items()
    }


def screen_symbols(
    candidates: dict[str, list[str]], period: str = "2y", chunk: int = 25
) -> tuple[list[ScreenResult], list[str]]:
    """Download and measure every candidate.

    Batched and threaded, unlike `ingest.fetch`. The difference is deliberate:
    that function archives raw data and prioritises reproducibility, while this
    one measures candidates to inform a decision and is re-runnable at will.
    Nothing it downloads is kept.
    """
    import yfinance as yf

    symbols = [sym for syms in candidates.values() for sym in syms]
    results: list[ScreenResult] = []
    missing: list[str] = []

    for start in range(0, len(symbols), chunk):
        batch = symbols[start : start + chunk]
        raw = yf.download(
            batch,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        for symbol in batch:
            frame = _extract(raw, symbol, len(batch))
            if frame is None:
                missing.append(symbol)
                continue
            result = screen_frame(symbol, frame)
            if result is None:
                missing.append(symbol)
            else:
                results.append(result)
    return results, missing


def _extract(raw: Any, symbol: str, batch_size: int) -> pd.DataFrame | None:
    """Pull one symbol out of a grouped multi-symbol download."""
    if raw is None or len(raw) == 0:
        return None
    try:
        frame = raw[symbol] if batch_size > 1 else raw
    except KeyError:
        return None
    if frame is None or frame.empty:
        return None

    frame = frame.reset_index()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    rename = {"date": "timestamp", "datetime": "timestamp", "index": "timestamp"}
    frame = frame.rename(columns=rename)
    needed = {CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL}
    if not needed.issubset(frame.columns):
        return None
    if frame[CLOSE_COL].dropna().empty:
        return None
    return frame
