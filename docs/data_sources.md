# Data sources, licensing, and usage restrictions

> Educational/research use only. Nothing in this repository is trading advice.

## Source: Yahoo Finance via `yfinance`

All OHLCV data in this project is fetched with the open-source
[`yfinance`](https://github.com/ranaroussi/yfinance) library, which scrapes
Yahoo Finance's public endpoints.

### Licensing position

| Item | Status |
|---|---|
| `yfinance` library | Apache 2.0 — freely usable, including commercially. |
| The **data** it returns | Owned by Yahoo and its upstream exchange/vendor licensors. **Not** covered by the library's licence. |
| Yahoo's terms | Yahoo's Terms of Service permit personal, non-commercial use. `yfinance` is not an official or supported API and Yahoo may change or block it at any time. |
| Redistribution | **We do not redistribute the data.** `data/` is gitignored. Anyone reproducing this project fetches their own copy with `msdata fetch`. |

### Consequences for this repository

1. **No market data is committed.** The `.gitignore` excludes `data/`. Only
   code, configs, and reports-about-data may be committed, and reports must not
   embed enough rows to reconstitute a series (they carry at most 10 sample
   timestamps per finding, no prices).
2. **Non-commercial, research use only.** If this project were ever to become
   commercial, the data source must be replaced with a licensed feed
   (Polygon.io, Databento, Nasdaq Data Link, or a broker API under its own
   terms) before that happens.
3. **Reproducibility is by re-fetch, not by redistribution.** Every raw file
   has a `.lineage.json` sidecar recording symbol, interval, requested range,
   `yfinance` version, fetch timestamp, and the file's sha256, so a
   re-fetch can be compared against the original bytes.

### Known data-quality characteristics of this source

These are why the validation gate exists at all:

- **History depth limits.** 1-minute bars: ~7 days. Other intraday intervals:
  ~60 days. Hourly: ~730 days. Daily: decades.
- **Adjustment semantics (verified, not assumed).** `auto_adjust=False`
  disables *dividend* adjustment only. yfinance back-adjusts **splits**
  regardless — checked against AAPL's 2020-08-31 4:1 split, which shows no
  discontinuity in the fetched series. Stored data is therefore
  **split-adjusted, dividend-unadjusted**.
  - Good: the series is continuous, so market structure is not broken by an
    artificial 75% gap at every split.
  - Bad: prices are not what was printed at the time, and a *future* split
    retroactively rewrites the whole history, changing the file's sha256.
    Reproducibility therefore rests on keeping the fetched file, not on
    assuming a re-fetch returns identical bytes.
  - The `ABNORMAL_GAP` warnings that remain are consequently real market
    events, not corporate actions — on AAPL they are the COVID crash
    (2020-03-16, -13%), the 2015-08-24 flash crash, the 2024-08-05 yen-carry
    unwind, and the 2019-01-03 guidance cut. That is a useful sanity signal
    that the check is calibrated sensibly.
- **Missing bars.** Illiquid instruments and some sessions simply have holes.
  Flagged as `MISSING_CANDLES` (WARNING), never repaired.
- **`^VIX` and similar indices report zero volume.** Flagged as
  `ZERO_VOLUME_RUN` (WARNING). Legitimate; the market engine must treat
  volume as optional for these symbols.
- **Timezones.** Intraday bars are timezone-aware in the exchange timezone and
  are converted to UTC on ingest. Daily bars come back as naive dates and are
  *labelled* UTC without shifting; the sidecar records this.
- **Calendars disagree between symbols.** `^VIX` carries a bar dated
  2026-05-25 (US Memorial Day) that no equity symbol has — 2927 rows against
  2926. Per-file validation structurally cannot see this. Any cross-symbol or
  multi-timeframe alignment must either inner-join on a reference symbol's
  session calendar (simple, drops the orphan bar) or carry a real exchange
  calendar (correct, more machinery).

## Adding another source later

Any new source must supply, before first use: the licence or terms under which
the data may be used, whether redistribution is permitted, the timezone
semantics of its timestamps, and whether prices are adjusted. Add it as a
section in this file, then implement it behind the same
`FetchSpec -> raw CSV + lineage sidecar` interface in `data_pipeline/ingest.py`.
