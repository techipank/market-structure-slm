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

### The instrument universe

**70 files: 50 NSE (India) equities at 1d, and 20 of them again at 1h.**

The set was chosen **empirically, not by reputation**, and the screen is now
code rather than a one-off analysis - `msdata screen` measures a candidate
pool and writes `data/reports/screen.csv`, so the universe can be re-derived
and defended later. It reproduced the original manual screen independently:
INFY at Rs.1,150cr / 2.22% against the Rs.1,141cr / 2.23% recorded below, and
it flagged both known corporate actions unprompted.

The first six (detailed in the table below) were picked by hand from that
method. The universe was later widened to 50 for a specific reason: the
distillation corpus is bounded by *bars*, and ten symbols supplied only
23,706 pre-cutoff bars, capping the corpus at ~237 non-overlapping examples.
Widening to 50 raised that to 103,639 bars, and the hourly tranche to 142,659.
See `docs/dataset.md`.

The expansion rule, applied to 186 candidates: clear of the corporate-action
detector, median turnover at or above Rs.100cr, ranked on
`log(turnover) x median ATR%`, and **capped at three per sector** so the set
cannot collapse into one correlated bloc. The hourly tranche was ranked on
turnover alone - volatility is not the scarce quality at that resolution,
liquidity is, because a thin name's hourly bars are mostly gaps and noise.

| symbol | sector | median turnover | median ATR% | why it is in the set |
|---|---|---|---|---|
| `ETERNAL.NS` | consumer tech | ₹979 cr | 3.40% | highest combined activity+volatility score |
| `INFY.NS` | IT services | ₹1,141 cr | 2.23% | second-highest turnover in the index; the liquid anchor |
| `M&M.NS` | automotive | ₹765 cr | 2.51% | high on both axes |
| `INDUSINDBK.NS` | private bank | ₹323 cr | 2.56% | the 2025 accounting disclosure supplies a real sustained downtrend |
| `HINDALCO.NS` | metals | ₹382 cr | 2.47% | a commodity cycle, structurally unlike the rest |
| `ADANIENT.NS` | infrastructure | ₹273 cr | 2.84% | repeated event-driven extremes populate the EXTREME volatility stratum |

The universe was widened twice more. The third tranche took 40 names; the
fourth took a further 79, drawn from the **same screen run** - they were the
candidates that were clean and above the turnover floor but had been cut by
the earlier per-sector cap, so no new measurement was needed. The cap is now
applied to the whole universe rather than per tranche, existing members
included, so no sector can dominate as the set grows.

The result is **129 daily symbols and 20 hourly files**, 322,703 pre-cutoff
bars, supporting 2,223 non-overlapping examples - against 237 when the set
was ten symbols.

Additions are listed in `configs/data.yaml` with the screen's numbers beside
each, and 20 of the most liquid names are fetched again at 1h.
yfinance caps hourly history at ~730 days, so those contribute roughly 2,100
pre-cutoff bars each rather than a decade - and two of them (`ETERNAL.NS`,
which appears to have hourly history only from the Zomato rename, and
`SWIGGY.NS`, listed in late 2024) contribute nothing before the cutoff at all.
They are kept because they still serve the post-cutoff holdout.

### Detecting corporate actions over full history

The screen sees two years. A split in 2018 would pass it unnoticed, so the
archived series are checked again after fetching. Five symbols showed
single-day moves above 35%, and all five turned out to be genuine:

| symbol | date | move | verdict |
|---|---|---|---|
| INDUSINDBK | 2020-03-26 | +44.7% | 20 of 40 symbols moved >5% - COVID rebound |
| BANDHANBNK | 2020-03-26 | +39.3% | same day, same cause |
| CANBK | 2017-10-25 | +38.7% | moved with UNIONBANK - PSU bank recapitalisation |
| UNIONBANK | 2017-10-25 | +34.2% | same day, same cause |
| IDEA | 2020-02-19 | +40.0% | isolated, but 5.5x median volume |
| ZEEL | 2021-09-14 | +40.0% | isolated, but 33.7x median volume |
| PNB | 2017-10-25 | +46.2% | fourth bank on the recapitalisation day |
| **CDSL** | **2017-07-03** | **-49.9%** | **not genuine - see below** |

`CDSL.NS` is the one that failed the test, and it is instructive. Its listing
bar (2017-06-30) prints at exactly twice the scale of everything after it -
261.75 -> 131.07, a ratio of 0.5008 - and the following bar trades a normal
8.2% range. A real 50% crash does not gap and then behave; an unadjusted
action or a bad first print does. It is therefore fetched from **2017-07-03**,
the same treatment as ADANIENT's demerger and for the same reason: the bar
itself sits inside the indicator warm-up and would never be sampled, but an
EMA-200 spanning it blends two price scales for two hundred bars afterwards.

The discriminator is worth keeping: **a corporate action is idiosyncratic; a
market event moves the market.** For the two isolated cases, volume and bar
shape settled it - a real move trades *through* its range, while an
unadjusted action gaps and then prints a normal-sized bar.

**Volatility was ranked on median ATR%, not on standard deviation of returns.**
Two otherwise-qualifying candidates were rejected because of this distinction:

- **VEDL** ranked *first* on annualised stdev at 81.8% — almost entirely from
  one bar: 773.60 → 289.50 on 2026-04-30, the Vedanta demerger, unadjusted.
  Its median ATR% is an unremarkable 2.89%.
- **TRENT** ranked second at 46.9%, driven by 4279.00 → 2852.67 on 2026-01-01.
  That ratio is exactly 1.5 — an unadjusted 3:2 split or bonus issue.

The ratio of stdev-vol to ATR-vol turned out to be a serviceable
corporate-action detector: ~11 for a clean name, 16 for TRENT, 28 for VEDL.

### Known data-quality characteristics of this source

These are why the validation gate exists at all:

- **History depth limits.** 1-minute bars: ~7 days. Other intraday intervals:
  ~60 days. Hourly: ~730 days (the same on NSE as on US markets). Daily:
  decades, subject to listing date — `ETERNAL.NS` only listed in July 2021.
- **Corporate-action handling is inconsistent between markets, and this is the
  single most dangerous quirk here.** On US data, yfinance back-adjusts splits
  cleanly — AAPL's 2020 4:1 split shows no discontinuity. On NSE data it
  frequently does not: verified unadjusted discontinuities include the
  Vedanta demerger, a Trent 3:2 split, and Adani Enterprises' 2015 demerger
  (97.64 → 312.39, a 220% jump). Do not assume adjustment; check.
  - `ADANIENT.NS` is therefore fetched from **2015-07-01**, after its
    demerger, rather than the global 2015-01-01 start. The bar itself sits
    inside the indicator warm-up and would never be sampled, but an EMA-200
    spanning it blends two different companies for roughly two hundred bars
    afterwards. Five months of history is a cheap price for removing that.
  - The `ABNORMAL_GAP` check is what surfaces these. It is not a nuisance
    warning; treat a large one as "investigate before training on this".
- **Circuit limits are an India-specific feature, not a data error.** NSE
  applies ±10%/±20% daily price bands to individual stocks. Gaps of *exactly*
  10.0% recur across INFY, INDUSINDBK and HINDALCO — those are opens at the
  circuit, and they are real market behaviour the model should learn, not
  artefacts to clean away.
- **NSE observes roughly 50% more holidays than US equity markets.** Measured
  across the daily files: 154 gaps of 2–3 business days, against 103 for the
  previous US universe. The observed maximum is 3 business days and there are
  no gaps longer than 4, so `max_holiday_business_days` is set to 4 — one day
  of headroom over anything present in eleven years of data.
- **Missing bars.** Illiquid instruments and some sessions simply have holes.
  Flagged as `MISSING_CANDLES` (WARNING), never repaired.
- **Instruments without volume.** Index series (`^VIX`, `^INDIAVIX`, `^NSEI`)
  report a constant zero. Flagged as `ZERO_VOLUME_RUN` (WARNING); the market
  engine treats volume as optional rather than fabricating it. No such symbol
  is in the current universe, so this path is covered by tests rather than by
  live data.
- **Timezones.** NSE intraday bars arrive tz-aware in `Asia/Kolkata` and are
  converted to UTC on ingest — so an hourly bar stamped 09:15 IST is stored as
  03:45 UTC. Daily bars come back naive and are *labelled* UTC without
  shifting, since shifting an exchange-local date would move it onto the wrong
  calendar day. The sidecar records which happened.
- **Prices are in INR.** Nothing in the pipeline is currency-aware and nothing
  needs to be: every derived feature is a ratio, a percentage, or an ATR
  multiple. The only currency-sensitive values are the raw prices themselves,
  which are only ever compared against other prices in the same series.

## Adding another source later

Any new source must supply, before first use: the licence or terms under which
the data may be used, whether redistribution is permitted, the timezone
semantics of its timestamps, and whether prices are adjusted. Add it as a
section in this file, then implement it behind the same
`FetchSpec -> raw CSV + lineage sidecar` interface in `data_pipeline/ingest.py`.
