# market-structure-slm

Domain-specific SLM for financial market-structure analysis — teacher-LLM
distillation, LoRA fine-tuning, and evaluation vs. base/teacher models.

> ## ⚠️ Educational and research use only
>
> Nothing produced by this repository — code, features, model output, reports,
> or demos — is financial or trading advice. The models here are trained to
> *describe* market structure, not to predict prices or recommend positions.
> Language models fabricate confident, plausible, wrong statements; that
> failure mode is what this project measures, not something it eliminates.
> Do not trade on any of it.

## Setup

```bash
py -3 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python` throughout.

## Data pipeline

Fetch OHLCV into `data/raw/` — immutable, with one `.lineage.json` sidecar per
file recording source, symbol, interval, requested range, `yfinance` version,
fetch time, and the file's sha256:

```bash
.venv/Scripts/msdata.exe fetch
```

Run the validation gate and write reports to `data/reports/`:

```bash
.venv/Scripts/msdata.exe validate -v
```

Tests:

```bash
.venv/Scripts/python.exe -m pytest -q
```

### The gate

Ten independent checks run over every file. Nothing is ever silently repaired —
there is no `fillna`, `dropna`, `drop_duplicates`, or `sort_values` in the
load-or-validate path, and a test enforces that by scanning the source. A
loader that sorted the frame would make the ordering check unfireable.

| check | worst severity | catches |
|---|---|---|
| `required_columns` | ERROR | a contracted column is absent |
| `dtypes` | ERROR | text in a price column, unparseable timestamps |
| `missing_values` | ERROR (volume: WARNING) | nulls and unparseable cells |
| `chronological_order` | ERROR | rows not oldest→newest |
| `duplicates` | ERROR | repeated timestamps or identical rows |
| `timezone_consistency` | ERROR (naive: WARNING) | mixed offsets, naive/aware mixtures |
| `ohlc_relationships` | ERROR | `high < low`, `high < open`, `low > close`, … |
| `value_domains` | ERROR | non-positive prices, negative volume |
| `missing_candles` | WARNING | unexplained holes in the timestamp grid |
| `abnormal_gaps` | WARNING | outlier overnight gaps and single-bar ranges |

**ERROR** means physically impossible — a real market cannot print `high < low`,
so the source or transport is broken. **WARNING** means unusual but genuinely
possible: feed holes, crash days, zero-volume index series. Verdict is `FAIL`
on any error, `PASS_WITH_WARNINGS` on any warning, else `PASS`. Exit code is
`1` on `FAIL` and `0` otherwise, so the command can gate CI.

The last two checks are heuristics — deciding them properly needs an exchange
trading calendar, which the pipeline does not carry. Their thresholds live in
`configs/data.yaml` and the rationale is in the docstrings of
[validators.py](data_pipeline/validators.py).

**No data is committed.** `data/` is gitignored — see
[docs/data_sources.md](docs/data_sources.md) for the licensing reasons, the
adjustment semantics, and this source's known quality quirks. Reproduce by
re-fetching.

## Market engine

Turns validated candles into a versioned `MarketContext` JSON object:
indicators, swing structure (HH/HL/LH/LL), BOS and CHoCH, support and
resistance, volatility and trend regime, higher-timeframe context, and
rule-based candidate setups.

Human-readable summary of what the engine sees in each file:

```bash
.venv/Scripts/msengine.exe summary
```

Emit a context for the most recent bar:

```bash
.venv/Scripts/msengine.exe context data/raw/SPY_1d.csv --bar -1
```

Regenerate the data dictionary from the schema:

```bash
.venv/Scripts/msengine.exe schema
```

The engine refuses to run on a file whose validation verdict is FAIL. Features
computed from impossible candles look plausible and are undetectable by the
time anyone notices the model is wrong.

### Causality

Every value in a context for bar `t` is a function of bars `0..t` only, and
that is enforced by a **truncation-invariance test**: computing the context
with the full series and with every later bar deleted must give an identical
result. The test was checked against two deliberately injected look-ahead
leaks and caught both.

Swings carry both the bar they printed on and the bar they became *knowable*
on (`index + swing_lookback`); nothing references a swing before it confirms.
The higher-timeframe view exposes only fully closed periods, so it is up to
one period stale by design.

- [docs/market_engine.md](docs/market_engine.md) — definitions, the BOS/CHoCH
  distinction, calibration against real data, payload cost.
- [docs/market_context.md](docs/market_context.md) — field-by-field data
  dictionary, generated from the schema.

## Layout

```
configs/         YAML config (thresholds, symbols). No secrets, ever.
data_pipeline/   Ingest, schema contract, validators, reports.
market_engine/   Deterministic indicators, structure, regime, setups.
teacher/         Teacher LLM over OpenRouter, prompts, generation.
datasets/        Quality pipeline and chronological splits.
training/        Base model baseline and LoRA fine-tuning.
evaluation/      Benchmark suite.
docs/            Design notes.
tests/
```

## Secrets

Never commit API keys. Copy `.env.example` to `.env` (gitignored) before
running anything that talks to a model provider.
