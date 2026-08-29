# Handoff — start here in a new session

Transient working note, not project documentation. Delete once absorbed.

---

## The project in one paragraph

Build a domain-specific small language model that interprets OHLCV market
structure, by distilling a teacher LLM into a fine-tuned SLM and benchmarking
all three (teacher / base SLM / fine-tuned SLM) on a leakage-free evaluation
suite. Scope is **Phases 1–10 only ("MVP 4")**, ending at evaluation. RAG,
agents, quantization, vLLM serving and production deployment are explicitly
**out of scope** — no speculative hooks or abstractions for them.

The project doubles as a hands-on curriculum: the user is an experienced
backend engineer deliberately building AI/ML engineering depth. So for every
significant AI component, explain what it is, why it's needed, the
alternatives, the tradeoffs, and how we'll know it works. Do not hide concepts
behind frameworks.

## Working method (the user cares about this)

One phase at a time. For each: explain architecture → name the concepts being
learned → define acceptance criteria → implement → test → give run commands →
**wait for the user's results** → debug together → only then move on.

- Wants decisions, not menus. When asked "which is better", give the answer
  and the reasoning.
- Values measurement over assertion. Run the probe, report the number.
- **Commits manually. Do not commit unless asked; propose a message.**
- Timeboxing: ~1–3 weeks part-time per phase; flag and cut scope rather than
  let a phase sprawl.

## Status

| Phase | State |
|---|---|
| 1 — Data pipeline & validation gate | committed |
| 2 — Deterministic market engine | committed |
| 3 — Teacher LLM over OpenRouter | committed |
| 4 — Teacher benchmark | **done, run live, not committed** |
| 5 — Distillation corpus | **built and generated, not committed** |
| 6–10 | not started |

`git log`: 4 commits, latest `0ebe203 Added teacher LLM`.
**Everything since is uncommitted — roughly 50 files.** This is the single
biggest risk in the repo right now.

Verified green at handoff: **309 tests pass, ruff clean.**

## Where things actually stand

**Teacher: `mimo` (mimo-v2.5), chosen on evidence.** Full benchmark over 163
frozen examples: quality 0.915, schema compliance 100%, grounded 87%,
**claim-level grounding 99.9% (3,358 of 3,362)**, p50 18.5s, first-attempt
100%. It cleared every gate. `alpha` and `nemotron` were never benchmarked —
a deliberate decision, not an omission: mimo cleared everything and nemotron's
50-calls/day free tier would have taken four days to say so.

**Data: 70 files, 51 symbols** (50 daily + 20 hourly), 142,659 pre-2025 bars,
all passing validation. Expanded from 11 files this session because the corpus
is bounded by bars, not ambition.

**Corpus: 655 accepted of 980 generated** (train 433 / val 222), 49 symbols,
all 20 strata, 2016-01-07 to 2024-12-31. Written to `data/dataset/`, 33MB.
Everything at or after 2025-01-01 remains untouched holdout.

**Student training data: exported and ready.** `msdataset export` renders the
corpus into chat-format pairs under `data/dataset/training/<view>/`, in three
input views (34MB). Nothing further is needed before Phase 6 can start
training - see open question 1 for the view comparison it should run.

## The one number that needs a decision

**The gate rejected 33%, not the 13% predicted.** The cause is arithmetic and
is written up in `docs/dataset.md`: the gate has two independent criteria and
only one was quoted. Grounding fails 14.9% (as benchmarked), contradictions
fail 21.8%, and they compound — 0.851 × 0.782 = 0.665. Measured overlap
matches independence to a third of a percentage point.

Re-gating is **free** (`msdataset regate`) because every answer is
checkpointed. The costed options:

```
any contradiction rejects (as shipped)      655 / 980
only directional errors reject              741 / 980
contradictions ignored entirely             834 / 980
```

The middle option tolerates `SETUP_NOT_OFFERED` (70) and
`EVIDENCE_DOUBLE_COUNTED` (49) — sloppy but not false — while still rejecting
`LEVEL_SIDE_WRONG` (67), which puts support above the price. **This is a real
decision about what the student may imitate and was deliberately left to the
user.** It was not settled by whichever number looked bigger.

## Do this first

```bash
git status                      # ~50 uncommitted files, four phases of work
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/msdataset.exe describe
```

Suggested commits, in order:

1. `Add teacher benchmark harness with model registry and recommendation`
2. `Ground citations by identity rather than list position`
3. `Expand instrument universe from 10 to 51 symbols with a reproducible screen`
4. `Build the distillation corpus with a measured quality gate`
5. `Export the corpus into verified training views`

## Open questions for Phase 6

1. **Does the student need the candle window?** Still open, but now
   *runnable*: `msdataset export` writes three views to
   `data/dataset/training/<view>/{train,val}.jsonl`, so Phase 6 can train
   two of them and measure rather than argue.

   ```
   full      655 pairs   median 13,864 sequence tokens
   tail      655 pairs   median  9,367   <- default
   compact   519 pairs   median  8,220   136 rows dropped
   ```

   `tail` (20 bars) is the default on measurement: every cited candle in the
   corpus sits within the last 10 bars, so it keeps 100% of citations at a
   fifth of the candle cost. `compact` drops 136 rows (20.8%) because their
   analyses cite candles - showing the student a prompt without the evidence
   its target cites would teach fabrication.
2. **655 examples may be thin** for LoRA SFT. Levers: re-gate (above),
   or widen the universe again. The hourly lever is spent — yfinance caps
   that history at ~730 days.
3. **UPTREND is the teacher's weak spot** — 54.5% accepted against 74.2% for
   RANGE, and the benchmark independently found the same thing (79% grounded,
   0.30 contradictions/example). Worth understanding before Phase 8, because
   the student will inherit it.
4. **Which base SLM.** Not started. Sequence length interacts with (1).
5. **The eval suite** (Phases 8–10) should be sampled from the full 51-symbol
   universe. The frozen 163-example benchmark set predates the expansion and
   is drawn from only 10 symbols — fine for what it did, wrong for evaluation.

## Decisions already made — don't re-litigate

- **Causality enforced by truncation invariance**, verified by injecting leaks.
- **Verification-first schema design**: numeric claims sit in structured
  fields naming their source; prose carries no digits.
- **Naming an indicator period is not a numeric claim** ("the 200-day EMA" is
  allowed, "RSI below 55" is not). Derived from the context's own integer
  keys, so it tracks engine parameters.
- **Citations name list elements, they do not count them.** Levels and swings
  carry stable ids; candles are named by timestamp. This took positional
  citations from ~50 defects to zero.
- **2025+ is holdout** and was never sampled.
- **Temporal train/val split**, not random — adjacent examples share a regime.
- **Reject rather than repair** in the corpus gate: a repaired answer is a
  different distribution and would silently misdescribe the corpus.
- **The corpus stores no rendered prompt** — see open question 1.
- **Reasoning models are the wrong shape for this teacher.** The engine
  already did the reasoning deterministically. Measured: nemotron with
  reasoning off was 12x faster and valid first time.
- **Repo is phase-agnostic.** Phase numbering is conversation vocabulary only.

## Live-call findings (each has a regression test)

Written up in `docs/teacher.md` under "What the first live calls taught us"
and "Citations name things; they do not count them".

- `require_parameters` filters on *declared* capability, not real capability.
- A provider can accept `response_format` and ignore it — so the schema
  outline now goes in the prompt in **every** mode.
- Chain-of-thought can arrive inside `content` undelimited; JSON extraction
  scans for balanced objects and prefers the last that validates.
- Reasoning models can spend the whole budget thinking and return empty.
- Timeouts must not be retried.
- `chars/4` underestimated tokens by 43%.
- yfinance returns an unclosed trailing bar during market hours (open, high,
  low and volume present, `close` null). Dropped at ingest, recorded in
  lineage.

## Two mistakes from this session worth not repeating

- **A test that encoded an assumption instead of the data.** The trailing-bar
  fix tested for *all* prices null; the real row has only `close` null. Test
  passed, fix didn't work. Shape fixtures on observed data. It happened a
  second time in the render tests: a fixture wrapped timestamps at 28 days,
  so a "deep" citation also matched a recent bar and the test proved nothing.
- **Predicting a rate from one of two independent gates.** When a report
  lists gates separately, they multiply.
- **Two published numbers were wrong and are now corrected** in
  `docs/dataset.md`: the candle window is 48% of the prompt (not 71% - that
  was its share of the context alone), and the token estimator was 45% low.
  Calibrated against 655 billed calls, this prompt family runs at **1.62
  characters per token**, not the `chars/4 x 1.35` used before.

## Reference

```
data_pipeline/   ingest, contract, validators, screen        (msdata)
market_engine/   indicators, structure, regime, setups       (msengine)
teacher/         provider seam, prompts, schema, grounding   (msteacher)
evaluation/      metrics, sampling, benchmark, recommend     (msbench)
dataset/         quality gate, corpus build, splits, views   (msdataset)
docs/            data_sources, market_engine, teacher,
                 teacher_benchmark, dataset
```

Windows venv: `.venv/Scripts/python.exe`. Run everything from the repo root.
