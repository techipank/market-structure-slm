# Teacher model benchmark — design notes

> Educational and research use only. Not trading advice.

Choosing the teacher is the highest-leverage decision in the project: every
label in the training set comes from it, and a flaw here is baked into the
student permanently. So it is run as an experiment, not settled by preference.

## Same inputs, or it measures nothing

The example set is built once, written to `samples.jsonl`, and every model
reads that file. Comparing models on separately-sampled data measures the
sampler.

## Sampling: three ways a benchmark quietly lies

**Near-duplicates.** Adjacent bars share 59 of 60 candles and nearly all their
derived features. Two hundred randomly chosen bars from one symbol can be two hundred views
of a handful of situations — a benchmark that measures one market condition
very precisely and everything else not at all. Selected bars from one source
are forced at least `min_separation` bars apart, defaulting to the candle
window so no two prompts share a candle.

**Unbalanced coverage.** Selection round-robins across
(structure state × volatility regime), then across symbols within each
stratum. Round-robin rather than proportional: the point is to find where a
model breaks, and rare states are where that happens. Proportional sampling
fills the set with common cases and leaves two examples of the interesting ones.

Getting this right took two fixes, both worth knowing about because both
produced output that *looked* fine:

1. Strata were iterated in sorted **name** order. Because the separation
   budget is shared per symbol — taking a bar blocks its neighbours for every
   stratum, not just the one that took it — whichever strata were served first
   consumed it. The result was a monotonic decline from `CONTRACTION` (11–12
   examples) to `UPTREND` (1–7), purely because "U" sorts last. Strata are now
   ordered scarcest-first, which is the standard fix: common states have
   candidates to spare, rare ones do not.
2. The inner loop covered strata × sources within a single pass, so one pass
   could take 20 × 10 = 200 examples — the whole target — and the round-robin
   never actually rotated. It now takes at most one example per stratum per
   pass, with the source rotating inside each stratum.

Either defect silently under-samples entire market states, which surfaces much
later as "the model is bad at uptrends" and gets blamed on the model. A test
asserts the spread across strata stays within two examples.

**Holdout contamination.** Choosing a teacher on examples from the eventual
test period is model selection against the test set — the same mistake as
tuning a prompt on it, and forbidden by the project's own leakage rule. The
sampler takes a hard date cutoff, defaulting to `2025-01-01`, so validation and
test eras are untouched.

Result on the current data:

```
163 examples · 10 NSE symbols · 2016-01-07 .. 2024-12-31
all 20 (structure × volatility) strata populated, 8–9 examples each
```

It asked for 200 and got 163: the separation rule refused the rest rather than
emit overlapping prompts. Coverage beats a round number. `ETERNAL_NS` and
`MAXHEALTH_NS` contribute fewest because they listed most recently; the
round-robin does not pad them with overlapping windows to compensate.

The universe grew from six symbols to ten specifically to support the 100-bar
candle window. At that window, non-overlapping prompts need 100 bars of
separation, and six symbols could only supply 106 examples with some strata
down to a single case. Widening the universe was the fix; narrowing the window
or relaxing separation would each have traded one defect for another.

The sampler reads the two stratification axes straight off the computed series
instead of materialising a context for every candidate bar. Building ~15,000
contexts to look at two fields each made the test suite take 154 seconds; it
now takes 6.

## Metrics: three families that must not be averaged

**Schema compliance** — did it produce the right shape, and in how many
attempts? `first_attempt_rate` is tracked separately from
`schema_compliance_rate`, because "valid after two repairs" and "valid first
time" are different models wearing the same score.

**Agreement with the deterministic engine** — only where the engine is
genuinely authoritative.

| field | engine value | status of disagreement |
|---|---|---|
| `market_structure` | `structure.trend` | error — it is a definition applied to confirmed swings |
| `higher_timeframe_bias` | `higher_timeframe.bias`, else `UNAVAILABLE` | error — inventing a bias for an absent timeframe is fabrication |
| `setup_type` | names in `setups` | *disagreement*, not error — the engine's rules are narrow |

Only the first two are error. `setup_agreement_rate` is reported as an
agreement rate and never relabelled accuracy.

**Internal contradictions** — checkable with no ground truth at all. A model
can be schema-valid and perfectly grounded while still saying something that
cannot be true of itself:

| contradiction | what it catches |
|---|---|
| `STATE_VS_STRUCTURE` | `TRENDING_UP` inside a `DOWNTREND` |
| `SETUP_VS_STATE` | a long setup in a market it just called falling |
| `SETUP_NOT_OFFERED` | naming a pattern whose conditions were not met |
| `LEVEL_SIDE_WRONG` | support above the current price |
| `SCENARIOS_ONE_SIDED` | two scenarios both pointing the same way |
| `SCENARIO_LEVELS_INVERTED` | a bullish path invalidated above its own trigger |
| `EVIDENCE_DOUBLE_COUNTED` | the same fact cited for and against |
| `CONFIDENCE_VS_EVIDENCE` | high confidence against its own counter-case |
| `HTF_CLAIMED_WHEN_ABSENT` | a higher-timeframe read with no higher-timeframe bar |

`SCENARIOS_ONE_SIDED` exists because the schema can require two scenarios but
not two *different* ones — a good illustration of where structured output
stops and semantics begin.

**Self-consistency** — the same example run three times, on an evenly strided
subset. This matters more than it looks: a model that answers differently each
time cannot be a teacher however good any single answer is, because training a
student on it distils noise. The subset is strided rather than taken from the
head, since the sample list is sorted by symbol and the head would measure one
instrument in one era.

### Hallucination is counted narrowly

`hallucinated_numbers` = invented prices + misquoted values. A *correct* number
cited at the *wrong* path is counted separately as `unresolvable_fields`.
Sloppy citation and fabrication are different failures, and merging them would
overstate how often the model invents things.

## Not choosing the biggest model

Quality is an explicit weighted sum, all components already 0–1:

| component | weight | why |
|---|---|---|
| grounded rate | 0.25 | the largest source of poisoned labels |
| schema compliance | 0.20 | unusable output is worth nothing whatever it says |
| fact agreement | 0.20 | disagreeing with a deterministic fact is error |
| no contradictions | 0.15 | inconsistent reasoning teaches nonsense |
| self consistency | 0.15 | an inconsistent teacher distils noise |
| valid first attempt | 0.05 | convenience and cost, not correctness |

Grounding outweighs compliance because **a schema-valid analysis full of
invented prices is worse than a malformed one** — the malformed one gets
rejected, the invented one gets trained on.

**Gates are applied before scoring and are not tradeable**: schema compliance
≥ 90%, grounded ≥ 80%, contradictions ≤ 1/example, p95 latency ≤ 120s. Without
gates a weighted mean would let a fast cheap model buy its way past a 60%
hallucination rate.

**The recommendation rule**: the *cheapest* model within 0.03 quality of the
best eligible one. Three points on a 163-example benchmark is inside the
sampling error, so paying ten times more for it buys noise. If nothing clears
the gates the report says so and refuses to recommend anything — the correct
output when the honest answer is "do not start generating".

Projected dataset cost divides by the success rate, so a model that fails one
call in ten is charged for the retries it will actually need.

## Resumability

163 examples × 4 models ≈ 650 paid calls. Results append to a JSONL checkpoint
as they land and a re-run skips completed `(model, example, run)` keys.

Append-only JSONL rather than a JSON array: a single append is close enough to
atomic, and a half-written final line costs one example instead of the whole
file. The checkpoint reader tolerates a truncated last line for exactly that
reason.

A failure inside one call is recorded as a result, not raised — one model
refusing one example must not abort the rest. Because the failure is
checkpointed it also is not silently retried forever on resume; it shows up in
the report instead.

Model/example pairs are interleaved model-major so an interrupted run leaves
partial coverage of *every* model rather than complete coverage of some.

## Order of operations

```
msbench sample            free    build and inspect the frozen set
msbench smoke --model X   1 call  prove one model works end to end
msbench run --model ...   paid    fan out
msbench report            free    aggregate, score, recommend
```

`smoke` exists because the expensive way to find a broken request shape is
eight hundred calls into a fan-out.

`msbench run` disables the structured-mode fallback by default. A model that
cannot honour a strict schema should show up as failing rather than be quietly
handed an easier task than its competitors.
