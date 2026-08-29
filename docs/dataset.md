# The distillation corpus

> Educational and research use only. Not trading advice.

The teacher benchmark answered "is this model good enough to teach?". This
stage asks a different question of every individual answer: **is this one fit
to be learned from?**

The difference in stakes is why the gate exists. In a benchmark a flawed
analysis is a data point - evidence about the model, usefully kept. In a
training corpus a flawed analysis is a *lesson*. An ungrounded number does not
lower a score here; it teaches the student to invent numbers, which is the
single behaviour this whole project is built to prevent.

## What bounds the corpus

Not ambition - bars.

```
pre-2025 bars, 70 files (50 daily + 20 hourly)   142,659
examples at 100-bar separation                     ~1,036 theoretical
examples the stratified sampler actually yields       980
```

Two rules do the bounding. Examples within a symbol must be **at least a full
context window apart**, so no two prompts share a candle; and the first 250
bars of every series are skipped, because the long EMAs and the ATR percentile
are not yet real there and benchmarking on warm-up data measures how a model
handles missing fields rather than how it reads structure.

Relaxing the separation rule would raise the count and lower the value:
overlapping windows are near-duplicates, and a student trained on them learns
to memorise rather than to read. Widening the universe is the only lever that
adds independent examples, which is why the instrument set went from 10 to 51
symbols before generation began. See `docs/data_sources.md`.

## The gate

Each rule rejects a defect the benchmark actually measured, rather than
applying a general notion of quality:

| defect | measured rate | why it disqualifies |
|---|---|---|
| ungrounded or misquoted number | 3 in 219 | teaches fabrication |
| unresolvable citation | 8 in 219 | teaches an address that is a lie |
| digits in prose | 16 in 219 | teaches numbers to escape checking |
| internal contradiction | 26 in 163 | teaches incoherent reasoning |

**Why reject rather than repair.** The runner already knows how to hand errors
back to the model, and that is deliberately not done here. A repaired answer
is drawn from a different distribution than a first answer - the model is
imitating its own correction - and mixing the two silently would make the
corpus something other than what it claims to be. Rejection is cheap; examples
are plentiful and calls are nearly free. What is left can be described
honestly as *answers the teacher got right unaided*.

**Why the gate is all-or-nothing.** A partial-credit rule ("accept if 90% of
claims are grounded") sounds more forgiving and is worse: the student cannot
see which 10% was wrong, so the lesson it draws from a mostly-correct example
includes the incorrect part.

**Rejects are kept**, in `rejected.jsonl`, with reasons. A corpus you cannot
interrogate is one you cannot debug - when the student later fabricates a
number, the first question is whether it learned that from the data, and that
question is unanswerable if the discarded rows were thrown away.

## The split is temporal, not random

A random split leaks. Adjacent examples share a market regime - the same
trend, the same volatility state, often the same news - so a validation
example drawn from the middle of the training period is partly predictable
from its neighbours, and validation loss flatters the model.

A date boundary is also the honest simulation of use: the model will be asked
about bars later than any it trained on. Examples on or after `--val-start`
(default 2024-01-01) are validation; everything earlier is training.

No separation band is needed at the boundary, because the sampler already
guarantees a full window between examples within a symbol, so no training
example's window can reach into a validation one.

Everything at or after **2025-01-01** was never sampled at all. That holdout
belongs to the final evaluation, where teacher, base SLM and fine-tuned SLM
are compared; sampling it here would be training on the test set.

## What a row does not contain

A rendered prompt.

Rows store the engine context and the teacher's analysis separately, and
turning those into a training pair is deferred to `msdataset export`. The
reason is a specific open question: the candle window is **68% of the context
and 48% of the whole prompt**, and the student may not need it. The teacher
needed 100 bars - measured: cited swings fell outside a 30-bar window 33.8% of
the time, 0% at 100 - but the student is handed those swings already computed,
and sequence length is the dominant cost in fine-tuning.

That question is answered by training both variants and measuring, not by
guessing now. Baking a prompt into the corpus would answer it by accident.

## Training views

`msdataset export` renders the corpus into chat-format pairs (system, user,
assistant) under `data/dataset/training/<view>/{train,val}.jsonl`. The target
is the analysis as JSON, so the student is taught to emit the same structured
object the teacher did - which is what makes the two directly comparable at
evaluation.

```
view      pairs                  sequence tokens (median / p95 / max)
full      655  (433/222)         13,864 / 14,445 / 15,224
tail      655  (433/222)          9,367 /  9,841 / 10,285
compact   519  (341/178)          8,220 /  8,677 /  9,041   136 rows dropped
```

**Why `tail` is the default.** Dropping candles entirely is not free: 2.0% of
citations point into `ohlcv_window`, and they are spread across **20.8% of
rows**. A candle-free prompt would leave one row in five citing evidence the
student cannot see - a direct lesson in fabrication. Measured on the corpus,
every cited candle sits within the **last 10 bars**, so a 20-bar tail
preserves 100% of citations with margin at a fifth of the candle cost.

Views are verified, not trusted: `render_pair` re-resolves every citation
against the *trimmed* context and refuses to emit a row whose evidence the
view just removed. That is why `compact` reports 136 dropped rows rather than
quietly shipping them.

### Two numbers corrected here

Both were quoted earlier in this project and both were wrong in a direction
that matters:

* **"The candle window is 71% of the prompt."** It is 68% of the *context*
  but 48% of the prompt - the system message is another 5,656 characters. The
  saving from dropping candles is smaller than that figure implied.
* **The token estimator was 45% low.** `chars/4` is off by a factor of 2.47
  on this content, and the earlier "1.35x" correction came from a single call
  on a shorter context. Calibrated against 655 billed calls, this prompt
  family runs at **1.62 characters per token** (range 1.57-1.68) - JSON
  structure, ISO timestamps and long decimal prices tokenise far worse than
  prose. This is the number that decides whether a sequence fits in a
  training run's context budget, so being 45% low here would surface as an
  out-of-memory failure halfway through an epoch.

## What the first full run produced

980 examples generated with `mimo` (mimo-v2.5), one call each, strict
`JSON_SCHEMA` mode, zero call failures.

```
accepted    655  (66.8%)     train 433 / val 222
rejected    325  (33.2%)     49 symbols, 1d 479 + 1h 176
strata      20 populated, 22-43 each
range       2016-01-07 .. 2024-12-31
```

**The reject rate was predicted at 13% and came in at 33%.** The prediction
was wrong for an arithmetic reason worth recording: the gate has two
independent criteria and only one of them was quoted.

```
fails grounding       146  (14.9%)   <- this is the 13% that was predicted
fails contradiction   214  (21.8%)   <- this was never multiplied in
fails both             35
fails either          325  (33.2%)
independence predicts        33.5%
```

Grounding came in almost exactly as benchmarked. The contradiction gate is a
separate check that the benchmark reported separately, and the two compound:
0.851 x 0.782 = 0.665. The measured overlap (35 rows) matches independence to
within a third of a percentage point, so these really are unrelated failure
modes rather than two views of the same bad answers.

### Where the teacher struggles

| structure | accepted | volatility | accepted |
|---|---|---|---|
| UPTREND | **54.5%** | EXTREME | 62.8% |
| DOWNTREND | 63.3% | HIGH | 65.4% |
| EXPANSION | 70.4% | NORMAL | 67.2% |
| CONTRACTION | 72.2% | LOW | 71.7% |
| RANGE | 74.2% | | |

UPTREND is much the weakest, which the benchmark also found (79% grounded,
0.30 contradictions per example against 0.16 overall). Whatever is wrong is
specific to reading uptrends, and it is now visible in two independent runs.

Two suspicions did **not** survive measurement:

* **Intraday is not harder.** 1h accepted at 65.4% against 1d at 67.4%.
* **The widened universe explains little.** The original ten symbols accepted
  at 70.1%, the 39 additions at 65.7% - four points, not twenty.

### The contradictions themselves

```
SETUP_NOT_OFFERED          70    naming a setup the engine did not offer
LEVEL_SIDE_WRONG           67    calling a level support when it sits above the close
EVIDENCE_DOUBLE_COUNTED    49    the same fact cited twice as independent support
SCENARIO_LEVELS_INVERTED   37    trigger and invalidation the wrong way round
STATE_VS_STRUCTURE         17    stated state contradicts the structure it cites
SETUP_VS_STATE              1
```

These are not equally serious, and the gate currently treats them as if they
were. Re-gating is free - the answers are already paid for and checkpointed -
so the alternatives were costed rather than argued about:

```
any contradiction rejects (as shipped)      655 / 980
only directional errors reject              741 / 980
contradictions ignored entirely             834 / 980
```

The middle option tolerates `SETUP_NOT_OFFERED` and
`EVIDENCE_DOUBLE_COUNTED` - sloppy but not false - while still rejecting an
answer that puts support above the price. It is a real decision about what
the student should be allowed to imitate, and it is deliberately left open
rather than settled by whichever number looked better.

## Files

```
data/dataset/examples.jsonl        the frozen example set (free to rebuild)
data/dataset/results.jsonl         every call, checkpointed; resume-safe
data/dataset/corpus.jsonl          accepted rows, each with its split
data/dataset/rejected.jsonl        refused rows, with reasons
data/dataset/corpus_summary.json   counts, rates, reject breakdown
```

`msdataset regate` re-applies the gate to `results.jsonl` without spending a
call. The gate will be tightened at least once, and re-judging answers already
paid for is preferable to a second generation run that would produce different
answers to the same questions.

## Lineage

Every row records the teacher alias and the model id it resolved to, the
provider that served it, the prompt version and hash, the structured mode,
the temperature, the engine version, the context schema version and the
parameter fingerprint.

This is not decoration. A corpus is a claim about how it was made, and a row
that cannot name what produced it cannot be reproduced, compared against a
later vintage, or selectively withdrawn when one of those inputs changes.
`msdataset describe` warns loudly if a single corpus contains more than one
prompt hash, because rows produced by different prompts are not one corpus.
