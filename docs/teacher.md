# Teacher LLM — design notes

> Model-generated interpretation of historical market structure, for
> educational and research purposes only. Not trading advice, not a
> prediction, and not a recommendation to buy or sell anything.

The teacher turns a deterministic `MarketContext` into a structured analysis.
It is the only non-deterministic component in the pipeline, and it is treated
accordingly: nothing it says is trusted, everything it says is recorded, and
every number it states is checked against the context it was given.

## Structured outputs: four mechanisms, not one

These are routinely conflated, and the difference decides what can go wrong.

| mechanism | what is enforced | how it fails |
|---|---|---|
| "reply with JSON" in the prompt | nothing | prose, markdown fences, trailing commas |
| **JSON mode** (`response_format: json_object`) | output is valid JSON | valid JSON, entirely wrong fields |
| **JSON Schema mode** (`strict: true`) | output matches your schema | not supported by every model or provider |
| tool / function calling | schema as a function signature | roundabout, but the widest support |

We use JSON Schema mode with automatic fallback to JSON mode and then to
nothing, appending a schema outline to the prompt once the decoder stops
enforcing it. **Which mode succeeded is recorded on every result**, because a
model that needed the fallback is not competing on equal terms with one that
did not, and comparing their compliance rates without noting that would be
misleading.

The critical thing constrained decoding does *not* do is make the output true.
A schema-valid analysis can cite support at a price that appears nowhere in
the data. Shape and grounding are independent problems.

## Three layers of enforcement

The schema sent to the model is deliberately **weaker** than the schema we
validate against.

```
 decoder    →  shape only: field names, types, enum members
 pydantic   →  everything the decoder cannot express:
               ≥2 supporting evidence, ≥1 conflicting, ≥2 scenarios,
               confidence within 0..1
 repair loop→  hands pydantic's complaints back to the model and retries
```

This split is forced by reality: strict modes *reject* `minItems`,
`minLength`, `minimum`, `pattern` and `default` rather than ignoring them, so
`jsonschema.py` strips them from the generation schema. Enforcing them in
pydantic and feeding failures back is what closes the gap.

`$defs`/`$ref` support is **not documented** by OpenRouter and varies by
provider, so the schema is dereferenced into a self-contained tree before it
is sent. That costs a few hundred tokens per request and avoids discovering
the limitation halfway through a paid run.

## Designing the schema so verification is cheap

This is the highest-leverage decision in the phase.

The lazy design lets the model write prose full of prices and then hunts for
hallucinations with a regex afterwards. That is guesswork: `452.10` in a
sentence might be a level, a target, a typo, or an invention, and no pattern
matching recovers intent.

Instead the schema forces every numeric claim into a structured field that
names its source:

```json
{"price": 739.51, "role": "SUPPORT", "context_field": "levels[0].price",
 "note": "A well-tested floor below price."}
```

Verification becomes an exact lookup. And because numbers now have a place to
live, **prose fields are required to contain no digits at all** — a far
stronger and cheaper rule than "prose numbers must be grounded", since
checking whether a sentence contains a digit needs no knowledge of intent.

### What the grounding checker reports

| finding | meaning |
|---|---|
| `UNRESOLVABLE_FIELD` | the cited path does not exist in the context |
| `VALUE_MISMATCH` | the path exists but holds a different value |
| `UNGROUNDED_PRICE` | the price appears nowhere in the context at all |
| `NUMBER_IN_PROSE` | a digit in a field that must be qualitative |

A real price cited at the wrong path is reported as `UNRESOLVABLE_FIELD`, not
`UNGROUNDED_PRICE` — sloppy citation and fabrication are different failures
and lumping them together would misstate how often the model actually invents
things.

Verified end to end against a real market context: a well-formed analysis scores
5/5 grounded with no issues, and three separately injected faults are each
caught by exactly the intended finding.

**This is a grounding check, not a correctness check.** It proves the teacher
copied a number that exists. It says nothing about whether citing that number
supports the claim being made. Semantic agreement is a separate problem for
the dataset quality pipeline.

### Known limitation

`Evidence.value` is a string, so a citation pointing at a list or object field
cannot be compared exactly; a non-empty container is treated as agreeing. A
model could hide a vague claim behind a container-valued path. Tightening this
means either typing `value` as a union (which strict mode handles poorly) or
restricting citations to scalar paths.

## Prompt engineering choices

Prompts are files under `teacher/prompts/<version>/`, hashed onto every
result. A prompt change is a semantic change to the dataset, and an edited
string literal in code leaves no trace that survives a `git log --oneline`.

**Positive constraints over negative ones.** "Do not invent prices" names a
forbidden behaviour without giving a procedure. The prompt instead says: every
number must be copied from a named field, and you must name the field. That is
something the model can *do*, and it lines up exactly with what the schema
requires and the verifier checks — instruction, structure, and enforcement all
saying the same thing.

**The vocabulary is defined in the prompt.** BOS and CHoCH have contested
meanings in trading material; the system prompt gives the exact definitions the
engine uses, including the fact that they are the same geometry distinguished
only by prevailing bias. Without this the model scores badly against the engine
for using a different convention rather than for being wrong.

**`conflicting_evidence` is a required field.** Models are agreeable; asked for
an analysis they produce a one-sided case. Making the counter-argument
structurally mandatory is far more reliable than politely asking for balance.
The same reasoning makes `setup_type: NONE` explicitly blessed in the prompt
as a common and correct answer.

**Message layout is a cost decision.** The system prompt is identical for every
example, so providers that cache prompt prefixes can reuse it; the per-example
context goes in the user message. Putting the context in the system message
would defeat caching at dataset scale.

## Cost per example

Measured on a real INFY.NS daily context:

| component | chars | ≈ tokens | notes |
|---|---|---|---|
| system prompt | 4,849 | 1,212 | identical every call; cacheable |
| user message (context) | 9,592 | 2,398 | varies per example |
| `response_format` schema | 6,334 | 1,583 | sent every call |
| **total input** | | **≈ 5,200** | |

The context was originally pretty-printed, which cost **1,285 extra tokens per
example — 37% of the payload — to transmit whitespace**. Over a
5,000-example run that is more than six million tokens spent on indentation.
It is now compact.

Remaining levers, in order of size: the response schema (1,583 tokens on every
call, unavoidable in strict mode), the candle window (`ohlcv_window_bars`,
halving it saves roughly 1,000 tokens), and prompt caching where the provider
supports it.

## What the first live calls taught us

Everything below was found by actually calling a model, and each one had
shipped as a bug. They are recorded because the same mistakes are easy to make
again, and because several contradict what this document originally claimed.

**`require_parameters` filters on declared capability, not real capability.**
This document argued the flag turns invisible degradation into a loud failure.
True, but incomplete: it also produces **false negatives**. OpenRouter refused
to route a strict `json_schema` request to `stealth/ox-alpha` because its
provider does not advertise structured outputs — yet the identical request
succeeds and returns valid schema-conforming JSON once the flag is dropped.

```
json_schema strict + require_parameters   404  No endpoints found...
json_schema strict, no require            200  {"ok":true}
```

The harness now relaxes the flag once on that specific rejection and records
the concession on the result. Dropping it is safe here in a way it would not
be in general: `response_format` is still sent, so a provider that silently
ignores it produces output that fails pydantic validation and the repair loop
catches it. The gate was guarding against a failure we already detect
downstream, at the price of rejecting models that work.

The rejection also arrives as a **404 with different wording** from the
documented 503 phrasing, which is why the first classifier missed it.

**A weaker structured mode can be worse than a stronger one.** The fallback
ladder assumed `JSON_SCHEMA → JSON_OBJECT → NONE` is monotonically more
permissive. On `ox-alpha`, `json_object` returns an *empty* body while
`json_schema` works. Empty output under a structured request is therefore
treated as a capability failure that steps the mode down, not a transient
blip that retries the same doomed call.

**Reasoning models can spend the entire output budget thinking.** At
`max_tokens: 4096`, `ox-alpha` produced `finish_reason: "length"`, zero
characters of content, and 12,032 characters of reasoning. There is now a
distinct `Truncated` error that names the remedy, because neither retrying
nor stepping down the mode addresses it. Note the provider's own
`completion_tokens_details.reasoning_tokens` reported **0** for that call, so
that field cannot be trusted to detect the condition.

**Timeouts must not be retried.** A request that was too slow will be too slow
again. Four retries at a 120s deadline is eight wasted minutes before the mode
fallback even starts — which is how one smoke call consumed fifteen minutes
and returned nothing.

**The token estimate was 43% low.** `chars/4` predicted 6,117 input tokens
where the provider billed 8,752. JSON punctuation and ISO timestamps tokenise
far worse than prose. `msteacher dry-run` now prints both the floor and a
calibrated figure, and any real cost projection should use measured
`prompt_tokens` from a smoke call instead.

**A provider can accept `response_format` and ignore it, and a model that was
never told the field names will invent its own.** `nvidia/nemotron-3.5-lightning`
is served by a provider that does not declare structured outputs, so every
request trips the `require_parameters` gate and is retried with routing
relaxed. The request then succeeds, the schema is discarded, and the model —
which in strict mode was deliberately *not* sent the schema outline, since the
decoder was supposed to enforce it — reasoned in its own output about not
having been given one, then produced a clean JSON object made entirely of keys
we do not have. The outline is now sent in **every** mode. It costs about 900
prompt tokens, roughly 8% of a request, and it is the difference between a
recoverable answer and an unparseable one whenever the constraint is dropped.

**The chain-of-thought arrives inside `content`, undelimited.** The same model
emits reasoning and answer in one field with no marker between them. The
parser's "first `{` to last `}`" recovery then splices thinking into the
payload and dies on a stray delimiter, because reasoning prose contains braces
and half-written drafts of the answer. Parsing now scans for *balanced*
top-level objects, string-aware, and prefers the last one that validates.

**Turning reasoning off can be a twelve-fold speedup, and here it is also
the correct design.** Left on, nemotron spent 10,832 reasoning tokens before
the answer began, ran out of budget mid-object, and failed three attempts in
twenty-one minutes. With `reasoning: {"enabled": false}` on the same example
and prompt: valid on the first attempt, 975 completion tokens, 33 seconds.

That is worth stating as a principle rather than a tuning tip. The market
engine already did the hard reasoning deterministically; the teacher's job is
to interpret facts it is handed, not to re-derive structure from raw candles.
A model burning twelve thousand characters rediscovering what the context
already states is both slow and displaying exactly the behaviour most likely
to produce ungrounded numbers. `extra_body` on the OpenRouter config is the
seam for saying so per model.

One encouraging measurement from the same response: `cached_tokens: 8704` of
8,752, meaning almost the whole prompt was served from cache. That vindicates
keeping the system prompt byte-stable and putting the per-example context in
the user message.

## Citations name things; they do not count them

The first benchmark run produced a result worth keeping: **95.5% of numeric
claims were grounded, but only 30% of examples were.** `is_grounded` is
all-or-nothing per example, so one bad citation in fifteen sinks the example.

The defects were not what the design anticipated. Across 919 claims there were
**zero ungrounded prices** - the model never invented a number. What it got
wrong was addressing:

* 27 citations dropped the root, writing `close_vs_ema.200` for
  `indicators.close_vs_ema.200`.
* Nearly all the rest were **positional**: `levels[4].price` when the price
  sat at a different index, `ohlcv_window[101].c` when the window holds 100
  bars, `ohlcv_window[86].c` quoted with the value from another bar.

The pattern is sharp. Citations to *named scalars* - `indicators.rsi`,
`structure.last_event.type` - were essentially perfect. Citations to
*positions in arrays* were where it broke, and counting list positions is a
weakness no amount of prompt instruction repairs.

So the context changed rather than the instruction: every level and swing now
carries a stable `id` (`L0`, `S4`), candles already carried their timestamp,
and the resolver accepts `levels.L2.price` or
`ohlcv_window.2016-07-22T00:00:00+00:00.c`. The model copies an identifier
sitting beside the value it is quoting instead of counting to it.

Measured on the same 30 examples, same model, same temperature:

```
                                         examples grounded   claims grounded
before, strict verifier                    20/60  (33.3%)          95.5%
after,  strict verifier                    37/60  (61.7%)         100.0%
after,  shipped verifier                   55/60  (91.7%)         100.0%
```

Of 684 citations in the second run, **not one was positional**: 76.8% named
scalars, 23.2% by identity. Claim-level grounding reached 100%.

Two verifier changes account for the rest of the gap, and it is worth being
precise about which did what:

* **Unique-suffix resolution.** A citation that does not resolve as written
  but names exactly one field in the context is resolved to it, and the
  concession is recorded as `FIELD_PATH_IMPRECISE` so a model needing it is
  distinguishable from one that does not. Ambiguity is still failure:
  `ema.20` exists under both `indicators` and `higher_timeframe`. **It fired
  zero times in the second run** - the prompt rule "start at a top-level key"
  removed the defect at source. It stays as a net for other models.
* **Naming an indicator period is not a numeric claim.** "The 200-day EMA"
  identifies a series; "RSI below 55" is a claim about the market. The
  exemption is derived from the context's own integer keys, so it tracks the
  engine's parameters and cannot excuse a period the engine never computed,
  and it requires the unit word so a bare number stays a violation. This is
  the change that carried 61.7% to 91.7%, so it should be read as a
  correction to an over-broad rule, not as evidence about the model.

The five residual failures are all real: two threshold claims and one price in
prose, one wrong level id, and one citation into an empty `setups` array.

## Reproducibility, and why it is weaker here

The market engine is byte-reproducible. The teacher is not, and pretending
otherwise would be worse than admitting it.

`temperature=0` reduces variance but does not eliminate it: mixture-of-experts
routing and server-side batching mean identical requests can produce different
tokens. So instead of assuming reproducibility, every result records what
actually happened — model id as requested *and* as resolved, the provider slug
that served it, structured mode, prompt version and hash, engine version,
context params fingerprint, token counts, cost, latency, and every attempt.

For runs that need to be as stable as possible, `provider.only` plus
`allow_fallbacks: false` pins a single provider. Two runs of "the same model"
served by different providers, or at different quantizations, are not the same
experiment.

`provider.require_parameters: true` is set whenever a `response_format` is
sent. Without it, a model that cannot honour the parameter may be silently
routed to and answer with unconstrained text — the request "succeeds" and the
degradation is invisible until parsing fails. With it, routing fails cleanly
and the fallback is deliberate and recorded. Turning an invisible failure into
a loud one is worth a slightly higher failure rate.

## Testing without spending money

Every teacher test runs offline. The `LLMProvider` protocol is the seam: the
repair loop, grounding checker and result plumbing are exercised with a
`StubProvider` returning canned responses, and the HTTP client is tested
against `httpx.MockTransport`. A suite that needs credentials gets skipped in
CI, and a skipped test protects nothing.

`msteacher dry-run` renders the exact messages and estimates their size
without calling anything. The most expensive way to find a prompt bug is
during a paid dataset run.

## Secrets

The API key is read from the environment only, never from YAML, and the config
dataclass suppresses it from `repr` so it cannot reach a log line, a traceback
or a checkpoint file. A test asserts this. `.env` is gitignored; `.env.example`
documents the variables.
