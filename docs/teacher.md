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

Verified end to end against a real SPY context: a well-formed analysis scores
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

Measured on a real SPY daily context:

| component | chars | ≈ tokens | notes |
|---|---|---|---|
| system prompt | 4,849 | 1,212 | identical every call; cacheable |
| user message (context) | 9,223 | 2,305 | varies per example |
| `response_format` schema | 6,334 | 1,583 | sent every call |
| **total input** | | **≈ 5,100** | |

The context was originally pretty-printed, which cost **1,285 extra tokens per
example — 37% of the payload — to transmit whitespace**. Over a
5,000-example run that is more than six million tokens spent on indentation.
It is now compact.

Remaining levers, in order of size: the response schema (1,583 tokens on every
call, unavoidable in strict mode), the candle window (`ohlcv_window_bars`,
halving it saves roughly 1,000 tokens), and prompt caching where the provider
supports it.

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
