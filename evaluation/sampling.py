"""Choose the benchmark examples, once, and freeze them.

Three properties matter, and each corrects a way benchmarks quietly lie.

**Identical inputs.** The sample is written to disk before any model runs, and
every model reads the same file. Comparing models on separately-sampled data
measures the sampler, not the models.

**No near-duplicates.** Adjacent bars share 59 of 60 candles and almost all
their derived features. Two hundred randomly chosen SPY bars can easily be two
hundred views of a handful of situations, producing a benchmark that measures
one market condition very precisely and everything else not at all. Selected
bars from the same symbol are therefore forced at least `min_separation` bars
apart, defaulting to the candle-window length so no two prompts overlap.

**No holdout contamination.** Choosing a teacher on examples from the eventual
test period is model selection against the test set, which is the same
mistake as tuning a prompt on it. The sampler takes a hard date cutoff and
defaults to the training era.

Stratification is over (structure state x volatility regime), the two axes the
engine's own calibration showed to be well populated. Within a stratum,
selection round-robins across symbols so no single instrument dominates.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.loaders import load_ohlcv_csv
from market_engine.context import MarketEngine
from market_engine.params import EngineParams
from market_engine.regime import classify_volatility


@dataclass(frozen=True)
class SampleSpec:
    target_count: int = 200
    #: Bars must be at least this far apart within one symbol. Defaults to the
    #: context window so selected prompts share no candles.
    min_separation: int | None = None
    #: Exclusive upper bound on bar timestamps. Everything at or after this is
    #: reserved for validation and test.
    cutoff: str = "2025-01-01"
    #: Bars before this index are skipped: the long EMAs and the ATR
    #: percentile are not yet real, and benchmarking on warm-up data measures
    #: how models handle missing fields, not how they read structure.
    min_bar_index: int = 250
    seed: int = 20260824


#: Module singleton so the default is not constructed per call.
DEFAULT_SAMPLE_SPEC = SampleSpec()


@dataclass(frozen=True)
class Example:
    """One frozen benchmark example."""

    example_id: str
    symbol: str
    interval: str
    source_file: str
    bar_index: int
    as_of: str
    stratum: str
    context: dict[str, Any]


def stratum_key(context: dict[str, Any]) -> str:
    structure = (context.get("structure") or {}).get("trend", "UNKNOWN")
    volatility = (context.get("volatility") or {}).get("regime", "UNKNOWN")
    return f"{structure}|{volatility}"


@dataclass(frozen=True)
class _Candidate:
    """A bar under consideration, without its context.

    Materialising a full context costs a 60-candle window and a dozen derived
    structures. Selection only needs the stratum and the bar index, so
    contexts are built for the ~200 chosen bars rather than the ~15,000
    considered ones - the difference between the sampler taking seconds and
    taking minutes.
    """

    source: str
    symbol: str
    interval: str
    file_name: str
    bar_index: int
    as_of: str
    stratum: str


def build_samples(
    files: list[Path],
    params: EngineParams,
    spec: SampleSpec = DEFAULT_SAMPLE_SPEC,
) -> list[Example]:
    separation = spec.min_separation or params.ohlcv_window_bars
    cutoff = pd.Timestamp(spec.cutoff, tz="UTC")
    engine = MarketEngine(params)

    # stratum -> source stem -> [candidate]
    pool: dict[str, dict[str, list[_Candidate]]] = defaultdict(lambda: defaultdict(list))
    computed_by_source: dict[str, Any] = {}

    for path in files:
        symbol, _, interval = path.stem.rpartition("_")
        symbol = symbol or path.stem
        interval = interval or "1d"
        computed = engine.compute(load_ohlcv_csv(path).frame, symbol, interval)
        computed_by_source[path.stem] = computed

        timestamps = computed.frame["datetime"]
        for index in range(spec.min_bar_index, computed.n):
            if timestamps.iloc[index] >= cutoff:
                break  # chronological, so everything after is also excluded
            # Read the two stratification axes straight off the computed
            # series rather than materialising a whole context to look at
            # two fields.
            percentile = computed.atr_percentile.iloc[index]
            regime = classify_volatility(
                None if pd.isna(percentile) else float(percentile)
            ).value
            pool[f"{computed.structure.trend[index].value}|{regime}"][path.stem].append(
                _Candidate(
                    source=path.stem,
                    symbol=symbol,
                    interval=interval,
                    file_name=path.name,
                    bar_index=index,
                    as_of=str(timestamps.iloc[index].isoformat()),
                    stratum=f"{computed.structure.trend[index].value}|{regime}",
                )
            )

    selected = _select(pool, spec, separation)

    examples: list[Example] = []
    for candidate in selected:
        computed = computed_by_source[candidate.source]
        context = engine.context_at(computed, candidate.bar_index).model_dump(
            mode="json", exclude_none=True
        )
        examples.append(
            Example(
                example_id=f"{candidate.source}#{candidate.bar_index}",
                symbol=candidate.symbol,
                interval=candidate.interval,
                source_file=candidate.file_name,
                bar_index=candidate.bar_index,
                as_of=context["as_of"],
                stratum=candidate.stratum,
                context=context,
            )
        )
    examples.sort(key=lambda e: (e.symbol, e.bar_index))
    return examples


def _select(
    pool: dict[str, dict[str, list[_Candidate]]],
    spec: SampleSpec,
    separation: int,
) -> list[_Candidate]:
    """Round-robin across strata, then across symbols within each stratum.

    Round-robin rather than proportional allocation: the point of the
    benchmark is to find where a model breaks, and rare states are exactly
    where that happens. Proportional sampling would fill the set with the
    common cases and leave two examples of the interesting ones.
    """
    rng = random.Random(spec.seed)
    chosen: list[_Candidate] = []
    taken: dict[str, list[int]] = defaultdict(list)

    # Scarcest stratum first, not alphabetical.
    #
    # The separation budget is shared per symbol: once a bar is taken, its
    # neighbours are blocked for *every* stratum, not just the one that took
    # it. So whichever strata are served first consume the budget, and the
    # ordering is not a cosmetic choice.
    #
    # Alphabetical ordering produced a textbook systematic bias -
    # CONTRACTION 11-12 examples per stratum declining monotonically to
    # UPTREND 1-7, purely because "U" sorts last. Uptrends were being
    # under-sampled in the benchmark and would have been under-represented in
    # the training set too, which is exactly the kind of defect that shows up
    # later as "the model is bad at bull markets" and gets blamed on the model.
    #
    # Serving the most constrained stratum first is the standard fix: common
    # states have candidates to spare, rare ones do not. Ties break on name so
    # the order stays deterministic.
    strata = sorted(pool, key=lambda k: (sum(len(v) for v in pool[k].values()), k))
    # Shuffle within each (stratum, symbol) bucket so selection is spread
    # across the whole history rather than clustered at its start.
    shuffled: dict[str, dict[str, list[_Candidate]]] = {}
    for stratum in strata:
        shuffled[stratum] = {}
        for source in sorted(pool[stratum]):
            candidates = list(pool[stratum][source])
            rng.shuffle(candidates)
            shuffled[stratum][source] = candidates

    # True round-robin: at most ONE example per stratum per pass, with the
    # source rotating within each stratum.
    #
    # The earlier version looped strata x sources inside a single pass, which
    # meant one pass could take 20 strata x 10 sources = 200 examples - the
    # entire target - so it never rotated at all and the first strata served
    # simply took everything. Rotating one at a time is what actually makes
    # the allocation even, and it is what "stratified" is supposed to mean.
    exhausted: set[tuple[str, str]] = set()
    cursor: dict[str, int] = dict.fromkeys(strata, 0)

    while len(chosen) < spec.target_count:
        progressed = False
        for stratum in strata:
            if len(chosen) >= spec.target_count:
                break
            sources = sorted(shuffled[stratum])
            if not sources:
                continue
            # Try each source once, starting where this stratum left off, so
            # no single symbol dominates a stratum either.
            for offset in range(len(sources)):
                source = sources[(cursor[stratum] + offset) % len(sources)]
                if (stratum, source) in exhausted:
                    continue
                candidate = _pop_separated(
                    shuffled[stratum][source], taken[source], separation
                )
                if candidate is None:
                    exhausted.add((stratum, source))
                    continue
                chosen.append(candidate)
                taken[source].append(candidate.bar_index)
                cursor[stratum] = (cursor[stratum] + offset + 1) % len(sources)
                progressed = True
                break
        if not progressed:
            break  # every bucket is empty or blocked by the separation rule

    return chosen


def _pop_separated(
    candidates: list[_Candidate], already: list[int], separation: int
) -> _Candidate | None:
    while candidates:
        candidate = candidates.pop()
        if all(abs(candidate.bar_index - other) >= separation for other in already):
            return candidate
    return None


# ------------------------------------------------------------------- io


def write_samples(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), separators=(",", ":")) + "\n")


def read_samples(path: Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(Example(**json.loads(line)))
    return examples


def describe(examples: list[Example]) -> str:
    by_stratum: dict[str, int] = defaultdict(int)
    by_symbol: dict[str, int] = defaultdict(int)
    for example in examples:
        by_stratum[example.stratum] += 1
        by_symbol[example.symbol] += 1

    lines = [f"{len(examples)} examples", "", "by symbol:"]
    lines += [f"  {name:12s} {count:4d}" for name, count in sorted(by_symbol.items())]
    lines += ["", "by stratum (structure | volatility):"]
    lines += [
        f"  {name:34s} {count:4d}"
        for name, count in sorted(by_stratum.items(), key=lambda kv: -kv[1])
    ]
    if examples:
        stamps = sorted(e.as_of for e in examples)
        lines += ["", f"date range: {stamps[0][:10]} .. {stamps[-1][:10]}"]
    return "\n".join(lines)
