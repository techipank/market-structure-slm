"""Concurrent, rate-limited, resumable execution of many teacher calls.

Resumability is not a nicety here. A benchmark of 200 examples across four
models is 800 paid calls; losing them to a laptop sleeping, a network blip or
a Ctrl-C is both expensive and demoralising. Results are appended to a JSONL
checkpoint as they complete, and a re-run skips every `(model, example_id)`
already present.

Why JSONL and append-only
    A single append is close enough to atomic for this purpose, and a
    half-written final line costs one example rather than the whole file. A
    JSON array would have to be rewritten in full on every completion, which
    turns any interruption into total loss.

Why a token-bucket limiter rather than sleeping between calls
    Providers limit requests per minute, not spacing. A bucket lets a burst
    through and then throttles, which is both faster and closer to how the
    limit is actually enforced. Backoff for the 429s that get through lives in
    the provider client, where the `Retry-After` header is visible.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RateLimiter:
    """Token bucket. Thread-safe, monotonic-clock based."""

    def __init__(self, per_minute: float) -> None:
        self.capacity = max(1.0, per_minute)
        self.tokens = self.capacity
        self.rate = per_minute / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait, 5.0))


@dataclass
class BatchProgress:
    total: int
    completed: int = 0
    skipped: int = 0
    failed: int = 0

    def line(self) -> str:
        done = self.completed + self.skipped
        return (
            f"{done}/{self.total} done "
            f"({self.skipped} resumed, {self.failed} failed)"
        )


class Checkpoint:
    """Append-only JSONL store keyed by a caller-supplied id."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line from an interrupted run. Skipping
                    # it costs one example; refusing to start costs the run.
                    continue
                key = record.get("_key")
                if key:
                    self._seen.add(key)

    def has(self, key: str) -> bool:
        return key in self._seen

    def append(self, key: str, record: dict[str, Any]) -> None:
        payload = {"_key": key, **record}
        line = json.dumps(payload, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
            self._seen.add(key)

    def records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def run_batch(
    items: Sequence[Any],
    key_of: Callable[[Any], str],
    work: Callable[[Any], dict[str, Any]],
    checkpoint: Checkpoint,
    concurrency: int = 4,
    per_minute: float = 60.0,
    on_progress: Callable[[BatchProgress], None] | None = None,
) -> BatchProgress:
    """Run `work` over `items`, skipping anything already checkpointed.

    A failure inside `work` is recorded as a result rather than raised: one
    model refusing one example must not abort the other 799 calls. The failure
    is in the checkpoint, so it is visible in the report and is not silently
    retried on resume - re-running a permanent failure forever is worse than
    reporting it once.
    """
    pending = [item for item in items if not checkpoint.has(key_of(item))]
    progress = BatchProgress(total=len(items), skipped=len(items) - len(pending))
    if on_progress:
        on_progress(progress)
    if not pending:
        return progress

    limiter = RateLimiter(per_minute)
    lock = threading.Lock()

    def task(item: Any) -> None:
        limiter.acquire()
        key = key_of(item)
        try:
            record = work(item)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            record = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        checkpoint.append(key, record)
        with lock:
            progress.completed += 1
            if not record.get("ok"):
                progress.failed += 1
            if on_progress:
                on_progress(progress)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(task, item) for item in pending]
        for future in as_completed(futures):
            future.result()  # re-raise anything the wrapper somehow missed

    return progress


def iter_pairs(models: Iterable[str], examples: Sequence[Any]) -> list[tuple[str, Any]]:
    """Cross product, ordered example-major.

    Interleaving models rather than finishing one at a time means an
    interrupted run leaves partial coverage of every model instead of complete
    coverage of some - far more useful for an early look at the numbers.
    """
    model_list = list(models)
    return [(model, example) for example in examples for model in model_list]
