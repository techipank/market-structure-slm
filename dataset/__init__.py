"""Distillation corpus: teacher analyses, filtered, split, and made auditable.

The benchmark asked "is this model good enough to teach?". This package asks
the different question "is *this particular answer* fit to be learned from?",
and the difference in stakes is the reason it exists as its own stage.

In a benchmark, a flawed analysis is a data point - evidence about the model,
usefully retained. In a training corpus, a flawed analysis is a lesson. An
ungrounded number does not lower a score here; it teaches the student to
invent numbers, and that behaviour is exactly what the whole project is built
to avoid. So the gate is absolute rather than statistical: an example that
fails verification is not written to the corpus at all, regardless of how good
the rest of the batch looks.
"""

from __future__ import annotations

#: Bumped when the corpus row layout changes. Recorded in every row, so a
#: mixed-vintage directory can still be read.
DATASET_SCHEMA_VERSION = "1.0.0"

__all__ = ["DATASET_SCHEMA_VERSION"]
