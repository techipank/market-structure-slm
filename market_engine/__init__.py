"""Deterministic market-structure feature engine.

Everything in this package is a pure function of the candles at or before the
bar being described. No feature may read a future bar, and that property is
enforced by a truncation-invariance test rather than by convention.

Versioning
----------
``ENGINE_VERSION`` must be bumped whenever the *meaning* of any output field
changes: a new indicator period, a different swing rule, a changed threshold.
Every emitted context records it, so a dataset generated months apart can be
told apart. Bug fixes that change output values count as semantic changes.
"""

ENGINE_VERSION = "1.0.0"
CONTEXT_SCHEMA_VERSION = "1.0.0"

DISCLAIMER = (
    "Deterministic description of historical price data for educational and "
    "research use only. Not trading advice, not a prediction, not a signal."
)
