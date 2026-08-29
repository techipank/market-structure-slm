"""Scoring, sampling, and benchmarking for model outputs.

Everything here is provider-agnostic and works on a `(TeacherAnalysis,
MarketContext)` pair. That is deliberate: the same metrics score the teacher
during model selection, the dataset during curation, and the fine-tuned
student during final evaluation. If the scoring differed between those three,
their numbers could not be compared - which is the whole point of the exercise.
"""

METRICS_VERSION = "1.0.0"
