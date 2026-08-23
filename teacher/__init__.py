"""Teacher LLM: turns a deterministic MarketContext into a structured analysis.

The teacher is the only non-deterministic component in the pipeline, and it is
treated accordingly: nothing it says is trusted, everything it says is
recorded, and every number it states is checked against the context it was
given.

Design boundaries
-----------------
* `provider.py` defines what the rest of the code may assume about a model
  vendor. `openrouter.py` is the only module that knows about HTTP, API keys,
  or OpenRouter's request shape. Swapping vendors means writing one new module.
* Prompts are files, not string literals, and their hash is recorded on every
  result. A prompt change is a semantic change to the dataset.
"""

TEACHER_SCHEMA_VERSION = "1.0.0"

ANALYSIS_DISCLAIMER = (
    "Model-generated interpretation of historical market structure, produced "
    "for educational and research purposes only. Not trading advice, not a "
    "prediction, and not a recommendation to buy or sell anything."
)
