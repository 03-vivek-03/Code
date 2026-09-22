"""Defence baselines.

These are the comparators for Study B. The argument is not that trace-based detection is
more accurate than everything else. It is that it works from a different signal at a
fraction of the cost. That argument only holds if the cost of the alternatives is
measured honestly, so every baseline reports its own inference overhead.

* ``perplexity``  input filtering by fluency. Cheap, and the standard first thing anyone
  tries.
* ``llm_judge``   ask a model whether a passage looks hostile. One extra call per document.
* ``loo_counterfactual``  the RAGuard-style leave-one-out check: regenerate the answer
  with each document removed and watch for semantic shift. k+1 generator passes.
"""

from argus.baselines.base import BaselineDefense, BaselineResult
from argus.baselines.llm_judge import LLMJudgeDefense
from argus.baselines.loo_counterfactual import LOOCounterfactualDefense
from argus.baselines.perplexity import PerplexityFilterDefense

__all__ = [
    "BaselineDefense",
    "BaselineResult",
    "LLMJudgeDefense",
    "LOOCounterfactualDefense",
    "PerplexityFilterDefense",
]
