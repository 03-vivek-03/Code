"""Baseline defence interface.

Every baseline must declare its inference overhead, expressed as a multiple of a single
agent pass. This is enforced by the interface rather than left to prose, because the
cost comparison is the central claim of Study B and it would be easy to report accuracy
alone and quietly omit the price.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from argus.telemetry.spans import Trace


@dataclass
class BaselineResult:
    """One baseline's verdict on one trace."""

    score: float  # higher means more suspicious
    flagged: bool
    n_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


class BaselineDefense(ABC):
    """Base class for comparator defences."""

    name: str = "base"
    #: Extra generator passes per query, as a multiple of one agent pass. Trace-based
    #: detection is 0.0 here, which is the point of the comparison.
    overhead_x: float = 0.0

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @abstractmethod
    def analyse(self, trace: Trace, **kwargs: Any) -> BaselineResult: ...

    def analyse_many(self, traces: list[Trace], **kwargs: Any) -> list[BaselineResult]:
        return [self.analyse(t, **kwargs) for t in traces]

    def evaluate(self, traces: list[Trace], **kwargs: Any) -> dict[str, Any]:
        """Score a set of traces and report both quality and cost."""
        import numpy as np

        from argus.detect.base import compute_metrics

        results = self.analyse_many(traces, **kwargs)
        y_true = np.array([1 if t.is_compromised else 0 for t in traces])
        scores = np.array([r.score for r in results])

        metrics = compute_metrics(y_true, scores, threshold=self.threshold)
        n = max(len(results), 1)
        return {
            "defense": self.name,
            **metrics,
            "n": len(results),
            "mean_llm_calls": sum(r.n_llm_calls for r in results) / n,
            "mean_input_tokens": sum(r.input_tokens for r in results) / n,
            "mean_latency_ms": sum(r.latency_ms for r in results) / n,
            "inference_overhead_x": self.overhead_x,
        }
