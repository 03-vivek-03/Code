"""LLM interface.

Token accounting is part of the interface rather than an add-on, because reporting cost
alongside robustness is one of the contributions of this work. Every call records its
input and output tokens so a run's dollar cost is known exactly rather than estimated
after the fact.
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = "stop"
    #: Per-token logprobs when the backend exposes them. Used by the perplexity baseline.
    logprobs: list[float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


@dataclass
class UsageTracker:
    """Running token and cost totals for one experiment.

    Shared across every agent run against one backend, and therefore mutated from several
    threads once the runner executes queries concurrently. `+=` on an int is a read, an
    add and a store — not atomic — so the counters are guarded. Without the lock the
    totals silently under-count under load, which would corrupt the cost table that is one
    of this project's reported results.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    #: US dollars per million tokens. Defaults approximate a small commercial model.
    input_price_per_m: float = 0.15
    output_price_per_m: float = 0.60
    #: True when the price above is a stand-in for an unrecognised model, so cost columns
    #: can be labelled notional rather than presented as money actually spent.
    price_is_notional: bool = False
    reference_model: str = ""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, resp: LLMResponse) -> None:
        with self._lock:
            self.input_tokens += resp.input_tokens
            self.output_tokens += resp.output_tokens
            self.calls += 1

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.input_price_per_m
            + self.output_tokens / 1_000_000 * self.output_price_per_m
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            # Tokens and latency are what a self-hosted run actually costs. The dollar
            # figure is priced at a reference model and is labelled as such.
            "cost_usd": round(self.cost_usd, 6),
            "cost_is_notional": self.price_is_notional,
            "cost_reference_model": self.reference_model,
        }


def estimate_tokens(text: str) -> int:
    """Rough token count.

    Roughly four characters per token, which is close enough for budgeting and is what
    the cost estimator uses before any real call is made.
    """
    return max(1, math.ceil(len(text) / 4))


class LLMBackend(ABC):
    """Base class for every generator backend."""

    name: str = "base"

    def __init__(self, model: str = "", temperature: float = 0.0, max_tokens: int = 256) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = UsageTracker()

    @abstractmethod
    def _generate(self, prompt: str, system: str = "", **kwargs: Any) -> LLMResponse:
        """Produce one completion."""

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> LLMResponse:
        resp = self._generate(prompt, system=system, **kwargs)
        self.usage.record(resp)
        return resp

    def reset_usage(self) -> None:
        self.usage = UsageTracker(
            input_price_per_m=self.usage.input_price_per_m,
            output_price_per_m=self.usage.output_price_per_m,
            price_is_notional=self.usage.price_is_notional,
            reference_model=self.usage.reference_model,
        )
