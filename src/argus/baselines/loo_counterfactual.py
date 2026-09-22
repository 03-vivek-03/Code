"""Leave-one-out counterfactual defence, in the style of RAGuard's Zero-Knowledge
Inference Patch.

Generate the answer from the full context, then regenerate with each document removed in
turn, and measure how far the answer shifts. A document whose removal changes the answer
is the document driving it, which is exactly what a successful poison looks like.

This is the closest published competitor to Study B's detector, and the cost comparison
is the point. With k documents in context it needs k+1 generator passes, so six at k
equals five. Trace-based detection needs none.

Implemented as a faithful reproduction of the mechanism, not of any specific codebase.
"""

from __future__ import annotations

import time
from typing import Any

from argus.agent.prompts import SYSTEM_ANSWER, answer_prompt
from argus.baselines.base import BaselineDefense, BaselineResult
from argus.corpus.store import Corpus
from argus.llm.base import LLMBackend
from argus.telemetry.spans import Trace


def _normalise(text: str) -> set[str]:
    return {
        t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t
    }


def _semantic_shift(a: str, b: str) -> float:
    """Jaccard distance between two answers. 0 means identical, 1 means disjoint."""
    ta, tb = _normalise(a), _normalise(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    return 1.0 - (len(ta & tb) / len(union)) if union else 0.0


class LOOCounterfactualDefense(BaselineDefense):
    """Leave-one-out regeneration with semantic-shift scoring."""

    name = "loo_counterfactual"

    def __init__(
        self,
        llm: LLMBackend,
        corpus: Corpus,
        threshold: float = 0.5,
        max_docs: int = 5,
        early_stop: bool = True,
    ) -> None:
        super().__init__(threshold=threshold)
        self.llm = llm
        self.corpus = corpus
        self.max_docs = max_docs
        self.early_stop = early_stop
        # k+1 generator passes: one full-context answer plus one per removed document.
        self.overhead_x = float(max_docs + 1)

    def analyse(self, trace: Trace, **kwargs: Any) -> BaselineResult:
        t0 = time.perf_counter()
        doc_ids = trace.context_doc_ids[: self.max_docs]
        docs = [d for d in (self.corpus.get(i) for i in doc_ids) if d is not None]

        if len(docs) < 2:
            return BaselineResult(score=0.0, flagged=False, latency_ms=0.0)

        def render(subset: list[Any]) -> str:
            return "\n\n".join(
                f"[DOC {i + 1}] {d.title}\n{d.text}" for i, d in enumerate(subset)
            )

        n_calls = 0
        in_tok = out_tok = 0

        full = self.llm.generate(
            answer_prompt(trace.query, render(docs)), system=SYSTEM_ANSWER, task="answer"
        )
        n_calls += 1
        in_tok += full.input_tokens
        out_tok += full.output_tokens

        shifts: list[float] = []
        for i in range(len(docs)):
            subset = docs[:i] + docs[i + 1 :]
            resp = self.llm.generate(
                answer_prompt(trace.query, render(subset)), system=SYSTEM_ANSWER, task="answer"
            )
            n_calls += 1
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens

            shift = _semantic_shift(full.text, resp.text)
            shifts.append(shift)

            # RAGuard's own optimisation: stop once a decisive shift is found. It cuts
            # the average cost but not the worst case.
            if self.early_stop and shift > 0.8:
                break

        score = max(shifts) if shifts else 0.0
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return BaselineResult(
            score=score,
            flagged=score >= self.threshold,
            n_llm_calls=n_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            detail={
                "n_docs": len(docs),
                "max_shift": score,
                "mean_shift": sum(shifts) / max(len(shifts), 1),
                "full_answer": full.text[:120],
            },
        )
