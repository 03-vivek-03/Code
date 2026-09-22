"""Cost estimation.

Reporting robustness without the inference bill attached is one of the gaps this project
identifies in the literature, so cost is treated as a first-class measurement rather than
an afterthought. This module answers "what will this cost" before a run starts, and the
runner tracks actual spend against a hard ceiling while it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argus.config import RunConfig, get_agent_config
from argus.llm.openai_compat import price_for

#: Approximate LLM calls per agent run, by mechanism. Derived from the engine: every run
#: makes one answer call, and each enabled mechanism adds calls per iteration.
CALLS_PER_RUN = {
    "answer": 1,
    "rewrite": 1,   # per iteration when M1 is on
    "inspect": 1,   # per iteration when M3 is on
    "reflect": 1,   # per iteration when M4 is on
}

#: Average prompt size in tokens for a *single-retrieval* run. The answer call dominates
#: because it carries the context.
#:
#: These are per-document-scaled below rather than fixed, because `max_context_docs` is
#: now 10 rather than 5: an iterating agent renders roughly twice the context a vanilla
#: one does, and a flat estimate would understate the cost of exactly the configurations
#: the cost table is meant to expose.
TOKENS = {
    "answer_base": 150,        # instructions and question, independent of context
    "tokens_per_context_doc": 130,
    "answer_output": 20,
    "rewrite_input": 250,
    "rewrite_output": 15,
    "inspect_input": 700,
    "inspect_output": 5,
    "reflect_base": 120,
    "reflect_output": 5,
}


@dataclass
class CostEstimate:
    n_runs: int = 0
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_agent_runs": self.n_runs,
            "n_llm_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost_usd": round(self.cost_usd, 2),
            "model": self.model,
        }

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            n_runs=self.n_runs + other.n_runs,
            n_calls=self.n_calls + other.n_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            model=self.model or other.model,
        )


class CostEstimator:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.in_price, self.out_price = price_for(model)

    def estimate_cell(self, run: RunConfig, include_clean_baseline: bool = True) -> CostEstimate:
        cfg = get_agent_config(run.config_id, run.iteration_budget)
        iters = cfg.iteration_budget if cfg.iterative_retrieval else 1

        # Documents actually rendered into the answer prompt. An iterating agent gathers
        # top_k per round and shows the best max_context_docs of them, so its prompt grows
        # with the budget until the context window caps it.
        context_docs = min(cfg.top_k * iters, cfg.max_context_docs)
        per_doc = TOKENS["tokens_per_context_doc"]

        in_tok = TOKENS["answer_base"] + context_docs * per_doc
        out_tok = TOKENS["answer_output"]
        calls = 1

        if cfg.query_rewriting:
            n = min(iters, cfg.max_rewrites)
            in_tok += n * TOKENS["rewrite_input"]
            out_tok += n * TOKENS["rewrite_output"]
            calls += n
        if cfg.document_inspection:
            n = min(iters, cfg.max_inspections)
            in_tok += n * TOKENS["inspect_input"]
            out_tok += n * TOKENS["inspect_output"]
            calls += n
        if cfg.reflection:
            # Reflection reads the evidence gathered so far, which grows each round.
            for it in range(iters):
                in_tok += TOKENS["reflect_base"] + min(
                    cfg.top_k * (it + 1), cfg.max_context_docs
                ) * per_doc
            out_tok += iters * TOKENS["reflect_output"]
            calls += iters

        # The paired clean baseline doubles the work for attacked cells, but it is
        # cached per (dataset, retriever, config), so the amortised factor is smaller.
        multiplier = 1.3 if include_clean_baseline else 1.0
        n_runs = int(run.n_queries * multiplier)

        return CostEstimate(
            n_runs=n_runs,
            n_calls=calls * n_runs,
            input_tokens=in_tok * n_runs,
            output_tokens=out_tok * n_runs,
            cost_usd=(in_tok * n_runs / 1e6 * self.in_price)
            + (out_tok * n_runs / 1e6 * self.out_price),
            model=self.model,
        )

    def estimate_grid(self, cells: list[RunConfig]) -> CostEstimate:
        total = CostEstimate(model=self.model)
        for cell in cells:
            total = total + self.estimate_cell(cell)
        return total


def estimate_grid_cost(cells: list[RunConfig], model: str = "gpt-4o-mini") -> dict[str, Any]:
    est = CostEstimator(model).estimate_grid(cells)
    out = est.to_dict()
    out["n_cells"] = len(cells)
    out["note"] = (
        "Estimate only. Mock backend costs nothing. Verify provider pricing before "
        "quoting any figure in the write-up."
    )
    return out
