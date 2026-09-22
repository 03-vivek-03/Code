#!/usr/bin/env python3
"""Worked examples of the platform's API.

    python scripts/examples.py            run all
    python scripts/examples.py 3          run one

Each example is short and self-contained, meant to be read as much as run. Everything
here works offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def example_1_one_run() -> None:
    """The smallest useful thing: build a corpus, answer one question, inspect the trace."""
    from argus.agent.engine import AgenticRAG
    from argus.config import MECHANISM_CONFIGS
    from argus.corpus.synthetic import build_synthetic_corpus
    from argus.llm.mock import MockLLM
    from argus.retrieval.bm25 import BM25Retriever

    corpus = build_synthetic_corpus(n_docs=400, n_queries=10, seed=1)
    retriever = BM25Retriever(corpus)
    retriever.build()

    agent = AgenticRAG(retriever, MockLLM(seed=1), MECHANISM_CONFIGS["C5"], corpus)
    trace = agent.run(corpus.queries[0])

    print(f"question : {trace.query}")
    print(f"answer   : {trace.final_answer}")
    print(f"gold     : {trace.gold_answer}")
    print(f"correct  : {trace.answered_correctly}")
    print(f"spans    : {len(trace.spans)} across {trace.n_iterations} retrieval rounds")
    print("\nexecution:")
    for span in trace.spans:
        print(f"  {span.name:<22s} {span.duration_ms:6.1f} ms")


def example_2_poisoning() -> None:
    """Poison a corpus and watch the same question go wrong."""
    from argus.agent.engine import AgenticRAG
    from argus.attacks.registry import build_attack
    from argus.config import MECHANISM_CONFIGS
    from argus.corpus.synthetic import build_synthetic_corpus
    from argus.llm.mock import MockLLM
    from argus.retrieval.bm25 import BM25Retriever

    corpus = build_synthetic_corpus(n_docs=400, n_queries=10, seed=2)
    result = build_attack("poisonedrag_black", n_poison_docs=5, seed=2).apply(corpus)

    retriever = BM25Retriever(result.corpus)
    retriever.build()
    agent = AgenticRAG(retriever, MockLLM(seed=2), MECHANISM_CONFIGS["C0"], result.corpus)

    query = result.corpus.queries[0]
    poison = result.poison_for(query.query_id)
    trace = agent.run(query, poison_doc_ids=poison)

    print(f"question       : {trace.query}")
    print(f"gold           : {trace.gold_answer}")
    print(f"attacker wants : {trace.target_answer}")
    print(f"model said     : {trace.final_answer}")
    print(f"attack worked  : {trace.attack_success}")
    print(f"poison in ctx  : {trace.poison_in_context} at rank {trace.poison_rank}")
    print(f"\npoison text:\n  {result.corpus.get(sorted(poison)[0]).text[:220]}")


def example_3_compare_mechanisms() -> None:
    """Run every configuration against the same poisoned corpus."""
    from argus.agent.engine import AgenticRAG
    from argus.attacks.registry import build_attack
    from argus.config import MECHANISM_CONFIGS
    from argus.corpus.synthetic import build_synthetic_corpus
    from argus.llm.mock import MockLLM
    from argus.retrieval.bm25 import BM25Retriever

    corpus = build_synthetic_corpus(n_docs=600, n_queries=25, seed=3)
    result = build_attack("poisonedrag_black", n_poison_docs=5, seed=3).apply(corpus)
    retriever = BM25Retriever(result.corpus)
    retriever.build()
    llm = MockLLM(seed=3)

    print(f"{'config':<10s} {'ASR':>6s} {'poison in ctx':>14s} {'iters':>6s} {'tokens':>8s}")
    for cid, cfg in MECHANISM_CONFIGS.items():
        agent = AgenticRAG(retriever, llm, cfg, result.corpus)
        traces = [
            agent.run(q, poison_doc_ids=result.poison_for(q.query_id))
            for q in result.corpus.queries
        ]
        n = len(traces)
        print(
            f"{cid:<10s} "
            f"{sum(t.attack_success for t in traces) / n:6.2f} "
            f"{sum(t.poison_in_context for t in traces) / n:14.2f} "
            f"{sum(t.n_iterations for t in traces) / n:6.2f} "
            f"{sum(t.total_input_tokens for t in traces) / n:8.0f}"
        )


def example_4_features() -> None:
    """Extract detection features from a trace."""
    from argus.agent.engine import AgenticRAG
    from argus.config import MECHANISM_CONFIGS
    from argus.corpus.synthetic import build_synthetic_corpus
    from argus.features.extractor import FeatureExtractor
    from argus.features.schema import FEATURE_FAMILIES
    from argus.llm.mock import MockLLM
    from argus.retrieval.bm25 import BM25Retriever

    corpus = build_synthetic_corpus(n_docs=400, n_queries=10, seed=4)
    retriever = BM25Retriever(corpus)
    retriever.build()
    agent = AgenticRAG(retriever, MockLLM(seed=4), MECHANISM_CONFIGS["C5"], corpus)

    traces = [agent.run(q) for q in corpus.queries]
    extractor = FeatureExtractor().fit(traces)
    features = extractor.extract(traces[0])

    for family, names in FEATURE_FAMILIES.items():
        print(f"\n{family}:")
        for name in names:
            print(f"  {name:<28s} {features[name]:10.4f}")


def example_5_stage_decomposition() -> None:
    """The retrieval versus reasoning split, on a small grid."""
    import tempfile

    import pandas as pd

    from argus.analysis.ablation import load_traces_frame
    from argus.analysis.stage import stage_attribution, stage_decomposition, verdict
    from argus.config import ArgusConfig, LLMConfig, RunConfig
    from argus.runner.experiment import ExperimentRunner

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ArgusConfig(data_dir=Path(tmp), seed=5, llm=LLMConfig(backend="mock"))
        cfg.ensure_dirs()
        runner = ExperimentRunner(cfg, progress=lambda m: None)

        for cid in ("C0", "C2", "C5"):
            runner.run_cell(
                RunConfig(
                    config_id=cid,
                    dataset="synthetic",
                    attack="poisonedrag_black",
                    n_poison_docs=5,
                    n_queries=25,
                    seed=5,
                ),
                n_docs=600,
            )

        traces = load_traces_frame(cfg.traces_dir)
        decomp = stage_decomposition(traces)

        print("P(success) = P(poison in context) x P(misled | poison in context)\n")
        with pd.option_context("display.width", 160):
            print(
                decomp[
                    ["label", "p_retrieval_stage", "p_reasoning_stage",
                     "p_predicted", "p_observed", "decomposition_residual"]
                ].round(4).to_string(index=False)
            )
        print(f"\n{verdict(stage_attribution(traces))}")


def example_6_custom_attack() -> None:
    """Add your own attack by subclassing Attack."""
    from argus.attacks.base import Attack
    from argus.corpus.store import Query
    from argus.corpus.synthetic import build_synthetic_corpus
    from argus.retrieval.bm25 import BM25Retriever

    class RepetitionAttack(Attack):
        """Repeat the query many times, then assert the target answer.

        Crude, but it shows the extension point: implement craft() and you are done.
        """

        name = "repetition_demo"

        def craft(self, query: Query, index: int) -> str:
            return f"{(query.text + ' ') * 3}The answer is {query.target_answer}."

    corpus = build_synthetic_corpus(n_docs=300, n_queries=8, seed=6)
    result = RepetitionAttack(n_poison_docs=3, seed=6).apply(corpus)

    retriever = BM25Retriever(result.corpus)
    retriever.build()

    hits = 0
    for q in result.corpus.queries:
        res = retriever.retrieve(q.text, top_k=5)
        if set(res.doc_ids) & result.poison_for(q.query_id):
            hits += 1

    print(f"injected {result.n_poison} documents")
    print(f"retrieved for {hits}/{len(result.corpus.queries)} queries")
    print(f"\nsample:\n  {result.corpus.get(result.poison_doc_ids[0]).text[:200]}")


EXAMPLES = [
    example_1_one_run,
    example_2_poisoning,
    example_3_compare_mechanisms,
    example_4_features,
    example_5_stage_decomposition,
    example_6_custom_attack,
]


def main() -> int:
    picked = EXAMPLES
    if len(sys.argv) > 1:
        idx = int(sys.argv[1]) - 1
        if not 0 <= idx < len(EXAMPLES):
            print(f"Pick 1 to {len(EXAMPLES)}")
            return 1
        picked = [EXAMPLES[idx]]

    for fn in picked:
        print(f"\n{'=' * 74}")
        print(f"{fn.__name__}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        print("=" * 74)
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
