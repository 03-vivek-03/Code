#!/usr/bin/env python3
"""Regenerate the tracked sample artefacts under data/samples/.

    python scripts/make_samples.py

These files are small, deterministic and committed to the repository so that someone
reading the code can see what a corpus, a poisoned document and an execution trace
actually look like without running anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.agent.engine import AgenticRAG  # noqa: E402
from argus.attacks.registry import build_attack  # noqa: E402
from argus.config import MECHANISM_CONFIGS  # noqa: E402
from argus.corpus.synthetic import build_synthetic_corpus  # noqa: E402
from argus.features.extractor import FeatureExtractor  # noqa: E402
from argus.features.schema import describe_features  # noqa: E402
from argus.llm.mock import MockLLM  # noqa: E402
from argus.retrieval.bm25 import BM25Retriever  # noqa: E402
from argus.telemetry.writer import TraceWriter  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = build_synthetic_corpus(n_docs=200, n_queries=10, seed=42)
    corpus.save(OUT / "sample_corpus.jsonl")

    result = build_attack("poisonedrag_black", n_poison_docs=3, seed=42).apply(corpus)
    query = result.corpus.queries[0]
    poison_ids = sorted(result.poison_for(query.query_id))

    (OUT / "sample_poisoned_docs.json").write_text(
        json.dumps(
            {
                "note": "Poisoned passages produced by the black-box PoisonedRAG "
                        "reproduction. The query text satisfies the retrieval condition; "
                        "the assertion satisfies the generation condition.",
                "query": query.text,
                "gold_answer": query.gold_answer,
                "target_answer": query.target_answer,
                "documents": [
                    result.corpus.get(d).to_dict() for d in poison_ids
                ],
            },
            indent=2,
        )
    )

    retriever = BM25Retriever(result.corpus)
    retriever.build()
    llm = MockLLM(seed=42)

    traces = []
    for cid in ("C0", "C5"):
        agent = AgenticRAG(retriever, llm, MECHANISM_CONFIGS[cid], result.corpus)
        for q in result.corpus.queries[:3]:
            t = agent.run(q, poison_doc_ids=result.poison_for(q.query_id))
            t.dataset, t.retriever, t.attack = "synthetic", "bm25", "poisonedrag_black"
            t.n_poison_in_corpus = 3
            t.label = "compromised" if t.attack_success else "benign"
            t.meta["attacked"] = True
            traces.append(t)

    with TraceWriter(OUT / "sample_traces.jsonl", append=False) as w:
        w.write_many(traces)

    extractor = FeatureExtractor().fit(traces)
    (OUT / "sample_features.json").write_text(
        json.dumps(
            {
                "note": "Features extracted from the first sample trace. Computed from "
                        "spans alone: no model internals, no generator re-execution.",
                "trace_id": traces[0].trace_id,
                "config": traces[0].config_id,
                "label": traces[0].label,
                "features": {k: round(v, 6) for k, v in extractor.extract(traces[0]).items()},
            },
            indent=2,
        )
    )

    (OUT / "feature_dictionary.json").write_text(json.dumps(describe_features(), indent=2))

    (OUT / "README.md").write_text(
        "# Sample artefacts\n\n"
        "Small, deterministic examples of what the platform produces. Regenerate with\n"
        "`python scripts/make_samples.py`.\n\n"
        "| File | Contents |\n"
        "|---|---|\n"
        "| `sample_corpus.jsonl` | 200 documents and 10 queries with gold and target answers |\n"
        "| `sample_poisoned_docs.json` | Poisoned passages from the black-box PoisonedRAG reproduction |\n"
        "| `sample_traces.jsonl` | Six execution traces, C0 and C5, with OpenTelemetry-shaped spans |\n"
        "| `sample_features.json` | The 40-feature vector extracted from one trace |\n"
        "| `feature_dictionary.json` | Every feature with its family and description |\n\n"
        "These come from the mock backend and are illustrative, not research results.\n"
    )

    for path in sorted(OUT.iterdir()):
        print(f"  {path.name:<32s} {path.stat().st_size:>8,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
