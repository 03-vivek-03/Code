"""Retriever factory."""

from __future__ import annotations

from pathlib import Path

from argus.config import RetrievalConfig
from argus.corpus.store import Corpus
from argus.retrieval.base import Retriever
from argus.retrieval.bm25 import BM25Retriever


def build_retriever(
    corpus: Corpus,
    config: RetrievalConfig | None = None,
    cache_dir: str | Path | None = None,
) -> Retriever:
    """Construct a retriever from config.

    BM25 is always available. Dense and hybrid import sentence-transformers lazily and
    raise an actionable error if it is missing.
    """
    config = config or RetrievalConfig()
    backend = config.backend.lower()

    if backend == "bm25":
        return BM25Retriever(corpus, k1=config.bm25_k1, b=config.bm25_b)

    if backend == "dense":
        from argus.retrieval.dense import DenseRetriever

        return DenseRetriever(corpus, model_name=config.dense_model, cache_dir=cache_dir)

    if backend == "hybrid":
        from argus.retrieval.dense import HybridRetriever

        return HybridRetriever(
            corpus,
            alpha=config.hybrid_alpha,
            model_name=config.dense_model,
            cache_dir=cache_dir,
        )

    raise ValueError(f"unknown retriever '{backend}', expected bm25, dense or hybrid")
