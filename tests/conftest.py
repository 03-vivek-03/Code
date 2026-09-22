"""Shared test fixtures.

Every test runs offline against the mock backend. No test may require an API key, a GPU
or network access, because a test suite that cannot run on the machine doing the work is
not a test suite.
"""

from __future__ import annotations

import pytest

from argus.agent.engine import AgenticRAG
from argus.attacks.registry import build_attack
from argus.config import (
    MECHANISM_CONFIGS,
    ArgusConfig,
    LLMConfig,
    RetrievalConfig,
)
from argus.corpus.synthetic import build_synthetic_corpus
from argus.llm.mock import MockLLM
from argus.retrieval.bm25 import BM25Retriever


@pytest.fixture(scope="session")
def corpus():
    return build_synthetic_corpus(n_docs=300, n_queries=15, seed=7)


@pytest.fixture(scope="session")
def poisoned(corpus):
    return build_attack("poisonedrag_black", n_poison_docs=3, seed=7).apply(corpus)


@pytest.fixture
def llm():
    return MockLLM(seed=7)


@pytest.fixture
def retriever(corpus):
    r = BM25Retriever(corpus)
    r.build()
    return r


@pytest.fixture
def agent(retriever, llm, corpus):
    return AgenticRAG(retriever, llm, MECHANISM_CONFIGS["C0"], corpus)


@pytest.fixture
def tmp_config(tmp_path):
    cfg = ArgusConfig(
        data_dir=tmp_path,
        seed=7,
        llm=LLMConfig(backend="mock"),
        retrieval=RetrievalConfig(backend="bm25"),
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def traces(retriever, llm, corpus):
    """A small set of traces spanning several configurations."""
    out = []
    for cid in ("C0", "C2", "C5"):
        ag = AgenticRAG(retriever, llm, MECHANISM_CONFIGS[cid], corpus)
        for q in corpus.queries[:5]:
            out.append(ag.run(q))
    return out
