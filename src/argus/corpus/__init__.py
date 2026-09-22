"""Corpus construction and loading."""

from argus.corpus.loaders import build_corpus, load_corpus, save_corpus
from argus.corpus.store import Corpus, Document, Query
from argus.corpus.synthetic import build_synthetic_corpus

__all__ = [
    "Corpus",
    "Document",
    "Query",
    "build_corpus",
    "build_synthetic_corpus",
    "load_corpus",
    "save_corpus",
]
