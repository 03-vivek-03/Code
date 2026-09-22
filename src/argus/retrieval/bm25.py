"""BM25 retrieval, implemented directly so the platform has no heavy dependency.

This is Okapi BM25 with the standard k1/b parameters. It is the sparse control in the
experiment design: if a finding holds for both BM25 and a dense retriever, it is not an
artefact of the embedding model.

The implementation is deliberately plain numpy so it runs anywhere, including a free
Colab CPU instance, and so that its behaviour is fully auditable for the thesis.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from argus.corpus.store import Corpus
from argus.retrieval.base import Retriever

_TOKEN = re.compile(r"[a-z0-9]+")

_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on that the to was were
    what which who with will would can could should this these those there their then than
    but or if not no do does did done been being other into over under about""".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in _STOP]
    return toks


class BM25Retriever(Retriever):
    """Okapi BM25 over an in-memory corpus."""

    name = "bm25"

    def __init__(
        self,
        corpus: Corpus,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ) -> None:
        super().__init__(corpus)
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        self._doc_ids: list[str] = []
        self._doc_len: np.ndarray = np.zeros(0)
        self._avgdl: float = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

    def build(self) -> None:
        if self._built:
            return

        self._doc_ids = []
        lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        df: Counter[str] = Counter()

        for idx, doc in enumerate(self.corpus.documents):
            self._doc_ids.append(doc.doc_id)
            text = f"{doc.title} {doc.text}" if doc.title else doc.text
            toks = tokenize(text, self.remove_stopwords)
            lengths.append(len(toks))
            tf = Counter(toks)
            for term, freq in tf.items():
                postings.setdefault(term, []).append((idx, freq))
            df.update(tf.keys())

        n_docs = max(len(self._doc_ids), 1)
        self._doc_len = np.asarray(lengths, dtype=np.float64)
        self._avgdl = float(self._doc_len.mean()) if len(self._doc_len) else 1.0
        self._postings = postings
        # Robertson/Sparck Jones IDF with the +1 guard that keeps it non-negative.
        self._idf = {
            term: math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }
        self._built = True

    def _search(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        q_terms = tokenize(query, self.remove_stopwords)
        if not q_terms or not self._doc_ids:
            return [], []

        scores = np.zeros(len(self._doc_ids), dtype=np.float64)
        denom_len = self.k1 * (1.0 - self.b + self.b * self._doc_len / max(self._avgdl, 1e-9))

        for term, q_freq in Counter(q_terms).items():
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf.get(term, 0.0)
            idx = np.fromiter((i for i, _ in posting), dtype=np.int64, count=len(posting))
            freq = np.fromiter((f for _, f in posting), dtype=np.float64, count=len(posting))
            contrib = idf * (freq * (self.k1 + 1.0)) / (freq + denom_len[idx])
            # Repeated query terms weight the match slightly more.
            np.add.at(scores, idx, contrib * (1.0 + 0.1 * (q_freq - 1)))

        k = min(top_k, len(scores))
        if k == 0:
            return [], []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        top = [i for i in top if scores[i] > 0.0]

        return [self._doc_ids[i] for i in top], [float(scores[i]) for i in top]
