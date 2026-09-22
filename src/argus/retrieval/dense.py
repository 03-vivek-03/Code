"""Dense retrieval via sentence-transformers.

Optional. Only imported when the dense or hybrid backend is requested, so the core
platform keeps working on a machine with no torch installed.

BGE-base is the default because it fits comfortably in free-tier memory. The white-box
attack uses this model's embeddings directly, which is exactly the threat model
PoisonedRAG assumes when the attacker knows the retriever.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from argus.corpus.store import Corpus
from argus.retrieval.base import Retriever


class DenseRetriever(Retriever):
    """Dense retrieval with cosine similarity over normalised embeddings."""

    name = "dense"

    def __init__(
        self,
        corpus: Corpus,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 64,
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(corpus)
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self._model = None
        self._emb: np.ndarray | None = None
        self._doc_ids: list[str] = []

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Dense retrieval needs sentence-transformers.\n"
                '  pip install -e ".[dense]"\n'
                "Or use --retriever bm25, which needs nothing extra."
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed and L2-normalise, so a dot product is cosine similarity."""
        model = self._load_model()
        if is_query and "bge" in self.model_name.lower():
            # BGE expects this instruction prefix on the query side only.
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------------ cache
    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(
            f"{self.model_name}|{self.corpus.name}|{len(self.corpus.documents)}".encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"emb_{key}.npz"

    def build(self) -> None:
        if self._built:
            return

        self._doc_ids = [d.doc_id for d in self.corpus.documents]
        cache = self._cache_path()

        if cache is not None and cache.exists():
            blob = np.load(cache, allow_pickle=True)
            if list(blob["doc_ids"]) == self._doc_ids:
                self._emb = blob["emb"]
                self._built = True
                return

        texts = [
            f"{d.title}. {d.text}" if d.title else d.text for d in self.corpus.documents
        ]
        self._emb = self.embed(texts, is_query=False)

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, emb=self._emb, doc_ids=np.array(self._doc_ids, dtype=object))

        self._built = True

    def _search(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        if self._emb is None or not len(self._doc_ids):
            return [], []
        q = self.embed([query], is_query=True)[0]
        sims = self._emb @ q
        k = min(top_k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [self._doc_ids[i] for i in top], [float(sims[i]) for i in top]


class HybridRetriever(Retriever):
    """Score-normalised fusion of BM25 and dense retrieval."""

    name = "hybrid"

    def __init__(self, corpus: Corpus, alpha: float = 0.5, **dense_kwargs) -> None:
        super().__init__(corpus)
        from argus.retrieval.bm25 import BM25Retriever

        self.alpha = alpha
        self.sparse = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus, **dense_kwargs)

    def build(self) -> None:
        if self._built:
            return
        self.sparse.build()
        self.dense.build()
        self._built = True

    @staticmethod
    def _norm(scores: list[float]) -> list[float]:
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-9:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def _search(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        pool = max(top_k * 4, 20)
        s_ids, s_sc = self.sparse._search(query, pool)
        d_ids, d_sc = self.dense._search(query, pool)

        merged: dict[str, float] = {}
        for did, sc in zip(s_ids, self._norm(s_sc)):
            merged[did] = merged.get(did, 0.0) + (1.0 - self.alpha) * sc
        for did, sc in zip(d_ids, self._norm(d_sc)):
            merged[did] = merged.get(did, 0.0) + self.alpha * sc

        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        return [d for d, _ in ranked], [float(s) for _, s in ranked]
