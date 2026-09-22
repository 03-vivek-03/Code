"""Config, corpus and retrieval tests."""

from __future__ import annotations

import pytest

from argus.config import MECHANISM_CONFIGS, AgentConfig, get_agent_config
from argus.corpus.store import Corpus, Document
from argus.corpus.synthetic import build_synthetic_corpus
from argus.retrieval.bm25 import BM25Retriever, tokenize


class TestAgentConfig:
    def test_ablation_configurations_exist(self):
        """C0..C5 are the ablation; C6 is the abstention control on C4."""
        from argus.config import ABLATION_CONFIGS

        assert ABLATION_CONFIGS == ["C0", "C1", "C2", "C3", "C4", "C5"]
        assert sorted(MECHANISM_CONFIGS) == ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

    def test_c6_is_c4_without_the_caution_instruction(self):
        """The only difference between C4 and C6 must be `reflection_caution`.

        Anything else differing would confound the abstention analysis, which reads the
        gap between them as the effect of declining to answer.
        """
        c4, c6 = MECHANISM_CONFIGS["C4"], MECHANISM_CONFIGS["C6"]
        differing = {
            k
            for k in c4.to_dict()
            if c4.to_dict()[k] != c6.to_dict()[k]
        }
        assert differing == {"name", "reflection_caution"}

    def test_c0_has_no_mechanisms(self):
        assert MECHANISM_CONFIGS["C0"].n_mechanisms == 0

    def test_c5_has_all_four(self):
        assert MECHANISM_CONFIGS["C5"].n_mechanisms == 4

    def test_single_mechanism_configs_enable_exactly_one(self):
        for cid in ("C1", "C2", "C3", "C4"):
            assert MECHANISM_CONFIGS[cid].n_mechanisms == 1, cid

    def test_each_mechanism_isolated_exactly_once(self):
        """C1..C4 must cover all four mechanisms with no overlap.

        This is the property that makes the ablation an ablation.
        """
        seen = [MECHANISM_CONFIGS[c].active_mechanisms[0] for c in ("C1", "C2", "C3", "C4")]
        assert sorted(seen) == ["M1_rewrite", "M2_iterate", "M3_inspect", "M4_reflect"]

    def test_budget_requires_iteration(self):
        """A stopping decision cannot exist without iteration. That is why there is no
        separate stopping-policy configuration."""
        with pytest.raises(ValueError, match="requires iterative_retrieval"):
            AgentConfig(name="bad", iteration_budget=3)

    def test_iterative_config_accepts_budget(self):
        cfg = get_agent_config("C2", iteration_budget=2)
        assert cfg.iteration_budget == 2

    def test_non_iterative_rejects_budget(self):
        with pytest.raises(ValueError, match="fixed at 1"):
            get_agent_config("C0", iteration_budget=3)

    def test_unknown_config_id(self):
        with pytest.raises(KeyError):
            get_agent_config("C99")


class TestCorpus:
    def test_deterministic(self):
        a = build_synthetic_corpus(n_docs=100, n_queries=5, seed=1)
        b = build_synthetic_corpus(n_docs=100, n_queries=5, seed=1)
        assert [d.text for d in a.documents] == [d.text for d in b.documents]

    def test_different_seeds_differ(self):
        a = build_synthetic_corpus(n_docs=100, n_queries=5, seed=1)
        b = build_synthetic_corpus(n_docs=100, n_queries=5, seed=2)
        assert [d.text for d in a.documents] != [d.text for d in b.documents]

    def test_queries_have_distinct_gold_and_target(self, corpus):
        for q in corpus.queries:
            assert q.gold_answer
            assert q.target_answer
            assert q.gold_answer != q.target_answer

    def test_gold_docs_exist_and_contain_answer(self, corpus):
        for q in corpus.queries:
            assert q.gold_doc_ids
            texts = " ".join(corpus.get(d).text for d in q.gold_doc_ids)
            assert q.gold_answer.lower() in texts.lower()

    def test_multi_hop_queries_need_two_documents(self, corpus):
        for q in corpus.queries:
            if q.multi_hop:
                assert len(q.gold_doc_ids) >= 2

    def test_roundtrip(self, corpus, tmp_path):
        path = corpus.save(tmp_path / "c.jsonl")
        loaded = Corpus.load(path)
        assert len(loaded.documents) == len(corpus.documents)
        assert len(loaded.queries) == len(corpus.queries)
        assert loaded.documents[0].text == corpus.documents[0].text

    def test_without_poison(self, poisoned):
        clean = poisoned.corpus.without_poison()
        assert clean.n_poison == 0
        assert len(clean) < len(poisoned.corpus)


class TestBM25:
    def test_tokenize_removes_stopwords(self):
        assert "the" not in tokenize("the quick brown fox")

    def test_retrieves_gold_document(self, corpus):
        r = BM25Retriever(corpus)
        r.build()
        hits = 0
        for q in corpus.queries:
            res = r.retrieve(q.text, top_k=10)
            if any(d in q.gold_doc_ids for d in res.doc_ids):
                hits += 1
        # Retrieval must be substantially better than chance, or the whole experiment
        # measures noise rather than poisoning.
        assert hits / len(corpus.queries) > 0.5

    def test_scores_descend(self, retriever, corpus):
        res = retriever.retrieve(corpus.queries[0].text, top_k=5)
        assert res.scores == sorted(res.scores, reverse=True)

    def test_exclude_is_respected(self, retriever, corpus):
        first = retriever.retrieve(corpus.queries[0].text, top_k=3)
        second = retriever.retrieve(
            corpus.queries[0].text, top_k=3, exclude=set(first.doc_ids)
        )
        assert not set(first.doc_ids) & set(second.doc_ids)

    def test_empty_query(self, retriever):
        assert len(retriever.retrieve("", top_k=5)) == 0

    def test_build_is_idempotent(self, corpus):
        r = BM25Retriever(corpus)
        r.build()
        n = len(r._doc_ids)
        r.build()
        assert len(r._doc_ids) == n

    def test_score_gap(self, retriever, corpus):
        res = retriever.retrieve(corpus.queries[0].text, top_k=5)
        assert res.score_gap >= 0

    def test_new_documents_are_indexed(self, corpus):
        c = Corpus(name="t", documents=list(corpus.documents), queries=list(corpus.queries))
        c.add([Document(doc_id="zz", text="a completely unique sentinel phrase xyzzy")])
        r = BM25Retriever(c)
        r.build()
        assert "zz" in r.retrieve("xyzzy sentinel", top_k=5).doc_ids
