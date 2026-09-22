"""Attack, feature, detector and analysis tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argus.agent.engine import AgenticRAG
from argus.attacks.registry import build_attack, list_attacks
from argus.config import MECHANISM_CONFIGS
from argus.detect.base import compute_metrics
from argus.detect.evaluate import leave_one_attack_out
from argus.detect.models import build_detector
from argus.features.extractor import FEATURE_NAMES, FeatureExtractor
from argus.retrieval.bm25 import BM25Retriever


class TestAttacks:
    def test_registry_lists_three(self):
        assert set(list_attacks()) == {
            "poisonedrag_black",
            "poisonedrag_white",
            "corpus_poisoning",
        }

    def test_injects_expected_count(self, corpus):
        res = build_attack("poisonedrag_black", n_poison_docs=3, seed=7).apply(corpus)
        assert res.n_poison == 3 * len(corpus.queries)

    def test_does_not_mutate_original(self, corpus):
        before = len(corpus.documents)
        build_attack("poisonedrag_black", n_poison_docs=3).apply(corpus)
        assert len(corpus.documents) == before

    def test_poison_is_labelled(self, poisoned):
        for doc_id in poisoned.poison_doc_ids:
            assert poisoned.corpus.get(doc_id).is_poison

    def test_poison_contains_target_answer(self, poisoned):
        qidx = poisoned.corpus.query_index()
        for qid, ids in poisoned.poison_by_query.items():
            target = qidx[qid].target_answer
            for doc_id in ids:
                assert target.lower() in poisoned.corpus.get(doc_id).text.lower()

    def test_poison_is_retrievable(self, poisoned):
        """The retrieval condition of the attack. If poison is never retrieved the
        attack is not being tested at all."""
        r = BM25Retriever(poisoned.corpus)
        r.build()
        hits = 0
        for q in poisoned.corpus.queries:
            res = r.retrieve(q.text, top_k=5)
            if set(res.doc_ids) & poisoned.poison_for(q.query_id):
                hits += 1
        assert hits / len(poisoned.corpus.queries) > 0.5

    def test_white_box_requires_retriever(self, corpus):
        atk = build_attack("poisonedrag_white", n_poison_docs=2)
        with pytest.raises(ValueError, match="white-box"):
            atk.apply(corpus)

    def test_white_box_runs_with_retriever(self, corpus, retriever):
        res = build_attack("poisonedrag_white", n_poison_docs=2).apply(
            corpus, retriever=retriever
        )
        assert res.n_poison == 2 * len(corpus.queries)

    def test_unknown_attack(self):
        with pytest.raises(KeyError):
            build_attack("does_not_exist")


class TestFeatures:
    def test_vector_is_complete_and_numeric(self, traces):
        f = FeatureExtractor().fit(traces).extract(traces[0])
        assert set(f) == set(FEATURE_NAMES)
        assert all(isinstance(v, float) and np.isfinite(v) for v in f.values())

    def test_no_ground_truth_leaks_into_features(self, traces):
        """Features must be computable without knowing the answer or the poison ids.

        This is the constraint that makes the detector deployable, so it is enforced
        rather than trusted.
        """
        forbidden = ("poison", "gold", "target", "label", "attack_success", "correct")
        for name in FEATURE_NAMES:
            assert not any(word in name.lower() for word in forbidden), name

    def test_iteration_shows_up_in_features(self, retriever, llm, corpus):
        ex = FeatureExtractor()
        c0 = AgenticRAG(retriever, llm, MECHANISM_CONFIGS["C0"], corpus).run(corpus.queries[0])
        c2 = AgenticRAG(retriever, llm, MECHANISM_CONFIGS["C2"], corpus).run(corpus.queries[0])
        assert ex.extract(c2)["n_iterations"] > ex.extract(c0)["n_iterations"]

    def test_extract_many_aligns(self, traces):
        rows, meta = FeatureExtractor().fit(traces).extract_many(traces)
        assert len(rows) == len(meta) == len(traces)
        assert meta[0]["trace_id"] == traces[0].trace_id

    def test_degenerate_trace_still_yields_vector(self):
        from argus.telemetry.spans import Trace

        f = FeatureExtractor().extract(Trace(trace_id="t", query_id="q", query="x"))
        assert set(f) == set(FEATURE_NAMES)


class TestMetrics:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.0, 0.1, 0.9, 1.0])
        m = compute_metrics(y, s)
        assert m["roc_auc"] == 1.0
        assert m["fpr_at_95_tpr"] == 0.0

    def test_random_scores_are_near_chance(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        m = compute_metrics(y, rng.random(400))
        assert 0.35 < m["roc_auc"] < 0.65

    def test_single_class_does_not_crash(self):
        m = compute_metrics(np.zeros(10), np.random.random(10))
        assert m["roc_auc"] == 0.5


class TestDetectors:
    @staticmethod
    def _data(n=300, seed=0):
        """Synthetic feature matrix with a genuine but noisy signal."""
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        X = pd.DataFrame(
            {name: rng.normal(size=n) for name in FEATURE_NAMES[:12]}
        )
        X["retr_score_top_gap"] += y * 1.5
        X["query_drift_total"] += y * 1.0
        meta = pd.DataFrame(
            {
                "y": y,
                "attack": rng.choice(
                    ["poisonedrag_black", "poisonedrag_white", "corpus_poisoning"], n
                ),
                "dataset": "synthetic",
                "trace_id": [f"t{i}" for i in range(n)],
            }
        )
        return X, meta

    @pytest.mark.parametrize("name", ["rules", "iforest", "gbdt"])
    def test_fit_and_score(self, name):
        X, meta = self._data()
        det = build_detector(name, feature_names=list(X.columns))
        det.fit(X.to_numpy(), meta["y"].to_numpy())
        s = det.score(X.to_numpy())
        assert len(s) == len(X)
        assert np.all((s >= 0) & (s <= 1))

    def test_supervised_beats_chance(self):
        X, meta = self._data(n=600)
        det = build_detector("gbdt", feature_names=list(X.columns))
        det.fit(X.to_numpy(), meta["y"].to_numpy())
        m = compute_metrics(meta["y"].to_numpy(), det.score(X.to_numpy()))
        assert m["roc_auc"] > 0.75

    def test_gbdt_needs_labels(self):
        X, _ = self._data()
        det = build_detector("gbdt", feature_names=list(X.columns))
        with pytest.raises(ValueError, match="supervised"):
            det.fit(X.to_numpy(), None)

    def test_loao_holds_out_every_attack(self):
        X, meta = self._data(n=600)
        res = leave_one_attack_out("gbdt", X, meta, seed=0)
        assert set(res.per_split) == {
            "poisonedrag_black",
            "poisonedrag_white",
            "corpus_poisoning",
        }
        assert "leave_one_attack_out" in res.protocol

    def test_loao_needs_two_attacks(self):
        X, meta = self._data(n=100)
        meta["attack"] = "only_one"
        with pytest.raises(ValueError, match="at least two attacks"):
            leave_one_attack_out("gbdt", X, meta)

    def test_save_load(self, tmp_path):
        from argus.detect.base import Detector

        X, meta = self._data()
        det = build_detector("gbdt", feature_names=list(X.columns))
        det.fit(X.to_numpy(), meta["y"].to_numpy())
        path = det.save(tmp_path / "d.pkl")
        assert np.allclose(
            Detector.load(path).score(X.to_numpy()), det.score(X.to_numpy())
        )


class TestStats:
    def test_wilson_interval_brackets_estimate(self):
        from argus.analysis.stats import proportion_ci

        lo, hi = proportion_ci(50, 100)
        assert lo < 0.5 < hi

    def test_interval_stays_in_range_at_extremes(self):
        from argus.analysis.stats import proportion_ci

        for lo, hi in (proportion_ci(0, 50), proportion_ci(50, 50)):
            assert 0.0 <= lo <= hi <= 1.0

    def test_cohens_h_zero_when_equal(self):
        from argus.analysis.stats import cohens_h

        assert abs(cohens_h(0.5, 0.5)) < 1e-9

    def test_two_proportion_detects_real_difference(self):
        from argus.analysis.stats import two_proportion_test

        assert two_proportion_test(90, 100, 20, 100)["p_value"] < 0.001

    def test_two_proportion_ignores_noise(self):
        from argus.analysis.stats import two_proportion_test

        assert two_proportion_test(50, 100, 52, 100)["p_value"] > 0.05

    def test_bh_correction_is_conservative(self):
        from argus.analysis.stats import benjamini_hochberg

        flags = benjamini_hochberg([0.001, 0.04, 0.6, 0.9])
        assert flags[0] and not flags[2] and not flags[3]
