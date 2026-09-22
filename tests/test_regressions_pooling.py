"""Regressions for the pooling and baseline-matching defects (D12-D15).

Separate from ``test_regressions.py`` only because that file had already reached the
length where finding anything in it is work. Same contract: one class per defect, named
for the defect, with the defect itself written down so a reader knows what the test is
holding shut.

These four were found after the full run completed and verified clean. None of them
changes the headline C0-C6 comparison, which is the point worth noticing: every one of
them is a *composition* error rather than a measurement error, and composition errors do
not announce themselves. Each row looked complete. The numbers were internally
consistent. They were computed over different populations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: The columns every analysis function expects, with values that make a row a valid
#: attacked run. Individual tests override the handful they care about.
_ROW_DEFAULTS = {
    "dataset": "nq",
    "retriever": "bm25",
    "attack": "a",
    "n_poison": 5,
    "attacked": True,
    "attack_success": False,
    "answered_correctly": False,
    "is_abstention": False,
    "poison_in_context": True,
    "poison_retrieved": True,
    "poison_rank": 0,
    "poison_context_fraction": 1.0,
    "n_context_docs": 5,
    "n_iterations": 1,
    "iteration_budget": 1,
    "input_tokens": 100,
    "latency_ms": 1.0,
    "cost_usd": 0.0,
    "label": "compromised",
}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_ROW_DEFAULTS, **r} for r in rows])


# ---------------------------------------------------------------------- D12
class TestD12DevDatasetsAreNotEvidence:
    """One 10-query smoke-test cell was left in the results directory. Because no other
    configuration ran it, C5 pooled over 5,110 attacked runs while every other
    configuration pooled over 5,100, and attack success read 0.3728 against a corrected
    0.3718. The size of the error is not the problem. The ablation's whole claim is that
    configurations differ only in the mechanism under test, and here one of them also
    differed in which queries it was scored on.
    """

    @staticmethod
    def _mixed() -> pd.DataFrame:
        rows = [
            {"config_id": cfg, "query_id": f"q{i}", "attack_success": i < 10}
            for cfg in ("C0_vanilla", "C5_full")
            for i in range(20)
        ]
        # The smoke-test cell: one configuration only, and every run compromised.
        rows += [
            {"config_id": "C5_full", "query_id": f"s{i}", "dataset": "synthetic",
             "attack_success": True}
            for i in range(10)
        ]
        return _frame(rows)

    def test_dev_dataset_rows_are_dropped(self):
        from argus.analysis.ablation import drop_dev_datasets

        out = drop_dev_datasets(self._mixed())
        assert set(out["dataset"]) == {"nq"}
        assert out.attrs["dev_rows_dropped"] == 10

    def test_matching_removes_the_smoke_test_cell(self):
        from argus.analysis.ablation import restrict_to_matched_queries, results_table

        table = results_table(restrict_to_matched_queries(self._mixed()))
        counts = dict(zip(table["config_id"], table["n"]))
        assert counts["C0_vanilla"] == counts["C5_full"] == 20, (
            f"configurations pooled over different query counts: {counts}"
        )
        asr = dict(zip(table["config_id"], table["asr"]))
        assert asr["C5_full"] == pytest.approx(0.5), "synthetic rows leaked into the pool"

    def test_single_configuration_cells_are_surfaced(self):
        from argus.analysis.ablation import restrict_to_matched_queries

        # Same shape, but the odd cell is a real dataset, so the dev-dataset rule does
        # not catch it. It still has to be reported rather than silently pooled.
        frame = self._mixed()
        frame.loc[frame["dataset"] == "synthetic", "dataset"] = "nq_extra"
        out = restrict_to_matched_queries(frame)
        assert out.attrs.get("single_config_blocks"), (
            "a cell only one configuration ran must be surfaced, not silently pooled"
        )


# ---------------------------------------------------------------------- D13
class TestD13BaselineIsMatchedToTheCellsCompared:
    """The budget sweep ran ``C2_iterate_b1`` on three of the eleven cells. Compared
    against C0 pooled over all eleven (0.596) it appeared to make the attack succeed 14.4
    points *more* often, at p < 1e-16. Against C0 on its own three cells (0.739) the
    difference is +0.1 points and not significant. Two of the four budget rows carried
    the wrong sign, and the error propagated into the cost table as a negative
    security-per-token figure.
    """

    @staticmethod
    def _two_cells() -> pd.DataFrame:
        # Cell "easy": C0 rarely fooled. Cell "hard": C0 usually fooled. Only C0 runs
        # both; the variant runs "hard" alone and behaves exactly as C0 does there.
        rows = []
        for i in range(100):
            rows += [
                {"config_id": "C0_vanilla", "query_id": f"e{i}", "attack": "easy",
                 "attack_success": i < 20},
                {"config_id": "C0_vanilla", "query_id": f"h{i}", "attack": "hard",
                 "attack_success": i < 80},
                {"config_id": "C2_iterate_b1", "query_id": f"h{i}", "attack": "hard",
                 "attack_success": i < 80},
            ]
        return _frame(rows)

    def test_effect_is_measured_against_the_same_cells(self):
        from argus.analysis.ablation import mechanism_effects

        row = mechanism_effects(self._two_cells()).set_index("config_id").loc["C2_iterate_b1"]
        assert row["diff"] == pytest.approx(0.0, abs=1e-9), (
            f"baseline pooled over cells the configuration never ran: diff={row['diff']}"
        )
        assert row["baseline"] == pytest.approx(0.8)
        assert row["baseline_n"] == 100
        assert not bool(row["significant_fdr"])

    def test_stage_attribution_uses_the_same_baseline(self):
        from argus.analysis.stage import stage_attribution

        row = stage_attribution(self._two_cells()).set_index("config_id").loc["C2_iterate_b1"]
        assert row["delta_reasoning_stage"] == pytest.approx(0.0, abs=1e-9)
        assert row["attribution"] == "no material change"

    def test_cost_table_reduction_is_matched(self):
        from argus.analysis.ablation import cost_table

        row = cost_table(self._two_cells()).set_index("config_id").loc["C2_iterate_b1"]
        assert row["asr_reduction"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------- D14
class TestD14BudgetCurveComparesLikeWithLike:
    """The budget figure plotted every ``config_id`` as its own series, so one
    configuration's three budget points arrived under three different names and never
    joined into a line: eleven loose markers that the eye connects anyway. The budget-3
    marker was the pooled eleven-cell run and the budget-1 and budget-2 markers were
    measured on three cells, so read left to right the figure showed iteration buying
    protection. On the cells all three budgets share it does not: C2 goes
    0.740 -> 0.710 -> 0.711 and C5 goes 0.453 -> 0.488 -> 0.488.
    """

    @staticmethod
    def _swept() -> pd.DataFrame:
        rows: list[dict] = []

        def add(cfg: str, attack: str, budget: int, n_success: int) -> None:
            rows.extend(
                {"config_id": cfg, "query_id": f"{attack}{i}", "attack": attack,
                 "iteration_budget": budget, "n_iterations": budget,
                 "attack_success": i < n_success}
                for i in range(100)
            )

        # "hard" is the swept cell. "easy" is only ever run at budget 3 by the parent
        # configuration, and pooling it drags the budget-3 point down to 0.455.
        add("C0_vanilla", "hard", 1, 74)
        add("C0_vanilla", "easy", 1, 20)
        add("C2_iterate", "hard", 3, 71)
        add("C2_iterate", "easy", 3, 20)
        add("C2_iterate_b1", "hard", 1, 74)
        add("C2_iterate_b2", "hard", 2, 71)
        return _frame(rows)

    def test_budget_points_share_one_configuration_and_one_cell_set(self):
        from argus.analysis.ablation import budget_curve

        curve = budget_curve(self._swept())
        assert set(curve["config_id"]) == {"C2_iterate"}, (
            "budget variants must fold onto the configuration they vary, not appear as "
            f"separate mechanisms: {sorted(set(curve['config_id']))}"
        )
        assert sorted(curve["iteration_budget"]) == [1, 2, 3]
        assert curve["n"].nunique() == 1, "budget points measured on different sample sizes"
        assert curve["n_blocks"].nunique() == 1

    def test_the_pooled_cell_does_not_contaminate_the_last_point(self):
        from argus.analysis.ablation import budget_curve

        curve = budget_curve(self._swept()).set_index("iteration_budget")
        assert curve.loc[3, "asr"] == pytest.approx(0.71), (
            "budget-3 point is still pooled over cells the other budgets never ran"
        )

    def test_baseline_is_carried_on_the_same_cells(self):
        from argus.analysis.ablation import budget_curve

        curve = budget_curve(self._swept())
        assert curve["baseline_asr_same_blocks"].iloc[0] == pytest.approx(0.74)

    def test_a_configuration_with_one_budget_is_not_a_curve(self):
        from argus.analysis.ablation import budget_curve

        curve = budget_curve(self._swept())
        assert "C0_vanilla" not in set(curve["config_id"])


# ---------------------------------------------------------------------- D15
class TestD15AbstentionShareNeedsAnEffectToExplain:
    """``share_explained_by_abstention`` is |change in abstention| over |change in attack
    success|, clipped to 1.0. For C3 and C6 — the two configurations that demonstrably do
    nothing — the denominator is noise, and the clip printed a confident 1.0 that reads
    as "entirely explained by abstention" when the honest answer is that there is no
    effect to explain.
    """

    @staticmethod
    def _pair(n_success: int, abstain_from: int) -> pd.DataFrame:
        rows = []
        for i in range(500):
            rows += [
                {"config_id": "C0_vanilla", "query_id": f"q{i}",
                 "attack_success": i < 300, "is_abstention": i >= 480},
                {"config_id": "CX", "query_id": f"q{i}",
                 "attack_success": i < n_success, "is_abstention": i >= abstain_from},
            ]
        return _frame(rows)

    def test_null_configuration_reports_no_share(self):
        from argus.analysis.ablation import abstention_table

        # 301/500 against 300/500: a difference of one run.
        row = abstention_table(self._pair(301, 470)).set_index("config_id").loc["CX"]
        assert np.isnan(row["share_explained_by_abstention"]), (
            "a share was reported for a configuration with no resolved effect"
        )

    def test_real_effect_still_reports_a_share(self):
        from argus.analysis.ablation import abstention_table

        # Attack success falls 40 points (0.60 -> 0.20) while abstention rises 36
        # (0.04 -> 0.40), so 0.36/0.40 of the reduction is the agent going quiet.
        row = abstention_table(self._pair(100, 300)).set_index("config_id").loc["CX"]
        assert row["asr_delta_p_value"] < 0.05
        assert row["asr_delta"] == pytest.approx(-0.4)
        assert row["share_explained_by_abstention"] == pytest.approx(0.9)


# ---------------------------------------------------------------------- D16
class TestD16DefenceComparisonIsProtocolMatched:
    """``baseline_comparison.csv`` put the content-based defences (450 traces, one attack,
    scored directly) in the same table as the trace detector (50,449 traces, three
    attacks, leave-one-attack-out) and invited a column-wise reading that nothing in it
    supported. The detector now also reports a row scored on exactly the traces the
    baselines saw, with their attack family still held out of training.
    """

    @staticmethod
    def _features(n: int = 400):
        """Two attack families and a clean pool, each family carrying both classes.

        Both classes in every family matters: an attacked run whose poison never reached
        the prompt is a negative, and it is what stops the training pool collapsing to a
        single class once a family is held out.
        """
        rng = np.random.default_rng(0)
        quarter = n // 4
        attacks = (["a"] * quarter * 2) + (["b"] * quarter) + (["none"] * (n - quarter * 3))
        # 80% of each attacked family is compromised; clean traffic never is.
        y = [1 if (a != "none" and i % 5) else 0 for i, a in enumerate(attacks)]
        meta = pd.DataFrame({
            "trace_id": [f"t{i}" for i in range(n)],
            "attack": attacks,
            "y": y,
        })
        X = pd.DataFrame({
            "f0": meta["y"] * 2.0 + rng.normal(0, 0.5, n),
            "f1": rng.normal(0, 1, n),
        })
        return X, meta

    def test_subset_scoring_holds_out_the_attack_and_the_rows(self):
        from argus.detect.evaluate import score_trace_subset_loao

        X, meta = self._features()
        subset = [t for t, a in zip(meta["trace_id"], meta["attack"]) if a in ("a", "none")]
        res = score_trace_subset_loao("gbdt", X, meta, subset, held_out_attack="a", seed=0)

        assert res.n_test == len(subset)
        # Training saw neither the held-out family nor any evaluated row.
        assert res.n_train == int((~meta["trace_id"].isin(subset) & (meta["attack"] != "a")).sum())
        assert "held out a" in res.protocol

    def test_a_single_class_subset_is_refused_not_scored(self):
        from argus.detect.evaluate import score_trace_subset_loao

        X, meta = self._features()
        positives = [t for t, y in zip(meta["trace_id"], meta["y"]) if y == 1]
        with pytest.raises(ValueError, match="single class"):
            score_trace_subset_loao("gbdt", X, meta, positives, held_out_attack="a", seed=0)
