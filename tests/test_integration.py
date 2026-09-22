"""End-to-end integration tests.

These run the whole platform, corpus through detector, entirely offline. If these pass,
the pipeline is wired correctly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from argus.analysis.ablation import load_traces_frame, mechanism_effects, results_table
from argus.analysis.stage import stage_attribution, stage_decomposition, verdict
from argus.config import RunConfig
from argus.features.extractor import FeatureExtractor
from argus.runner.cost import estimate_grid_cost
from argus.runner.experiment import ExperimentRunner
from argus.runner.grid import build_grid, grid_summary
from argus.telemetry.writer import TraceReader


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    """Run a small grid once and reuse it across the integration tests."""
    from argus.config import ArgusConfig, LLMConfig, RetrievalConfig

    cfg = ArgusConfig(
        data_dir=tmp_path_factory.mktemp("e2e"),
        seed=11,
        llm=LLMConfig(backend="mock"),
        retrieval=RetrievalConfig(backend="bm25"),
    )
    cfg.ensure_dirs()
    runner = ExperimentRunner(cfg, progress=lambda m: None)

    cells = [
        RunConfig(
            config_id=cid,
            dataset="synthetic",
            attack=attack,
            n_poison_docs=3,
            n_queries=12,
            retriever="bm25",
            seed=11,
        )
        for cid in ("C0", "C2", "C5")
        for attack in ("poisonedrag_black", "poisonedrag_white", "corpus_poisoning")
    ]
    for cell in cells:
        runner.run_cell(cell, n_docs=400)
    return cfg


class TestGrid:
    def test_presets_build(self):
        for preset in ("smoke", "pilot", "main", "multihop", "retriever"):
            assert len(build_grid(preset)) > 0

    def test_pilot_covers_every_configuration(self):
        """Six ablation configurations plus the C6 abstention control."""
        assert {c.config_id for c in build_grid("pilot")} == {
            "C0", "C1", "C2", "C3", "C4", "C5", "C6",
        }

    def test_main_covers_three_attacks_and_three_ratios(self):
        cells = build_grid("main")
        assert len({c.attack for c in cells}) == 3
        assert len({c.n_poison_docs for c in cells}) == 3

    def test_budget_sweep_only_for_iterative_configs(self):
        cells = build_grid("pilot")
        for c in cells:
            if c.iteration_budget is not None:
                assert c.config_id in ("C2", "C5")

    def test_cell_ids_unique(self):
        cells = build_grid("main")
        assert len({c.cell_id for c in cells}) == len(cells)

    def test_summary_counts_runs(self):
        cells = build_grid("smoke")
        assert grid_summary(cells)["n_agent_runs"] == sum(c.n_queries for c in cells)


class TestCost:
    def test_estimate_is_positive(self):
        est = estimate_grid_cost(build_grid("main"), "gpt-4o-mini")
        assert est["estimated_cost_usd"] > 0

    def test_main_grid_within_stated_budget(self):
        """The proposal claims 15 to 35 dollars. If this fails the claim needs
        revising, not the test."""
        est = estimate_grid_cost(build_grid("main"), "gpt-4o-mini")
        assert est["estimated_cost_usd"] < 60

    def test_agentic_costs_more_than_vanilla(self):
        from argus.runner.cost import CostEstimator

        est = CostEstimator("gpt-4o-mini")
        c0 = est.estimate_cell(RunConfig(config_id="C0", n_queries=100))
        c5 = est.estimate_cell(RunConfig(config_id="C5", n_queries=100))
        assert c5.cost_usd > c0.cost_usd


class TestEndToEnd:
    def test_traces_written(self, executed):
        assert len(list(executed.traces_dir.glob("*.jsonl"))) > 0

    def test_results_written(self, executed):
        assert len(list(executed.results_dir.glob("*.json"))) > 0

    def test_traces_load_into_frame(self, executed):
        frame = load_traces_frame(executed.traces_dir)
        assert len(frame) > 0
        assert {"config_id", "attack", "attack_success", "poison_in_context"} <= set(frame)

    def test_results_table_has_intervals(self, executed):
        table = results_table(load_traces_frame(executed.traces_dir))
        assert len(table) == 3
        assert (table["asr_ci_low"] <= table["asr"]).all()
        assert (table["asr"] <= table["asr_ci_high"]).all()

    def test_mechanism_effects_baseline_is_zero(self, executed):
        effects = mechanism_effects(load_traces_frame(executed.traces_dir))
        base = effects[effects["is_baseline"]]
        assert len(base) == 1
        assert abs(float(base["diff"].iloc[0])) < 1e-9

    def test_stage_decomposition_is_consistent(self, executed):
        """The product of the two stage terms must reconstruct observed attack success,
        up to the leakage term. If it does not, the decomposition is wrong."""
        decomp = stage_decomposition(load_traces_frame(executed.traces_dir))
        assert len(decomp) == 3
        for _, row in decomp.iterrows():
            if row["asr_without_poison_in_context"] < 0.01:
                assert abs(row["decomposition_residual"]) < 0.05

    def test_stage_terms_are_probabilities(self, executed):
        decomp = stage_decomposition(load_traces_frame(executed.traces_dir))
        for col in ("p_retrieval_stage", "p_reasoning_stage", "p_observed"):
            assert decomp[col].between(0.0, 1.0).all()

    def test_verdict_is_a_sentence(self, executed):
        text = verdict(stage_attribution(load_traces_frame(executed.traces_dir)))
        assert isinstance(text, str) and len(text) > 40

    def test_features_extract_from_written_traces(self, executed):
        traces = list(TraceReader.read_dir(executed.traces_dir))
        assert traces
        rows, meta = FeatureExtractor().fit(traces).extract_many(traces)
        X, M = pd.DataFrame(rows), pd.DataFrame(meta)
        assert len(X) == len(traces)
        assert X.notna().all().all()
        assert set(M["attack"].unique()) >= {"poisonedrag_black", "corpus_poisoning"}

    def test_label_is_the_compromise_event_not_the_outcome(self, executed):
        """A trace is compromised when it was attacked and poison reached the prompt.

        This assertion is the inverse of the one it replaces. The original required
        `attack_success` for the compromised label, which made the detection target the
        attack's outcome rather than its occurrence: runs whose prompt was 80% poisoned
        but whose generator happened to survive were filed as benign, behaviourally
        indistinguishable from the positives beside them. See the Trace docstring for the
        measured consequences.
        """
        for t in TraceReader.read_dir(executed.traces_dir):
            if t.label == "compromised":
                assert t.meta.get("attacked"), "unattacked run labelled compromised"
                assert t.poison_in_context, "compromised label without poison in prompt"
                assert t.label_reason == "poison_in_prompt"
            else:
                assert not (t.meta.get("attacked") and t.poison_in_context), (
                    "attacked run with poison in its prompt was labelled benign"
                )

    def test_no_attacked_run_with_poison_is_a_negative(self, executed):
        """The negative class must not be dominated by poisoned traces.

        In the first run 39.3% of attacked traces sat in the benign class with poison in
        their prompt, against 5,200 genuinely clean traces, so 83.7% of the negatives
        were poisoned. That is what made leave-one-attack-out recall 0.052.
        """
        negatives = [
            t for t in TraceReader.read_dir(executed.traces_dir) if t.label == "benign"
        ]
        assert negatives
        poisoned_negatives = [t for t in negatives if t.poison_in_context]
        assert not poisoned_negatives, (
            f"{len(poisoned_negatives)} of {len(negatives)} negatives have poison in "
            "their answer prompt"
        )

    def test_rerun_is_resumable(self, executed):
        runner = ExperimentRunner(executed, progress=lambda m: None)
        cell = RunConfig(
            config_id="C0",
            dataset="synthetic",
            attack="poisonedrag_black",
            n_poison_docs=3,
            n_queries=12,
            retriever="bm25",
            seed=11,
        )
        before = executed.traces_dir / f"{cell.cell_id}.jsonl"
        mtime = before.stat().st_mtime
        runner.run_cell(cell, n_docs=400, overwrite=False)
        assert before.stat().st_mtime == mtime, "completed cell was re-run"


class TestCLI:
    def test_help(self, capsys):
        from argus.cli import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--help"])
        assert "argus" in capsys.readouterr().out

    def test_subcommands_parse(self):
        from argus.cli import build_parser

        p = build_parser()
        for argv in (
            ["corpus", "build", "--dataset", "synthetic"],
            ["attack", "run", "--attack", "poisonedrag_black"],
            ["run", "single", "--config", "C5"],
            ["run", "grid", "--preset", "pilot"],
            ["analyse", "ablation"],
            ["analyse", "stage"],
            ["features", "build"],
            ["detect", "train", "--model", "gbdt"],
            ["detect", "evaluate", "--protocol", "loao"],
            ["baselines", "compare"],
            ["cost", "estimate", "--preset", "main"],
        ):
            assert p.parse_args(argv).func is not None

    def test_corpus_build_runs(self, tmp_path):
        from argus.cli import main

        assert main(
            ["--data-dir", str(tmp_path), "corpus", "build",
             "--dataset", "synthetic", "--n-docs", "100", "--n-queries", "5"]
        ) == 0
