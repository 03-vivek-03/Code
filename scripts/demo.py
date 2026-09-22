#!/usr/bin/env python3
"""End-to-end demonstration, fully offline.

Runs the whole platform in about thirty seconds with no API key and no GPU:

    corpus -> attack -> ablation -> stage decomposition -> features -> detector

If this prints tables at the end, the installation works.

The numbers produced here are not research findings. They come from the deterministic
mock backend and exist to prove the pipeline is wired correctly.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from argus.analysis.ablation import (  # noqa: E402
    load_traces_frame,
    mechanism_effects,
    results_table,
)
from argus.analysis.stage import stage_attribution, stage_decomposition, verdict  # noqa: E402
from argus.config import ArgusConfig, LLMConfig, RetrievalConfig, RunConfig  # noqa: E402
from argus.detect.evaluate import feature_ablation, leave_one_attack_out  # noqa: E402
from argus.features.extractor import FeatureExtractor  # noqa: E402
from argus.runner.experiment import ExperimentRunner  # noqa: E402
from argus.telemetry.writer import TraceReader  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show(frame: pd.DataFrame, cols: list[str] | None = None) -> None:
    if frame.empty:
        print("(no rows)")
        return
    view = frame[[c for c in (cols or frame.columns) if c in frame.columns]]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(view.round(4).to_string(index=False))


def main() -> int:
    t0 = time.time()
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)

    cfg = ArgusConfig(
        data_dir=DEMO_DIR,
        seed=42,
        llm=LLMConfig(backend="mock"),
        retrieval=RetrievalConfig(backend="bm25"),
    )
    cfg.ensure_dirs()

    banner("ARGUS DEMO  |  offline, no API key, no GPU")
    print("Mock backend: results exercise the pipeline, they are not research findings.")

    # ---------------------------------------------------------------- run grid
    banner("1. Running the ablation grid")
    runner = ExperimentRunner(cfg, progress=lambda m: None)

    cells = [
        RunConfig(
            config_id=cid,
            dataset="synthetic",
            attack=attack,
            n_poison_docs=ratio,
            n_queries=30,
            retriever="bm25",
            seed=42,
        )
        for cid in ("C0", "C1", "C2", "C3", "C4", "C5")
        for attack in ("poisonedrag_black", "poisonedrag_white", "corpus_poisoning")
        for ratio in (1, 5)
    ]

    for i, cell in enumerate(cells, 1):
        runner.run_cell(cell, n_docs=800)
        print(f"  [{i:>2}/{len(cells)}] {cell.cell_id}", flush=True)

    traces = load_traces_frame(cfg.traces_dir)
    print(f"\n  {len(traces)} agent runs recorded")

    # -------------------------------------------------------------- ablation
    banner("2. Study A, part one: which mechanism matters (RQ1)")
    show(
        results_table(traces),
        ["label", "n", "asr", "asr_ci_low", "asr_ci_high", "clean_accuracy",
         "poison_retrieval_rate", "mean_iterations", "mean_input_tokens"],
    )

    print("\nEffect versus C0 vanilla:")
    show(
        mechanism_effects(traces),
        ["label", "attack_success", "diff", "diff_ci_low", "diff_ci_high",
         "cohens_h", "effect", "p_value", "significant_fdr"],
    )

    # ----------------------------------------------------------------- stage
    banner("3. Study A, part two: retrieval stage or reasoning stage (RQ2)")
    print("P(success) = P(poison in context) x P(misled | poison in context)\n")
    show(
        stage_decomposition(traces),
        ["label", "n", "p_retrieval_stage", "p_reasoning_stage",
         "p_predicted", "p_observed", "asr_without_poison_in_context"],
    )

    attrib = stage_attribution(traces)
    print("\nAttribution:")
    show(
        attrib,
        ["label", "delta_retrieval_stage", "delta_reasoning_stage",
         "share_from_reasoning", "attribution"],
    )
    print(f"\n  VERDICT: {verdict(attrib)}")

    # -------------------------------------------------------------- features
    banner("4. Study B: features from execution traces")
    all_traces = list(TraceReader.read_dir(cfg.traces_dir))
    extractor = FeatureExtractor().fit(all_traces)
    rows, meta = extractor.extract_many(all_traces)
    X, M = pd.DataFrame(rows), pd.DataFrame(meta)

    print(f"  {len(X)} traces x {X.shape[1]} features")
    print(f"  compromised {int(M['y'].sum())}   benign {int((M['y'] == 0).sum())}")
    print(f"  attacks: {sorted(a for a in M['attack'].unique() if a != 'none')}")
    print("\n  No model internals used. No generator re-execution.")

    # -------------------------------------------------------------- detection
    banner("5. Study B: detection, held out on an unseen attack")
    if M["y"].sum() < 10 or M["y"].nunique() < 2:
        print("  Not enough compromised traces in this small demo to train a detector.")
    else:
        results = []
        for name in ("rules", "iforest", "gbdt"):
            try:
                res = leave_one_attack_out(name, X, M, seed=42)
            except ValueError as exc:
                print(f"  {name}: skipped ({exc})")
                continue
            results.append(res)
            print(f"  {res.summary_line()}")

        if results:
            best = results[-1]
            print("\n  Per held-out attack:")
            for attack, m in best.per_split.items():
                if "roc_auc" in m:
                    print(
                        f"    {attack:<22s} AUC={m['roc_auc']:.3f}  "
                        f"F1={m['f1']:.3f}  FPR@95TPR={m['fpr_at_95_tpr']:.3f}"
                    )
            if best.feature_importance:
                print("\n  Top features:")
                for feat, imp in list(best.feature_importance.items())[:8]:
                    print(f"    {feat:<28s} {imp:.4f}")

            print("\n  Feature family ablation:")
            try:
                show(feature_ablation("gbdt", X, M, seed=42))
            except ValueError as exc:
                print(f"    skipped ({exc})")

    # ------------------------------------------------------------------ done
    banner(f"Demo complete in {time.time() - t0:.1f}s")
    print(f"Artefacts under {DEMO_DIR}")
    print(
        "\nNext:\n"
        "  argus run grid --preset pilot     larger offline rehearsal\n"
        "  argus analyse ablation --save     write the analysis to CSV\n"
        "  argus dashboard                   interactive explorer\n"
        "\nFor real results set ARGUS_LLM_BACKEND=openai in .env and use --preset main."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
