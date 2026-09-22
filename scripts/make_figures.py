#!/usr/bin/env python3
"""Generate publication-ready figures from experiment results.

    python scripts/make_figures.py [--data-dir data] [--format pdf]

Writes to data/results/figures/. Every figure carries confidence intervals where the
underlying quantity is a proportion, because a bar without an interval overstates what
the data supports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from argus.analysis.ablation import (  # noqa: E402
    abstention_table,
    budget_curve,
    cost_table,
    load_traces_frame,
    mechanism_effects,
    poison_ratio_curve,
    restrict_to_matched_queries,
    results_table,
)
from argus.analysis.stage import stage_decomposition  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SHORT = {
    "C0 vanilla": "C0\nvanilla",
    "C1 query rewriting": "C1\nrewrite",
    "C2 iterative retrieval": "C2\niterate",
    "C3 document inspection": "C3\ninspect",
    "C4 reflection": "C4\nreflect",
    "C5 full agentic": "C5\nfull",
    "C6_reflect_nocaution": "C6\nno caution",
}


def _short(labels) -> list[str]:
    return [SHORT.get(str(x), str(x)) for x in labels]


def fig_outcome_composition(traces, out: Path, fmt: str) -> None:
    """The result the first run's headline hid.

    Attack success, correct answers and abstentions are three outcomes, not two. A
    configuration that stops answering scores a low attack success rate too, and reporting
    only that number makes avoidance look like resistance. In the first run reflection cut
    attack success by 9.6 points while refusals rose 13.0 and correct answers fell 2.9.
    """
    table = results_table(traces)
    if table.empty or "abstention_rate" not in table:
        return

    other = (
        1.0 - table["asr"] - table["poisoned_accuracy"] - table["abstention_rate"]
    ).clip(lower=0)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for values, colour, name in (
        (table["poisoned_accuracy"], "#27ae60", "answered correctly"),
        (table["asr"], "#c0392b", "answered with the attacker's value"),
        (table["abstention_rate"], "#7f8c8d", "declined to answer"),
        (other, "#d5d8dc", "other"),
    ):
        ax.bar(x, values, bottom=bottom, color=colour, label=name, width=0.7)
        bottom = bottom + np.asarray(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(_short(table["label"]))
    ax.set_ylabel("share of attacked runs")
    ax.set_ylim(0, 1)
    ax.set_title("Outcome composition under attack: protection or avoidance?")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out / f"fig8_outcome_composition.{fmt}")
    plt.close(fig)


def fig_abstention_share(traces, out: Path, fmt: str) -> None:
    """How much of each mechanism's protection is explained by abstention."""
    table = abstention_table(traces)
    if table.empty or "share_explained_by_abstention" not in table:
        return
    table = table[table["config_id"] != "C0_vanilla"].dropna(
        subset=["share_explained_by_abstention"]
    )
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(table))
    ax.barh(y, table["share_explained_by_abstention"], color="#7f8c8d", alpha=0.9)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=1,
               label="half the reduction is the agent going quiet")
    ax.set_yticks(y)
    ax.set_yticklabels([str(x).replace("\n", " ") for x in _short(table["label"])])
    ax.set_xlabel("share of the attack-success reduction explained by abstention")
    ax.set_xlim(0, 1)
    ax.set_title("Is the mechanism resisting poison, or declining to answer?")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / f"fig9_abstention_share.{fmt}")
    plt.close(fig)


def fig_asr_by_config(traces, out: Path, fmt: str) -> None:
    """RQ1: attack success by configuration, with Wilson intervals."""
    table = results_table(traces)
    if table.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(table))
    err = np.vstack(
        [table["asr"] - table["asr_ci_low"], table["asr_ci_high"] - table["asr"]]
    )
    ax.bar(x, table["asr"], yerr=err, capsize=4, color="#c0392b", alpha=0.85,
           label="attack success")
    if "clean_accuracy" in table:
        ax.plot(x, table["clean_accuracy"], "o--", color="#1f2d3d", label="clean accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(_short(table["label"]))
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Attack success and clean accuracy by mechanism configuration")
    ax.legend(frameon=False)
    fig.savefig(out / f"fig1_asr_by_config.{fmt}")
    plt.close(fig)


def fig_mechanism_effects(traces, out: Path, fmt: str) -> None:
    """RQ1: effect of each mechanism relative to vanilla."""
    effects = mechanism_effects(traces)
    effects = effects[~effects["is_baseline"]]
    if effects.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(effects))
    err = np.vstack(
        [effects["diff"] - effects["diff_ci_low"], effects["diff_ci_high"] - effects["diff"]]
    )
    colors = ["#27ae60" if d < 0 else "#c0392b" for d in effects["diff"]]
    ax.barh(y, effects["diff"], xerr=err, capsize=4, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("\n", " ") for s in _short(effects["label"])])
    ax.set_xlabel("change in attack success versus C0 vanilla")
    ax.set_title("Mechanism effect on poisoning susceptibility")
    for i, row in enumerate(effects.itertuples()):
        if row.significant_fdr:
            ax.text(row.diff, i, "  *", va="center", fontweight="bold")
    fig.savefig(out / f"fig2_mechanism_effects.{fmt}")
    plt.close(fig)


def fig_stage_decomposition(traces, out: Path, fmt: str) -> None:
    """RQ2: retrieval stage versus reasoning stage."""
    decomp = stage_decomposition(traces)
    if decomp.empty:
        return

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(decomp))
    w = 0.38
    ax.bar(x - w / 2, decomp["p_retrieval_stage"], w,
           label="retrieval stage: P(poison in context)", color="#2980b9", alpha=0.9)
    ax.bar(x + w / 2, decomp["p_reasoning_stage"], w,
           label="reasoning stage: P(misled | poison in context)", color="#e67e22", alpha=0.9)
    ax.plot(x, decomp["p_observed"], "k^--", ms=6, label="observed attack success")

    ax.set_xticks(x)
    ax.set_xticklabels(_short(decomp["label"]))
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where does the protective effect act?")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(out / f"fig3_stage_decomposition.{fmt}")
    plt.close(fig)


def fig_budget_curve(traces, out: Path, fmt: str) -> None:
    """RQ3: does more iteration keep helping?

    The answer is no, and the earlier version of this figure said yes. It plotted every
    ``config_id`` as its own series, so the three budget points of one configuration
    arrived under three different names and never joined into a line — eleven loose
    markers that the eye connects anyway, downwards, because the budget-three marker was
    the pooled eleven-cell run sitting far below the two budget markers measured on three
    cells. budget_curve() now matches the cells before comparing, and the line is flat.
    """
    budget = budget_curve(traces)
    if budget.empty or budget["iteration_budget"].nunique() < 2:
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for config_id, grp in budget.groupby("config_id"):
        grp = grp.sort_values("iteration_budget")
        err = np.vstack(
            [grp["asr"] - grp["asr_ci_low"], grp["asr_ci_high"] - grp["asr"]]
        )
        ax.errorbar(grp["iteration_budget"], grp["asr"], yerr=err,
                    marker="o", capsize=3, lw=1.8,
                    label=f"{config_id} (n={int(grp['n'].iloc[0])} per point)")

    base = budget["baseline_asr_same_blocks"].dropna()
    if len(base):
        ax.axhline(float(base.iloc[0]), ls="--", lw=1.2, color="0.35",
                   label="C0 vanilla, same cells")

    ax.set_xticks(sorted(budget["iteration_budget"].unique()))
    ax.set_xlabel("iteration budget")
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Does more agency keep protecting?")
    ax.legend(frameon=False, fontsize=7.5)
    fig.savefig(out / f"fig4_budget_curve.{fmt}")
    plt.close(fig)


def fig_poison_ratio(traces, out: Path, fmt: str) -> None:
    """Dose-response against the number of injected documents."""
    ratios = poison_ratio_curve(traces)
    if ratios["n_poison"].nunique() < 2:
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for config_id, grp in ratios.groupby("config_id"):
        grp = grp.sort_values("n_poison")
        ax.plot(grp["n_poison"], grp["asr"], marker="o", label=str(config_id))

    ax.set_xlabel("poisoned documents per query")
    ax.set_ylabel("attack success rate")
    ax.set_title("Dose response")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / f"fig5_poison_ratio.{fmt}")
    plt.close(fig)


def fig_security_vs_cost(traces, out: Path, fmt: str) -> None:
    """The tradeoff the literature usually reports one half of."""
    costs = cost_table(traces)
    if costs.empty or "token_overhead_x" not in costs:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(costs["token_overhead_x"], costs["asr"], s=90, c="#c0392b", alpha=0.85, zorder=3)
    for row in costs.itertuples():
        ax.annotate(
            str(row.config_id),
            (row.token_overhead_x, row.asr),
            textcoords="offset points", xytext=(7, 4), fontsize=8,
        )
    ax.set_xlabel("token cost relative to C0 vanilla")
    ax.set_ylabel("attack success rate")
    ax.set_title("Security against cost")
    fig.savefig(out / f"fig6_security_vs_cost.{fmt}")
    plt.close(fig)


def fig_detection(cfg, out: Path, fmt: str) -> None:
    """Study B: detection under the leave-one-attack-out protocol."""
    import pandas as pd

    from argus.detect.evaluate import leave_one_attack_out
    from argus.features.extractor import FeatureExtractor
    from argus.telemetry.writer import TraceReader

    traces = list(TraceReader.read_dir(cfg.traces_dir))
    if not traces:
        return

    rows, meta = FeatureExtractor().fit(traces).extract_many(traces)
    X, M = pd.DataFrame(rows), pd.DataFrame(meta)
    if M["y"].nunique() < 2 or M["attack"].nunique() < 3:
        return

    results = []
    for name in ("rules", "iforest", "gbdt"):
        try:
            results.append(leave_one_attack_out(name, X, M))
        except ValueError:
            continue
    if not results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    names = [r.detector for r in results]
    x = np.arange(len(names))
    w = 0.36
    ax1.bar(x - w / 2, [r.roc_auc for r in results], w, label="ROC AUC", color="#2980b9")
    ax1.bar(x + w / 2, [r.fpr_at_95_tpr for r in results], w,
            label="FPR at 95% TPR", color="#c0392b")
    ax1.axhline(0.5, ls="--", c="grey", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 1)
    ax1.set_title("Detection on an unseen attack")
    ax1.legend(frameon=False, fontsize=8)

    best = results[-1]
    if best.feature_importance:
        top = list(best.feature_importance.items())[:10][::-1]
        ax2.barh(range(len(top)), [v for _, v in top], color="#1f2d3d", alpha=0.85)
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels([k for k, _ in top], fontsize=7.5)
        ax2.set_xlabel("importance")
        ax2.set_title(f"Top trace features ({best.detector})")

    fig.savefig(out / f"fig7_detection.{fmt}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate figures from results")
    # Defaults to ARGUS_DATA_DIR like the rest of the CLI. It used to hardcode "data"
    # and ignore the environment, so pointing the pipeline at a scratch directory still
    # overwrote the figures in the project's own data directory.
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument(
        "--no-matched", action="store_true",
        help="plot the raw data instead of the shared query set across configurations",
    )
    args = ap.parse_args()

    from argus.config import load_config

    cfg = load_config()
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)

    if not cfg.traces_dir.exists():
        print(f"No traces under {cfg.traces_dir}. Run: argus run grid --preset pilot")
        return 1

    traces = load_traces_frame(cfg.traces_dir)
    if traces.empty:
        print("No traces found.")
        return 1

    if not args.no_matched:
        # Same restriction the analysis applies: configurations are only comparable on
        # the queries they all answered.
        traces = restrict_to_matched_queries(traces)

    out = cfg.results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    for fn in (
        fig_asr_by_config,
        fig_mechanism_effects,
        fig_stage_decomposition,
        fig_budget_curve,
        fig_poison_ratio,
        fig_security_vs_cost,
        fig_outcome_composition,
        fig_abstention_share,
    ):
        try:
            fn(traces, out, args.format)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
            print(f"  skipped {fn.__name__}: {exc}")

    try:
        fig_detection(cfg, out, args.format)
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped fig_detection: {exc}")

    made = sorted(out.glob(f"*.{args.format}"))
    print(f"Wrote {len(made)} figures to {out}")
    for path in made:
        print(f"  {path.name}")

    if set(traces["llm_backend"].unique()) <= {"mock"}:
        print(
            "\nNOTE: these figures come from mock-backend traces. They show the pipeline "
            "works; they are not research results."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
