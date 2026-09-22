"""Mechanism ablation analysis.

This is the analysis behind research question one: which agentic mechanism changes
poisoning susceptibility, and by how much.

Every mechanism configuration is compared against the vanilla C0 baseline on the same
queries, the same attack and the same poison ratio, so the difference is attributable to
the mechanism rather than to anything else. Effect sizes come with confidence intervals
and a false-discovery-rate correction, because six comparisons made at once will produce
a spurious "significant" result if left uncorrected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from argus.analysis.stats import (
    benjamini_hochberg,
    cohens_h,
    interpret_h,
    proportion_ci,
    two_proportion_test,
)
from argus.telemetry.writer import TraceReader

MECHANISM_LABELS = {
    "C0_vanilla": "C0 vanilla",
    "C1_rewrite": "C1 query rewriting",
    "C2_iterate": "C2 iterative retrieval",
    "C3_inspect": "C3 document inspection",
    "C4_reflect": "C4 reflection",
    "C5_full": "C5 full agentic",
}

#: Datasets that exist for development and smoke tests only. The corpus loader already
#: refuses to build a reported result on them; the analysis has to refuse too. In the
#: verified run exactly one 10-query smoke-test cell
#: (``synthetic__bm25__C5__poisonedrag_black__p5__b0``) was left in the results directory,
#: and because no other configuration ran it, C5 pooled over 5,110 attacked runs while
#: every other configuration pooled over 5,100. The effect on the headline was small
#: (attack success 0.3728 against a corrected 0.3718) and the principle is not: the
#: ablation's whole claim is that configurations differ only in the mechanism under test.
DEV_DATASETS = frozenset({"synthetic"})

#: The experimental cell. Two runs are comparable only if they agree on all four, so this
#: is the unit at which a baseline has to be matched.
BLOCK_KEYS = ("dataset", "retriever", "attack", "n_poison")

#: ``C2_iterate_b1`` is ``C2_iterate`` run at iteration budget one, not a seventh
#: mechanism. The budget sweep names them separately so their cells do not collide on
#: disk, which means anything grouping by ``config_id`` sees them as unrelated.
_BUDGET_SUFFIX = re.compile(r"_b\d+$")


def base_config(config_id: str) -> str:
    """Strip a budget-variant suffix: ``C5_full_b2`` -> ``C5_full``."""
    return _BUDGET_SUFFIX.sub("", str(config_id))


def drop_dev_datasets(traces: pd.DataFrame) -> pd.DataFrame:
    """Remove development-only datasets before anything is pooled.

    See :data:`DEV_DATASETS`. Records what it dropped in ``attrs`` so a caller can say so
    rather than silently reporting a different number than the one on disk.
    """
    if traces.empty or "dataset" not in traces.columns:
        return traces
    mask = traces["dataset"].isin(DEV_DATASETS)
    if not mask.any():
        return traces
    out = traces[~mask].copy()
    out.attrs.update(traces.attrs)
    out.attrs["dev_rows_dropped"] = int(mask.sum())
    out.attrs["dev_datasets_dropped"] = sorted(set(traces.loc[mask, "dataset"]))
    return out


def _blocks_of(frame: pd.DataFrame) -> set[tuple]:
    """The set of experimental cells a frame covers."""
    if frame.empty:
        return set()
    return set(map(tuple, frame[list(BLOCK_KEYS)].itertuples(index=False, name=None)))


def matched_baseline(
    attacked: pd.DataFrame, blocks: set[tuple], baseline: str = "C0_vanilla"
) -> pd.DataFrame:
    """The baseline configuration restricted to the cells another configuration ran.

    Comparing a configuration against a baseline pooled over *more* cells than the
    configuration itself covers is the same class of error as comparing on mismatched
    query counts, and it is easy to miss because both rows look complete.

    It bit the budget sweep. ``C2_iterate_b1`` ran only the three
    nq/bm25/poisonedrag_white cells, where the vanilla baseline scores 0.739; compared
    against C0 pooled over all eleven cells (0.596) it appeared to make the attack
    *succeed* 14.4 points more often, significant at p < 1e-16. Against the baseline on
    its own three cells the difference is +0.1 points, which is nothing. Two of the four
    budget rows in the first analysis had the wrong sign for this reason.
    """
    base = attacked[attacked["config_id"] == baseline]
    if base.empty or not blocks:
        return base
    keys = pd.MultiIndex.from_frame(base[list(BLOCK_KEYS)])
    return base[keys.isin(blocks)]


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load every per-cell outcome JSON into a flat frame."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name.startswith("clean__"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "cell_id" not in data:
            continue
        row = {k: v for k, v in data.items() if k not in ("run_config", "meta")}
        row.update({f"cfg_{k}": v for k, v in data.get("run_config", {}).items()})
        row.update({f"meta_{k}": v for k, v in data.get("meta", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_traces_frame(traces_dir: str | Path, include_clean: bool = True) -> pd.DataFrame:
    """Load traces into a per-run frame.

    Clean (unattacked) runs are included by default. They used to be skipped, with the
    consequence that :func:`results_table` silently dropped its `clean_accuracy` column:
    the column is only added when unattacked rows are present, and there never were any.
    The published results table therefore reported six security metrics and no utility
    metric at all, which is precisely the tradeoff the project set out to make visible.
    """
    rows: list[dict[str, Any]] = []
    corrupt = 0
    for path in sorted(Path(traces_dir).glob("*.jsonl*")):
        is_clean_file = path.name.startswith("clean__")
        if is_clean_file and not include_clean:
            continue
        reader = TraceReader(path)
        for rec in reader.read_dicts():
            out = rec.get("outcomes", {})
            meta = rec.get("meta", {})
            rows.append(
                {
                    "trace_id": rec["trace_id"],
                    "query_id": rec["query_id"],
                    "config_id": rec.get("config_id", ""),
                    "dataset": rec.get("dataset", ""),
                    "retriever": rec.get("retriever", ""),
                    "attack": rec.get("attack", "none"),
                    "n_poison": rec.get("n_poison_in_corpus", 0),
                    "iteration_budget": rec.get("config", {}).get("iteration_budget", 1),
                    "attacked": bool(meta.get("attacked", False)),
                    "label": rec.get("label", "benign"),
                    "label_reason": rec.get("label_reason", ""),
                    "attack_success": out.get("attack_success", False),
                    "answered_correctly": out.get("answered_correctly", False),
                    # Three outcomes, not two. See RunOutcome.abstention_rate.
                    "is_abstention": out.get("is_abstention", False),
                    # Poison in the answer prompt (retrieval-stage term)...
                    "poison_in_context": out.get("poison_in_context", False),
                    # ...as distinct from poison merely gathered during retrieval.
                    "poison_retrieved": out.get(
                        "poison_retrieved", out.get("poison_in_context", False)
                    ),
                    "poison_rank": out.get("poison_rank", -1),
                    "poison_context_fraction": out.get("poison_context_fraction", 0.0),
                    "n_context_docs": out.get("n_context_docs", 0),
                    "n_iterations": out.get("n_iterations", 1),
                    "caution_applied": bool(meta.get("caution_applied", False)),
                    "reflection_insufficient": bool(meta.get("reflection_insufficient", False)),
                    "input_tokens": rec.get("total_input_tokens", 0),
                    "output_tokens": rec.get("total_output_tokens", 0),
                    "latency_ms": rec.get("total_latency_ms", 0.0),
                    "cost_usd": rec.get("cost_usd", 0.0),
                    "llm_backend": rec.get("llm_backend", ""),
                    "llm_model": rec.get("llm_model", ""),
                }
            )
        corrupt += reader.n_corrupt

    frame = pd.DataFrame(rows)
    if corrupt:
        frame.attrs["corrupt_lines"] = corrupt
    return frame


def restrict_to_matched_queries(
    traces: pd.DataFrame, group_by: str = "config_id", within: tuple[str, ...] = ("dataset", "retriever", "attack", "n_poison")
) -> pd.DataFrame:
    """Keep only the queries every configuration actually answered.

    Two cells are comparable only if they ran the same queries. Each cell answers
    `queries[:n_queries]`, so a cell with a smaller `n_queries` used a prefix of the
    other's set. In the first run C0/C1/C2 used 1,000 queries and C3/C4/C5 used 500, and
    the mechanism effects were computed straight across: C0's attack success rate is
    0.3029 over its full set but 0.2873 over the matched prefix, which inflated C4's
    reported reduction from its true −9.6 points to −12.0.

    Restricting to the intersection costs a little power and removes the bias entirely.
    """
    if traces.empty:
        return traces

    original_n = len(traces)
    traces = drop_dev_datasets(traces)

    keep_parts = []
    solo_blocks: list[tuple] = []
    for key, block in traces.groupby(list(within), dropna=False):
        per_config = block.groupby(group_by)["query_id"].apply(set)
        if per_config.empty:
            continue
        # A cell only one configuration ran cannot support a comparison, and pooling it
        # gives that configuration a composition no other configuration has. Recorded
        # rather than dropped, because a single-configuration study is a legitimate use.
        if len(per_config) < 2:
            solo_blocks.append(key if isinstance(key, tuple) else (key,))
        shared = set.intersection(*per_config.tolist())
        if not shared:
            continue
        keep_parts.append(block[block["query_id"].isin(shared)])

    if not keep_parts:
        return traces.iloc[0:0]

    out = pd.concat(keep_parts, ignore_index=True)
    out.attrs.update(traces.attrs)
    out.attrs["matched"] = True
    out.attrs["n_dropped"] = original_n - len(out)
    if solo_blocks:
        out.attrs["single_config_blocks"] = solo_blocks
    return out


def results_table(traces: pd.DataFrame, group_by: str = "config_id") -> pd.DataFrame:
    """Per-configuration summary with Wilson intervals on the rate metrics.

    Reports attack success, correctness and abstention side by side. Attack success alone
    is not enough to describe a mechanism: a configuration that answers "I cannot
    determine" to everything scores an attack success rate of zero while being useless.
    `asr_when_answered` is the number that says whether a mechanism resists poison rather
    than avoiding the question.
    """
    attacked = traces[traces["attacked"]]
    rows: list[dict[str, Any]] = []

    for key, grp in attacked.groupby(group_by):
        n = len(grp)
        succ = int(grp["attack_success"].sum())
        retr = int(grp["poison_in_context"].sum())
        lo, hi = proportion_ci(succ, n)
        ranks = grp.loc[grp["poison_rank"] >= 0, "poison_rank"]
        answered = grp[~grp["is_abstention"]]

        rows.append(
            {
                group_by: key,
                "label": MECHANISM_LABELS.get(str(key), str(key)),
                "n": n,
                "asr": succ / n if n else 0.0,
                "asr_ci_low": lo,
                "asr_ci_high": hi,
                "asr_when_answered": (
                    float(answered["attack_success"].mean()) if len(answered) else 0.0
                ),
                "poisoned_accuracy": float(grp["answered_correctly"].mean()),
                "abstention_rate": float(grp["is_abstention"].mean()),
                "poison_retrieval_rate": retr / n if n else 0.0,
                "poison_gathered_rate": float(grp["poison_retrieved"].mean()),
                "mean_poison_context_fraction": float(grp["poison_context_fraction"].mean()),
                "mean_poison_rank": float(ranks.mean()) if len(ranks) else -1.0,
                "mean_iterations": float(grp["n_iterations"].mean()),
                "mean_context_docs": float(grp["n_context_docs"].mean()),
                "mean_input_tokens": float(grp["input_tokens"].mean()),
                "mean_latency_ms": float(grp["latency_ms"].mean()),
                "total_cost_usd": float(grp["cost_usd"].sum()),
            }
        )

    # Clean accuracy comes from the paired unattacked runs, which load_traces_frame now
    # includes. Without them this block never ran and the utility columns vanished.
    clean = traces[~traces["attacked"]]
    if len(clean):
        acc = clean.groupby(group_by)["answered_correctly"].mean().to_dict()
        abst = clean.groupby(group_by)["is_abstention"].mean().to_dict()
        for row in rows:
            key = row[group_by]
            row["clean_accuracy"] = float(acc.get(key, float("nan")))
            row["clean_abstention_rate"] = float(abst.get(key, float("nan")))
            # The utility cost of the mechanism, stated next to its security benefit.
            row["clean_accuracy_delta_vs_C0"] = float(
                acc.get(key, float("nan")) - acc.get("C0_vanilla", float("nan"))
            )

    return pd.DataFrame(rows).sort_values(group_by).reset_index(drop=True)


def abstention_table(traces: pd.DataFrame) -> pd.DataFrame:
    """How much of each mechanism's protection is abstention rather than resistance.

    Answers the question the first run's headline could not. Reflection reduced attack
    success by 9.6 points; refusals rose by 13.0 and correct answers fell by 2.9, so the
    reduction was the agent going quiet rather than the agent resisting. Comparing C4
    against C6 (reflection with the caution instruction disabled) isolates it directly.
    """
    attacked = traces[traces["attacked"]]
    if attacked.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, grp in attacked.groupby("config_id"):
        base = matched_baseline(attacked, _blocks_of(grp))
        b_asr = float(base["attack_success"].mean()) if len(base) else float("nan")
        b_acc = float(base["answered_correctly"].mean()) if len(base) else float("nan")
        b_abs = float(base["is_abstention"].mean()) if len(base) else float("nan")

        asr = float(grp["attack_success"].mean())
        acc = float(grp["answered_correctly"].mean())
        abst = float(grp["is_abstention"].mean())
        d_asr, d_abst = asr - b_asr, abst - b_abs

        # Is the attack-success difference distinguishable from zero at all? The share
        # below is a ratio with that difference in the denominator, so when the
        # difference is noise the ratio is noise divided by noise. The first analysis
        # clipped it to 1.0 and printed "1.0" for C3 and C6 — the two configurations that
        # demonstrably do nothing — which reads as "entirely explained by abstention"
        # when the honest answer is "there is no effect here to explain".
        test = two_proportion_test(
            int(grp["attack_success"].sum()), len(grp),
            int(base["attack_success"].sum()), max(len(base), 1),
        )
        resolved = test["p_value"] < 0.05
        rows.append(
            {
                "config_id": key,
                "label": MECHANISM_LABELS.get(str(key), str(key)),
                "n": len(grp),
                "baseline_n": len(base),
                "asr": asr,
                "asr_delta": d_asr,
                "asr_delta_p_value": test["p_value"],
                "abstention_rate": abst,
                "abstention_delta": d_abst,
                "accuracy": acc,
                "accuracy_delta": acc - b_acc,
                # Share of the attack-success reduction that is explained by the rise in
                # abstention. Near 1.0 means the mechanism is declining to answer, not
                # resisting poison. Undefined where there is no reduction to explain.
                "share_explained_by_abstention": (
                    min(abs(d_abst) / abs(d_asr), 1.0)
                    if resolved and abs(d_asr) > 1e-9
                    else float("nan")
                ),
                "asr_when_answered": (
                    float(grp[~grp["is_abstention"]]["attack_success"].mean())
                    if (~grp["is_abstention"]).any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)


def mechanism_effects(
    traces: pd.DataFrame,
    baseline: str = "C0_vanilla",
    metric: str = "attack_success",
) -> pd.DataFrame:
    """Effect of each mechanism relative to the vanilla baseline.

    Returns one row per configuration with the difference from baseline, a confidence
    interval on that difference, Cohen's h, and an FDR-corrected significance flag.
    """
    attacked = traces[traces["attacked"]]
    if baseline not in set(attacked["config_id"]):
        raise ValueError(
            f"baseline '{baseline}' not present in traces; found {sorted(set(attacked['config_id']))}"
        )

    rows: list[dict[str, Any]] = []
    for config_id, grp in attacked.groupby("config_id"):
        n, s = len(grp), int(grp[metric].sum())
        p = s / n if n else 0.0
        # Baseline restricted to the cells this configuration actually ran. See
        # matched_baseline() for what pooling it over everything did to the budget rows.
        base = matched_baseline(attacked, _blocks_of(grp), baseline)
        n_base, s_base = len(base), int(base[metric].sum())
        p_base = s_base / n_base if n_base else 0.0
        test = two_proportion_test(s, n, s_base, n_base)
        h = cohens_h(p, p_base)
        rows.append(
            {
                "config_id": config_id,
                "label": MECHANISM_LABELS.get(str(config_id), str(config_id)),
                "n": n,
                metric: p,
                "baseline": p_base,
                "baseline_n": n_base,
                "n_blocks": len(_blocks_of(grp)),
                "diff": test["diff"],
                "diff_ci_low": test["ci_low"],
                "diff_ci_high": test["ci_high"],
                "cohens_h": h,
                "effect": interpret_h(h),
                "p_value": test["p_value"],
                "is_baseline": config_id == baseline,
            }
        )

    frame = pd.DataFrame(rows)
    frame["significant_fdr"] = False
    mask = ~frame["is_baseline"]
    if mask.any():
        flags = benjamini_hochberg(list(frame.loc[mask, "p_value"]))
        frame.loc[mask, "significant_fdr"] = pd.Series(
            flags, index=frame.index[mask], dtype=bool
        )

    return frame.sort_values("config_id").reset_index(drop=True)


def budget_curve(traces: pd.DataFrame) -> pd.DataFrame:
    """Dose-response of attack success against the iteration budget.

    This is research question three: does agency keep protecting as the budget grows, or
    does a longer trajectory eventually pull in more poison than it filters out?

    A budget curve is only a curve if every point on it differs *in the budget alone*.
    Grouping by ``config_id`` does not give that. The sweep runs the budget-one and
    budget-two variants under their own names (``C2_iterate_b1``, ``C2_iterate_b2``) on
    the three nq/bm25/poisonedrag_white cells, while the budget-three run is the ordinary
    ``C2_iterate`` cell pooled over all eleven cells including two other attacks, a second
    dataset and a second retriever. The three points are then plotted on one axis. Read
    left to right they showed C2 falling 0.740 -> 0.710 -> 0.565 and C5 falling
    0.453 -> 0.488 -> 0.373, which looks like iteration buying protection.

    On the cells all three budgets share, it does not: C2 goes 0.740 -> 0.710 -> 0.711 and
    C5 goes 0.453 -> 0.488 -> 0.488. Flat, and for C5 mildly the wrong way.

    So: fold the ``_bN`` variants back onto the configuration they are variants of, keep
    only the cells that configuration ran at *every* budget, and carry the vanilla
    baseline measured on those same cells so the curve has something to be flat against.
    """
    attacked = traces[traces["attacked"]]
    if attacked.empty:
        return pd.DataFrame()

    attacked = attacked.assign(base_config=attacked["config_id"].map(base_config))

    rows: list[dict[str, Any]] = []
    for cfg, family in attacked.groupby("base_config"):
        budgets = sorted(family["iteration_budget"].unique())
        if len(budgets) < 2:
            continue  # nothing swept, so nothing to plot a dose against

        # Cells present at every budget. Anything else is a composition change wearing
        # the costume of a budget change.
        shared = set.intersection(
            *(_blocks_of(family[family["iteration_budget"] == b]) for b in budgets)
        )
        if not shared:
            continue

        keys = pd.MultiIndex.from_frame(family[list(BLOCK_KEYS)])
        matched = family[keys.isin(shared)]
        base = matched_baseline(attacked, shared)
        base_asr = float(base["attack_success"].mean()) if len(base) else float("nan")

        for budget in budgets:
            grp = matched[matched["iteration_budget"] == budget]
            n = len(grp)
            if not n:
                continue
            succ = int(grp["attack_success"].sum())
            lo, hi = proportion_ci(succ, n)
            rows.append(
                {
                    "config_id": cfg,
                    "iteration_budget": budget,
                    "n": n,
                    "n_blocks": len(shared),
                    "asr": succ / n,
                    "asr_ci_low": lo,
                    "asr_ci_high": hi,
                    "baseline_asr_same_blocks": base_asr,
                    "baseline_n": len(base),
                    "abstention_rate": float(grp["is_abstention"].mean()),
                    "poisoned_accuracy": float(grp["answered_correctly"].mean()),
                    "poison_retrieval_rate": float(grp["poison_in_context"].mean()),
                    "mean_iterations": float(grp["n_iterations"].mean()),
                    "mean_input_tokens": float(grp["input_tokens"].mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["config_id", "iteration_budget"]).reset_index(drop=True)


def poison_ratio_curve(traces: pd.DataFrame) -> pd.DataFrame:
    """Attack success against the number of injected documents."""
    attacked = traces[traces["attacked"]]
    rows: list[dict[str, Any]] = []
    for (config_id, ratio), grp in attacked.groupby(["config_id", "n_poison"]):
        n = len(grp)
        succ = int(grp["attack_success"].sum())
        lo, hi = proportion_ci(succ, n)
        rows.append(
            {
                "config_id": config_id,
                "n_poison": ratio,
                "n": n,
                "asr": succ / n if n else 0.0,
                "asr_ci_low": lo,
                "asr_ci_high": hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["config_id", "n_poison"]).reset_index(drop=True)


def cost_table(traces: pd.DataFrame) -> pd.DataFrame:
    """Security against cost, which the literature usually reports separately."""
    summary = results_table(traces)
    if summary.empty:
        return summary
    base = summary[summary["config_id"] == "C0_vanilla"]
    base_tokens = float(base["mean_input_tokens"].iloc[0]) if len(base) else 1.0
    summary["token_overhead_x"] = summary["mean_input_tokens"] / max(base_tokens, 1.0)
    if len(base):
        # Reduction against the baseline on the *same* cells, for the reason in
        # matched_baseline(). Against the pooled baseline the budget-one rows appeared to
        # make the attack 14 points more likely, which is an artifact of comparing three
        # cells against eleven, and it propagated straight into the cost-effectiveness
        # column as a negative security-per-token figure.
        attacked = traces[traces["attacked"]]
        reductions = []
        for cid in summary["config_id"]:
            grp = attacked[attacked["config_id"] == cid]
            ref = matched_baseline(attacked, _blocks_of(grp))
            b = float(ref["attack_success"].mean()) if len(ref) else float("nan")
            reductions.append(b - float(grp["attack_success"].mean()))
        summary["asr_reduction"] = reductions
        summary["asr_reduction_per_token_x"] = summary["asr_reduction"] / summary[
            "token_overhead_x"
        ].replace(0, float("nan"))
    return summary
