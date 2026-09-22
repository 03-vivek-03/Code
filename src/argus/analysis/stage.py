"""Retrieval-stage versus reasoning-stage decomposition.

This is research question two, and it is the cheapest contribution in the project
because it needs no additional experiments. Prior work suggests the protective effect of
agency comes from reasoning after retrieval rather than from better retrieval, but
presents that as a suggestion, because without per-document logging the two stages
cannot be separated. This platform logs which documents actually entered context, so the
separation is a matter of arithmetic.

    P(attack succeeds)
        = P(poison enters context) x P(model misled | poison in context)
             retrieval stage              reasoning stage

If, as a mechanism is enabled, the retrieval term stays flat while the reasoning term
falls, the protection is happening after retrieval and the prior hypothesis is confirmed.
If the retrieval term falls instead, the protection is happening at retrieval and the
hypothesis is refuted. Either outcome is a result.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from argus.analysis.ablation import (
    MECHANISM_LABELS,
    _blocks_of,
    base_config,
    matched_baseline,
)
from argus.analysis.stats import proportion_ci, two_proportion_test


def stage_decomposition(traces: pd.DataFrame, group_by: str = "config_id") -> pd.DataFrame:
    """Split attack success into its retrieval and reasoning components."""
    attacked = traces[traces["attacked"]]
    rows: list[dict[str, Any]] = []

    for key, grp in attacked.groupby(group_by):
        n = len(grp)
        if n == 0:
            continue

        n_retrieved = int(grp["poison_in_context"].sum())
        p_retrieval = n_retrieved / n

        with_poison = grp[grp["poison_in_context"]]
        n_misled = int(with_poison["attack_success"].sum())
        p_reasoning = n_misled / len(with_poison) if len(with_poison) else 0.0

        n_success = int(grp["attack_success"].sum())
        p_observed = n_success / n

        r_lo, r_hi = proportion_ci(n_retrieved, n)
        g_lo, g_hi = proportion_ci(n_misled, max(len(with_poison), 1))

        # Attack success without poison in context should be near zero. When it is not,
        # the target answer is being produced for some other reason and the decomposition
        # is contaminated, so it is surfaced rather than hidden.
        without_poison = grp[~grp["poison_in_context"]]
        leakage = (
            float(without_poison["attack_success"].mean()) if len(without_poison) else 0.0
        )

        rows.append(
            {
                group_by: key,
                "label": MECHANISM_LABELS.get(str(key), str(key)),
                "n": n,
                "p_retrieval_stage": p_retrieval,
                "retrieval_ci_low": r_lo,
                "retrieval_ci_high": r_hi,
                "n_with_poison_in_context": len(with_poison),
                "p_reasoning_stage": p_reasoning,
                "reasoning_ci_low": g_lo,
                "reasoning_ci_high": g_hi,
                "p_predicted": p_retrieval * p_reasoning,
                "p_observed": p_observed,
                "decomposition_residual": p_observed - (p_retrieval * p_reasoning),
                "asr_without_poison_in_context": leakage,
            }
        )

    return pd.DataFrame(rows).sort_values(group_by).reset_index(drop=True)


def stage_attribution(
    traces: pd.DataFrame, baseline: str = "C0_vanilla"
) -> pd.DataFrame:
    """Attribute each mechanism's effect to the retrieval or the reasoning stage.

    For every configuration, compare both stage terms against the vanilla baseline and
    report which stage accounts for the change.
    """
    attacked = traces[traces["attacked"]]
    if baseline not in set(attacked["config_id"]):
        raise ValueError(f"baseline '{baseline}' not found in traces")

    rows: list[dict[str, Any]] = []
    for config_id, grp in attacked.groupby("config_id"):
        # Baseline restricted to the cells this configuration ran, for the reason given
        # in argus.analysis.ablation.matched_baseline: the budget-sweep configurations
        # cover three of the eleven cells, and both stage terms differ sharply between
        # those three and the pooled set.
        base = matched_baseline(attacked, _blocks_of(grp), baseline)
        base_n = len(base)
        base_retr = int(base["poison_in_context"].sum())
        base_with = base[base["poison_in_context"]]
        base_misled = int(base_with["attack_success"].sum())

        n = len(grp)
        retr = int(grp["poison_in_context"].sum())
        with_poison = grp[grp["poison_in_context"]]
        misled = int(with_poison["attack_success"].sum())

        retr_test = two_proportion_test(retr, n, base_retr, base_n)
        reas_test = two_proportion_test(
            misled, max(len(with_poison), 1), base_misled, max(len(base_with), 1)
        )

        d_retr, d_reas = retr_test["diff"], reas_test["diff"]
        total = abs(d_retr) + abs(d_reas)

        if config_id == baseline:
            attribution = "baseline"
        elif total < 0.02:
            attribution = "no material change"
        elif abs(d_reas) > 2 * abs(d_retr):
            attribution = "reasoning stage"
        elif abs(d_retr) > 2 * abs(d_reas):
            attribution = "retrieval stage"
        else:
            attribution = "both stages"

        rows.append(
            {
                "config_id": config_id,
                "label": MECHANISM_LABELS.get(str(config_id), str(config_id)),
                "n": n,
                "baseline_n": base_n,
                "delta_retrieval_stage": d_retr,
                "retrieval_p_value": retr_test["p_value"],
                "delta_reasoning_stage": d_reas,
                "reasoning_p_value": reas_test["p_value"],
                "share_from_reasoning": (abs(d_reas) / total) if total > 0 else 0.0,
                "attribution": attribution,
            }
        )

    return pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)


def stage_table(traces: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Both stage frames together, for reporting."""
    return {
        "decomposition": stage_decomposition(traces),
        "attribution": stage_attribution(traces),
    }


def verdict(attribution: pd.DataFrame) -> str:
    """One-line statement of what the decomposition found.

    Written to be quotable directly in the write-up, and honest about the case where the
    evidence is mixed.
    """
    # Budget variants are the same mechanism run at a different budget, so counting
    # C5_full, C5_full_b1 and C5_full_b2 as three votes weights the full stack three
    # times over in an average that is supposed to describe the mechanism set.
    rows = attribution[attribution["config_id"].map(base_config) == attribution["config_id"]]
    rows = rows[rows["attribution"] != "baseline"]
    rows = rows[rows["attribution"] != "no material change"]
    if rows.empty:
        return (
            "No mechanism produced a material change in either stage. The decomposition "
            "is inconclusive on this data."
        )

    counts = rows["attribution"].value_counts()
    top = counts.index[0]
    share = float(rows["share_from_reasoning"].mean())

    if top == "reasoning stage":
        return (
            f"The protective effect acts mainly after retrieval: on average "
            f"{share:.0%} of the change is attributable to the reasoning stage. This "
            f"is consistent with the explanation offered by prior work, and here it is "
            f"measured rather than assumed."
        )
    if top == "retrieval stage":
        return (
            f"The protective effect acts mainly at retrieval, with only {share:.0%} of "
            f"the change attributable to reasoning. This contradicts the explanation "
            f"offered by prior work."
        )
    return (
        f"The effect is split across both stages, with {share:.0%} attributable to "
        f"reasoning. Neither a purely retrieval nor a purely reasoning explanation fits."
    )
