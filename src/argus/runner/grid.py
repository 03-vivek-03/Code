"""Grid construction.

The design is one full grid on the primary axis plus small, focused confirmation runs on
the secondary axes, rather than a full cross product of everything.

What changed after the first run
--------------------------------

None of these changes trade statistical power for cost. The first run's problem was never
sample size — it was that four of the measurements were reading off broken apparatus, and
that configurations were being compared across different query sets. Cost is reported as
a first-class result (that is GAP-5), not used as a design constraint.

**Matched query counts.** Cells answer `queries[:n_queries]`, so cells with different
`n_queries` compare different query sets. The first run used 1,000 queries for C0, C1 and
C2 and 500 for C3, C4 and C5, which put a 1.6-point offset into every effect size: C0's
attack success rate is 0.3029 over its full set and 0.2873 over the matched prefix, so
C4's headline reduction was reported as −12.0 points rather than its true −9.6.
:func:`check_matched_queries` refuses such a grid before it starts.

At 500 queries per cell every configuration pools 4,500 attacked runs across the nine
attack-by-ratio cells, a Wilson half-width of about ±1.3 points at a 30% attack success
rate. That is ample resolution for the roughly 10-point mechanism effects under test.

**The budget sweep moved out of the main grid.** It used to multiply C2 and C5 across all
nine attack-by-ratio cells. That tripled their sample size relative to the other
configurations, made their aggregate rows incomparable, and — worst — folded budget 1,
which is identical to vanilla by construction, into the "iterative retrieval" arm. Any
effect was guaranteed to be diluted toward zero before a single run executed. RQ3 gets its
own preset, sweeping budget against poison ratio so the result is a surface rather than a
line.

**An abstention control.** C6 is reflection with the caution instruction disabled. It runs
across the full main grid alongside C4, so the analysis can separate reflection's two
effects — judging the evidence, and declining to answer — at every attack and ratio rather
than at one point.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from argus.config import RunConfig

#: The attack used wherever a preset needs a single strong attack. White-box PoisonedRAG
#: is the strongest in the measured data and is the natural choice for dose-response and
#: confirmation runs.
STRONGEST_ATTACK = "poisonedrag_white"

ALL_ATTACKS = ["poisonedrag_black", "poisonedrag_white", "corpus_poisoning"]

#: Grid presets. `pilot` is an offline rehearsal, `main` is the real experiment.
GRID_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "description": "Tiny grid for tests and demos. Seconds, offline.",
        "n_queries": 5,
        "n_docs": 300,
        "configs": ["C0", "C5"],
        "attacks": ["poisonedrag_black"],
        "poison_ratios": [5],
        "datasets": ["synthetic"],
        "retrievers": ["bm25"],
        "budgets": None,
    },
    "pilot": {
        "description": "Small end-to-end rehearsal of the real design. Offline, free.",
        "n_queries": 20,
        "n_docs": 1200,
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        # All three attacks, so leave-one-attack-out is exercised properly during the
        # rehearsal rather than only on the main grid.
        "attacks": ALL_ATTACKS,
        "poison_ratios": [1, 5],
        "datasets": ["synthetic"],
        "retrievers": ["bm25"],
        "budgets": None,
    },
    "main": {
        "description": "The main grid: 7 configs x 3 attacks x 3 ratios, matched queries.",
        "n_queries": 500,
        "n_docs": 60_000,
        # C6 runs across the whole grid rather than in a corner, so the abstention
        # decomposition is available at every attack and every poison ratio.
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        "attacks": ALL_ATTACKS,
        "poison_ratios": [1, 5, 10],
        "datasets": ["nq"],
        "retrievers": ["bm25"],
        "budgets": None,
    },
    "budget": {
        "description": "RQ3: iteration budget against poison ratio, as a surface.",
        "n_queries": 500,
        "n_docs": 60_000,
        "configs": ["C2", "C5"],
        "attacks": [STRONGEST_ATTACK],
        "poison_ratios": [1, 5, 10],
        "datasets": ["nq"],
        "retrievers": ["bm25"],
        # Budget 3 is the default and is already covered by the main grid.
        "budgets": [1, 2],
    },
    "abstention": {
        "description": "C4 against C6 alone, for re-running the abstention control.",
        "n_queries": 500,
        "n_docs": 60_000,
        "configs": ["C4", "C6"],
        "attacks": ALL_ATTACKS,
        "poison_ratios": [1, 5, 10],
        "datasets": ["nq"],
        "retrievers": ["bm25"],
        "budgets": None,
    },
    "multihop": {
        "description": "Multi-hop confirmation run: HotpotQA, all configs.",
        "n_queries": 300,
        "n_docs": 40_000,
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        "attacks": [STRONGEST_ATTACK],
        "poison_ratios": [5],
        "datasets": ["hotpotqa"],
        "retrievers": ["bm25"],
        "budgets": None,
    },
    "retriever": {
        "description": "Retriever confirmation run: dense retrieval on the main dataset.",
        "n_queries": 300,
        "n_docs": 40_000,
        "configs": ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        "attacks": [STRONGEST_ATTACK],
        "poison_ratios": [5],
        "datasets": ["nq"],
        "retrievers": ["dense"],
        "budgets": None,
    },
}

#: Iteration budgets for the dose-response curve of research question three.
BUDGET_SWEEP = [1, 2, 3]

ITERATIVE_CONFIGS = {"C2", "C5"}


class GridConsistencyError(ValueError):
    """Raised when a grid would produce cells that cannot be compared."""


def build_grid(preset: str = "pilot", seed: int = 42, **overrides: Any) -> list[RunConfig]:
    """Expand a preset into the list of cells to run."""
    if preset not in GRID_PRESETS:
        raise KeyError(f"unknown preset '{preset}', expected one of {sorted(GRID_PRESETS)}")

    spec = {**GRID_PRESETS[preset], **overrides}
    cells: list[RunConfig] = []
    budgets = spec.get("budgets")

    for dataset in spec["datasets"]:
        for retriever in spec["retrievers"]:
            for config_id in spec["configs"]:
                if budgets and config_id in ITERATIVE_CONFIGS:
                    cell_budgets: list[int | None] = list(budgets)
                else:
                    cell_budgets = [None]
                for budget in cell_budgets:
                    for attack in spec["attacks"]:
                        for ratio in spec["poison_ratios"]:
                            cells.append(
                                RunConfig(
                                    config_id=config_id,
                                    dataset=dataset,
                                    attack=attack,
                                    n_poison_docs=ratio,
                                    n_queries=spec["n_queries"],
                                    retriever=retriever,
                                    iteration_budget=budget,
                                    seed=seed,
                                )
                            )

    check_matched_queries(cells)
    return cells


def check_matched_queries(cells: list[RunConfig]) -> None:
    """Refuse a grid whose configurations answer different query sets.

    Every cell answers `queries[:n_queries]`, so a cell with a smaller `n_queries` used a
    prefix of another's set. Comparing configurations across different prefixes is what
    put a 1.6-point offset into every effect size of the first run, so the check runs
    before anything is executed rather than being discovered afterwards.
    """
    by_dataset: dict[str, set[int]] = defaultdict(set)
    for cell in cells:
        by_dataset[cell.dataset].add(cell.n_queries)

    bad = {ds: sorted(counts) for ds, counts in by_dataset.items() if len(counts) > 1}
    if bad:
        detail = "; ".join(f"{ds}: {counts}" for ds, counts in bad.items())
        raise GridConsistencyError(
            f"cells within a dataset must all use the same n_queries, found {detail}.\n"
            "Configurations answer queries[:n_queries], so differing counts mean the "
            "configurations are compared across different query sets."
        )


def grid_summary(cells: list[RunConfig], n_docs: int = 2000) -> dict[str, Any]:
    """Human-readable description of what a grid will execute."""
    configs = sorted({c.config_id for c in cells})
    return {
        "n_cells": len(cells),
        "n_agent_runs": sum(c.n_queries for c in cells),
        "configs": configs,
        "attacks": sorted({c.attack for c in cells}),
        "poison_ratios": sorted({c.n_poison_docs for c in cells}),
        "datasets": sorted({c.dataset for c in cells}),
        "retrievers": sorted({c.retriever for c in cells}),
        "iteration_budgets": sorted({c.iteration_budget or 0 for c in cells}),
        "n_queries_per_cell": sorted({c.n_queries for c in cells}),
        # Clean baselines are cached per (dataset, retriever, agent config), so this is
        # the extra work the paired baselines add rather than one run per cell.
        "n_clean_baseline_runs": len(
            {(c.dataset, c.retriever, c.config_id, c.iteration_budget) for c in cells}
        )
        * (cells[0].n_queries if cells else 0),
        "n_docs": n_docs,
    }
