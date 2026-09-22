"""Evaluation protocols.

The design here answers the objection that would otherwise sink Study B: that a detector
trained on self-generated attacks has merely memorised them.

Four protocols, in decreasing order of how much they should be trusted:

* ``leave_one_attack_out``  train on two attacks, test on a third never seen. This is the
  headline result and the only number that should lead the write-up.
* ``cross_dataset``  train on one dataset's traces, test on another's.
* ``feature_ablation``  drop one feature family at a time, to show no family carries the
  whole detector and there is no trivial shortcut.
* ``random_split``  ordinary stratified split. Reported only as an upper bound, clearly
  labelled as optimistic.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from argus.detect.base import Detector, DetectorResult, compute_metrics
from argus.detect.models import build_detector
from argus.features.schema import FEATURE_FAMILIES


def _fit_score(
    detector_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    **params: Any,
) -> tuple[np.ndarray, Detector, float]:
    det = build_detector(detector_name, feature_names=feature_names, **params)
    det.fit(X_train, y_train)
    t0 = time.perf_counter()
    scores = det.score(X_test)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(len(X_test), 1)
    return scores, det, latency_ms


def evaluate_detector(
    detector_name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
    **params: Any,
) -> DetectorResult:
    """Ordinary stratified split. Optimistic by construction, reported as an upper bound."""
    from sklearn.model_selection import train_test_split

    y = meta["y"].to_numpy()
    feature_names = list(X.columns)
    idx = np.arange(len(X))
    stratify = y if len(np.unique(y)) > 1 else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)

    scores, det, latency = _fit_score(
        detector_name, X.to_numpy()[tr], y[tr], X.to_numpy()[te], feature_names, **params
    )
    m = compute_metrics(y[te], scores)

    return DetectorResult(
        detector=detector_name,
        protocol="random_split (optimistic upper bound)",
        n_train=len(tr),
        n_test=len(te),
        n_positive=int(y[te].sum()),
        detect_latency_ms=latency,
        inference_overhead_x=TRACE_DETECTOR_OVERHEAD_X,
        feature_importance=det.importance(),
        **m,
    )


#: Extra generator passes the trace detector needs per query. It reads spans a run has
#: already emitted, so the answer is none. This is the number the whole of Study B is
#: arguing about, against RAGuard's k+1, so it is a named constant rather than a literal
#: repeated at four call sites, where it had been left at a placeholder 0.0 that read as
#: "not measured" rather than "measured, and zero".
TRACE_DETECTOR_OVERHEAD_X = 0.0

def score_trace_subset_loao(
    detector_name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    trace_ids: list[str],
    held_out_attack: str,
    seed: int = 42,
    **params: Any,
) -> DetectorResult:
    """Score one named set of traces, holding out their attack family from training.

    Exists so the trace detector can be put on the same footing as the content-based
    defences in ``argus baselines compare``. Those defences are expensive — the
    leave-one-out counterfactual re-answers the question once per retrieved document —
    so they are evaluated on a few hundred traces from a single attack group. Reporting
    the detector's full leave-one-attack-out figure next to them compared 50,449 traces
    spanning three attacks against 450 traces spanning one, under a different protocol,
    and the resulting table invited a straight column-wise reading that nothing in it
    supported.

    This trains on every trace outside the held-out attack family *and* outside the
    evaluated set, then scores exactly the traces the baselines scored. Same rows, same
    labels, still no sight of the attack family at training time.
    """
    ids = set(trace_ids)
    id_col = meta["trace_id"].astype(str).to_numpy()
    attack_col = meta["attack"].to_numpy()

    in_subset = np.isin(id_col, list(ids))
    is_held = attack_col == held_out_attack
    train_mask = ~in_subset & ~is_held
    if train_mask.sum() < 10 or in_subset.sum() < 5:
        raise ValueError(
            f"not enough rows to score the subset: {int(train_mask.sum())} train, "
            f"{int(in_subset.sum())} test"
        )

    y = meta["y"].to_numpy()
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[in_subset])) < 2:
        raise ValueError("subset or training pool has a single class; AUC undefined")

    Xv = X.to_numpy()
    scores, det, latency = _fit_score(
        detector_name, Xv[train_mask], y[train_mask], Xv[in_subset], list(X.columns), **params
    )
    m = compute_metrics(y[in_subset], scores)
    return DetectorResult(
        detector=detector_name,
        protocol=f"loao on the baseline sample (held out {held_out_attack})",
        n_train=int(train_mask.sum()),
        n_test=int(in_subset.sum()),
        n_positive=int(y[in_subset].sum()),
        detect_latency_ms=latency,
        inference_overhead_x=TRACE_DETECTOR_OVERHEAD_X,
        feature_importance=det.importance(),
        **m,
    )


#: A fold with fewer positives than this cannot support a meaningful AUC, so it is
#: reported as skipped rather than pooled. The corpus-poisoning fold of the first run had
#: 46 positives in 16,459 rows because the attack never retrieved, and pooling it pulled
#: every detector below chance.
MIN_FOLD_POSITIVES = 50


def leave_one_attack_out(
    detector_name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    seed: int = 42,
    min_positives: int = MIN_FOLD_POSITIVES,
    **params: Any,
) -> DetectorResult:
    """Train on all attacks but one, test on the held-out attack. The headline metric.

    Two things this gets right that the first implementation did not.

    **No leakage from the held-out attack.** The benign pool was previously defined as
    ``attack in {none, ""} or y == 0``, so every *unsuccessful* run of the held-out
    attack counted as benign and was then split 70/30 into train and test. The detector
    therefore saw the held-out attack's behavioural distribution during training, which
    is exactly what the protocol exists to prevent. Negatives are now partitioned by
    origin: genuinely clean traces are shared across folds, attacked-but-negative traces
    follow their own attack into either train or test.

    **Folds without enough positives are skipped, not pooled.** A fold whose attack never
    retrieves contributes noise with the authority of a metric.
    """
    attacks = sorted(a for a in meta["attack"].unique() if a not in ("none", ""))
    if len(attacks) < 2:
        raise ValueError(
            f"leave-one-attack-out needs at least two attacks, found {attacks}. "
            "Run the grid with more attacks first."
        )

    y = meta["y"].to_numpy()
    Xv = X.to_numpy()
    attack_col = meta["attack"].to_numpy()
    feature_names = list(X.columns)
    rng = np.random.default_rng(seed)

    # Genuinely clean traffic: never attacked. These are the only negatives that may be
    # shared between train and test, and they are what makes the false-positive rate mean
    # anything operationally.
    is_clean = np.isin(attack_col, ["none", ""])
    clean_idx = np.flatnonzero(is_clean)

    per_split: dict[str, Any] = {}
    all_true: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    latencies: list[float] = []
    importances: list[dict[str, float]] = []
    skipped: list[str] = []

    for held_out in attacks:
        is_held = (attack_col == held_out)

        shuffled = rng.permutation(clean_idx)
        cut = int(len(shuffled) * 0.7)
        clean_train, clean_test = shuffled[:cut], shuffled[cut:]

        # Everything belonging to the held-out attack goes to test, both classes.
        # Everything belonging to the other attacks goes to train, both classes.
        train_idx = np.union1d(np.flatnonzero(~is_held & ~is_clean), clean_train)
        test_idx = np.union1d(np.flatnonzero(is_held), clean_test)

        n_pos = int(y[test_idx].sum())
        if len(train_idx) < 10 or len(test_idx) < 5:
            per_split[held_out] = {"skipped": "not enough samples"}
            skipped.append(held_out)
            continue
        if n_pos < min_positives:
            per_split[held_out] = {
                "skipped": (
                    f"only {n_pos} positives (need >= {min_positives}); this attack "
                    "produced almost no compromised traces, so the fold cannot support "
                    "a meaningful AUC"
                ),
                "n_test": len(test_idx),
                "n_positive": n_pos,
            }
            skipped.append(held_out)
            continue
        if len(np.unique(y[train_idx])) < 2:
            per_split[held_out] = {"skipped": "training fold has a single class"}
            skipped.append(held_out)
            continue

        scores, det, latency = _fit_score(
            detector_name, Xv[train_idx], y[train_idx], Xv[test_idx], feature_names, **params
        )
        m = compute_metrics(y[test_idx], scores)
        per_split[held_out] = {
            **m,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_positive": n_pos,
        }
        all_true.append(y[test_idx])
        all_scores.append(scores)
        latencies.append(latency)
        if det.importance():
            importances.append(det.importance())

    if not all_true:
        raise ValueError(
            "every leave-one-attack-out fold was skipped. Either the grid is too small, "
            "or the attacks are not producing compromised traces. Run "
            "`argus validate attacks` to check the retrieval condition of each attack."
        )

    pooled = compute_metrics(np.concatenate(all_true), np.concatenate(all_scores))
    mean_importance: dict[str, float] = {}
    if importances:
        for name in importances[0]:
            mean_importance[name] = float(np.mean([imp.get(name, 0.0) for imp in importances]))
        mean_importance = dict(sorted(mean_importance.items(), key=lambda kv: -kv[1]))

    return DetectorResult(
        detector=detector_name,
        protocol="leave_one_attack_out (headline)",
        n_train=int(np.mean([v.get("n_train", 0) for v in per_split.values() if "n_train" in v])),
        n_test=int(sum(len(t) for t in all_true)),
        n_positive=int(sum(int(t.sum()) for t in all_true)),
        detect_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        inference_overhead_x=TRACE_DETECTOR_OVERHEAD_X,
        per_split=per_split,
        feature_importance=mean_importance,
        meta={"folds_evaluated": len(all_true), "folds_skipped": skipped},
        **pooled,
    )


def cross_dataset_evaluation(
    detector_name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
    **params: Any,
) -> DetectorResult:
    """Train on one dataset, test on another."""
    y = meta["y"].to_numpy()
    Xv = X.to_numpy()
    tr = np.flatnonzero((meta["dataset"] == train_dataset).to_numpy())
    te = np.flatnonzero((meta["dataset"] == test_dataset).to_numpy())

    if len(tr) < 10 or len(te) < 5:
        raise ValueError(
            f"not enough traces for cross-dataset transfer "
            f"({train_dataset}: {len(tr)}, {test_dataset}: {len(te)})"
        )

    scores, det, latency = _fit_score(
        detector_name, Xv[tr], y[tr], Xv[te], list(X.columns), **params
    )
    m = compute_metrics(y[te], scores)
    return DetectorResult(
        detector=detector_name,
        protocol=f"cross_dataset ({train_dataset} -> {test_dataset})",
        n_train=len(tr),
        n_test=len(te),
        n_positive=int(y[te].sum()),
        detect_latency_ms=latency,
        inference_overhead_x=TRACE_DETECTOR_OVERHEAD_X,
        feature_importance=det.importance(),
        **m,
    )


def feature_ablation(
    detector_name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    protocol: str = "loao",
    seed: int = 42,
    **params: Any,
) -> pd.DataFrame:
    """Drop one feature family at a time and re-evaluate.

    If removing any single family collapses performance, the detector is leaning on a
    shortcut and the result should be treated with suspicion.
    """
    runner = leave_one_attack_out if protocol == "loao" else evaluate_detector

    def _run(frame: pd.DataFrame) -> DetectorResult:
        return runner(detector_name, frame, meta, seed=seed, **params)

    baseline = _run(X)
    rows = [
        {
            "ablation": "none (all features)",
            "n_features": X.shape[1],
            "roc_auc": baseline.roc_auc,
            "f1": baseline.f1,
            "fpr_at_95_tpr": baseline.fpr_at_95_tpr,
            "delta_auc": 0.0,
        }
    ]

    for family, feats in FEATURE_FAMILIES.items():
        keep = [c for c in X.columns if c not in feats]
        if len(keep) < 3:
            continue
        try:
            res = _run(X[keep])
        except ValueError:
            continue
        rows.append(
            {
                "ablation": f"without {family}",
                "n_features": len(keep),
                "roc_auc": res.roc_auc,
                "f1": res.f1,
                "fpr_at_95_tpr": res.fpr_at_95_tpr,
                "delta_auc": round(res.roc_auc - baseline.roc_auc, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("delta_auc")
