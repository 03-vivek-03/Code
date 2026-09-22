"""Detector models.

Deliberately ordered from simple to complex, so the marginal value of complexity is
demonstrated rather than assumed:

1. ``rules``    interpretable thresholds. Establishes the floor. If this does well, the
                finding is that trace features carry an obvious signal, which is itself
                worth reporting.
2. ``iforest``  Isolation Forest trained on benign traces only. This is the realistic
                deployment setting: a defender usually has normal traffic and no
                labelled attacks.
3. ``gbdt``     gradient boosted trees, supervised. Establishes the ceiling.

XGBoost is used when available and scikit-learn's HistGradientBoosting otherwise, so the
platform never hard-depends on an optional package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from argus.detect.base import Detector


class RuleDetector(Detector):
    """Interpretable threshold rules over a handful of features.

    Each rule encodes a stated hypothesis about what compromise looks like in a trace, so
    a positive result is explainable and a negative result is informative.

    Why the scoring was rewritten
    -----------------------------

    The first version scored a trace as the *fraction of six rules that fired*, where
    each rule fired above the 90th percentile of benign traffic. Evaluation then
    thresholded that at 0.5, which required three independent 90th-percentile rules to
    trip on the same trace. That essentially never happens: measured recall was 0.000 in
    every leave-one-attack-out fold and pooled AUC was 0.408, below chance. The floor was
    not weak, it was unreachable, and it made the "simple baseline" row of the comparison
    meaningless.

    Two changes fix it. The score is now continuous — each rule contributes how far the
    trace sits into the suspicious tail of the benign distribution, not a binary hit — so
    ROC-AUC is well defined. And the operating point is calibrated at fit time so that
    0.5 corresponds to a chosen false-positive rate on benign traffic, which is what the
    fixed 0.5 threshold downstream assumes.
    """

    name = "rules"
    unsupervised = True

    #: (feature, direction). direction 1 means high values are suspicious.
    RULES: list[tuple[str, int]] = [
        ("retr_score_top_gap", 1),      # one passage dominating the ranking
        ("retr_score_skew", 1),         # a heavy upper tail in the score distribution
        ("query_drift_total", 1),       # the query wandered far from the question
        ("late_context_fraction", 1),   # evidence arrived only at the end
        ("answer_len_zscore", -1),      # unusually terse answers
        ("repeat_doc_ratio", 1),        # the agent kept resurfacing the same documents
    ]

    #: Benign false-positive rate the 0.5 operating point is calibrated to.
    TARGET_FPR = 0.05

    def __init__(self, target_fpr: float = TARGET_FPR, **params: Any) -> None:
        super().__init__(target_fpr=target_fpr, **params)
        #: feature -> (direction, benign sample used to build the empirical CDF)
        self._reference: dict[str, tuple[int, np.ndarray]] = {}
        self._operating_point = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RuleDetector:
        # Calibrate on benign traces only when labels are available, which keeps the
        # rules honest about what "normal" means.
        Xb = X[y == 0] if y is not None and (y == 0).any() else X
        idx = {name: i for i, name in enumerate(self.feature_names)}

        for feat, direction in self.RULES:
            if feat not in idx:
                continue
            col = np.asarray(Xb[:, idx[feat]], dtype=float)
            # Store a sorted reference sample; the score is this feature's empirical CDF
            # position, oriented so that 1.0 is maximally suspicious.
            self._reference[feat] = (direction, np.sort(col))

        raw = self._raw_score(Xb)
        # Map the benign (1 - target_fpr) quantile onto 0.5 so that thresholding at 0.5
        # yields approximately the intended false-positive rate.
        self._operating_point = float(
            np.quantile(raw, 1.0 - self.params.get("target_fpr", self.TARGET_FPR))
        )
        self._fitted = True
        return self

    def _raw_score(self, X: np.ndarray) -> np.ndarray:
        idx = {name: i for i, name in enumerate(self.feature_names)}
        if not self._reference:
            return np.zeros(len(X), dtype=float)

        total = np.zeros(len(X), dtype=float)
        for feat, (direction, ref) in self._reference.items():
            col = np.asarray(X[:, idx[feat]], dtype=float)
            # Empirical CDF position of each value within the benign reference sample.
            pos = np.searchsorted(ref, col, side="right") / max(len(ref), 1)
            total += pos if direction > 0 else (1.0 - pos)
        return total / len(self._reference)

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_score(X)
        op = self._operating_point
        # Piecewise-linear rescale putting the calibrated operating point at 0.5, so the
        # score stays in [0, 1] and ordering is preserved (AUC is unaffected).
        below = raw / max(op, 1e-9) * 0.5
        above = 0.5 + (raw - op) / max(1.0 - op, 1e-9) * 0.5
        return np.clip(np.where(raw <= op, below, above), 0.0, 1.0)

    def importance(self) -> dict[str, float]:
        return {f: 1.0 / max(len(self._reference), 1) for f in self._reference}


class IsolationForestDetector(Detector):
    """Isolation Forest over benign traces.

    Trained on normal traffic alone, so it needs no labelled attacks. This is the variant
    a real deployment could actually adopt on day one.
    """

    name = "iforest"
    unsupervised = True

    def __init__(self, n_estimators: int = 300, contamination: float = 0.1, seed: int = 42, **params: Any) -> None:
        super().__init__(n_estimators=n_estimators, contamination=contamination, seed=seed, **params)
        self._model = None
        self._scaler = None
        self._lo = 0.0
        self._hi = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IsolationForestDetector:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        Xb = X[y == 0] if y is not None and (y == 0).any() else X

        self._scaler = StandardScaler().fit(Xb)
        self._model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            contamination=self.params["contamination"],
            random_state=self.params["seed"],
            n_jobs=-1,
        ).fit(self._scaler.transform(Xb))

        # Calibrate to [0, 1] using the training distribution so the 0.5 threshold means
        # something consistent across runs.
        raw = -self._model.score_samples(self._scaler.transform(Xb))
        self._lo, self._hi = float(raw.min()), float(raw.max())
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = -self._model.score_samples(self._scaler.transform(X))
        span = max(self._hi - self._lo, 1e-9)
        return np.clip((raw - self._lo) / span, 0.0, 1.0)


class GBDTDetector(Detector):
    """Supervised gradient boosted trees. The performance ceiling."""

    name = "gbdt"
    unsupervised = False

    def __init__(self, n_estimators: int = 300, max_depth: int = 5, learning_rate: float = 0.08, seed: int = 42, **params: Any) -> None:
        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, seed=seed, **params,
        )
        self._model = None
        self._backend = ""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GBDTDetector:
        if y is None:
            raise ValueError("GBDTDetector is supervised and needs labels")

        if len(np.unique(y)) < 2:
            # A degenerate split still has to produce a usable object, otherwise a
            # leave-one-attack-out fold can crash the whole evaluation.
            self._model, self._backend = None, "constant"
            self._const = float(y[0]) if len(y) else 0.0
            self._fitted = True
            return self

        try:
            from xgboost import XGBClassifier

            self._model = XGBClassifier(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["seed"],
                eval_metric="logloss",
                n_jobs=-1,
            )
            self._backend = "xgboost"
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._model = HistGradientBoostingClassifier(
                max_iter=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                random_state=self.params["seed"],
            )
            self._backend = "sklearn"

        self._model.fit(X, y)
        self._fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "constant":
            return np.full(len(X), self._const, dtype=float)
        return self._model.predict_proba(X)[:, 1]

    def importance(self) -> dict[str, float]:
        if self._model is None or not self.feature_names:
            return {}
        if hasattr(self._model, "feature_importances_"):
            vals = np.asarray(self._model.feature_importances_, dtype=float)
        else:  # pragma: no cover - permutation path is slow, used only when needed
            return {}
        total = vals.sum() or 1.0
        return {
            name: float(v / total)
            for name, v in sorted(
                zip(self.feature_names, vals), key=lambda kv: -kv[1]
            )
        }


DETECTORS: dict[str, type[Detector]] = {
    "rules": RuleDetector,
    "iforest": IsolationForestDetector,
    "gbdt": GBDTDetector,
}


def build_detector(name: str, feature_names: list[str] | None = None, **params: Any) -> Detector:
    key = name.lower()
    if key not in DETECTORS:
        raise KeyError(f"unknown detector '{name}', expected one of {sorted(DETECTORS)}")
    det = DETECTORS[key](**params)
    det.feature_names = list(feature_names or [])
    return det
