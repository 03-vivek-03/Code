"""Detector interface and metrics.

The headline metric of this project is not accuracy. It is generalisation to an attack
the detector has never seen, reported together with the false-positive rate on clean
traffic. Both are defined here so that no part of the codebase can quietly report a
friendlier number instead.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetectorResult:
    """Evaluation output for one detector under one protocol."""

    detector: str
    protocol: str = ""
    n_train: int = 0
    n_test: int = 0
    n_positive: int = 0

    roc_auc: float = 0.0
    pr_auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    #: The operational metric. A detector that fires constantly on clean traffic is
    #: useless no matter how high its recall.
    fpr_at_95_tpr: float = 1.0
    tpr_at_1pct_fpr: float = 0.0
    threshold: float = 0.5

    #: Runtime cost relative to a single agent pass. The whole argument for trace-based
    #: detection is that this number stays near zero.
    inference_overhead_x: float = 0.0
    detect_latency_ms: float = 0.0

    per_split: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"{self.detector:<20s} {self.protocol:<22s} "
            f"AUC={self.roc_auc:.3f}  F1={self.f1:.3f}  "
            f"FPR@95TPR={self.fpr_at_95_tpr:.3f}  n={self.n_test}"
        )


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Standard metrics plus the two operational ones."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.5,
        "pr_auc": float(y_true.mean()) if len(y_true) else 0.0,
        "fpr_at_95_tpr": 1.0,
        "tpr_at_1pct_fpr": 0.0,
    }

    # Degenerate single-class splits cannot yield a meaningful AUC.
    if len(np.unique(y_true)) < 2:
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    out["pr_auc"] = float(average_precision_score(y_true, scores))

    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, 0.95, side="left")
    out["fpr_at_95_tpr"] = float(fpr[min(idx, len(fpr) - 1)])
    idx2 = np.searchsorted(fpr, 0.01, side="right") - 1
    out["tpr_at_1pct_fpr"] = float(tpr[max(idx2, 0)])
    return out


class Detector(ABC):
    """Base class for compromise detectors."""

    name: str = "base"
    #: True when the detector trains on benign traces only, which is the realistic
    #: deployment case since defenders rarely hold labelled attacks.
    unsupervised: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.feature_names: list[str] = []
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Detector: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher means more likely compromised. Range should be roughly [0, 1]."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def importance(self) -> dict[str, float]:
        return {}

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        meta = path.with_suffix(".meta.json")
        meta.write_text(
            json.dumps(
                {
                    "detector": self.name,
                    "params": {k: str(v) for k, v in self.params.items()},
                    "feature_names": self.feature_names,
                    "unsupervised": self.unsupervised,
                },
                indent=2,
            )
        )
        return path

    @staticmethod
    def load(path: str | Path) -> Detector:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
