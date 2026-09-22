"""Statistics.

Effect sizes with confidence intervals rather than bare point estimates, because the
whole claim of Study A is that a specific mechanism moves attack success by a specific
amount. A number without an interval cannot support that claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def proportion_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval.

    Wilson rather than the normal approximation because attack success rates often sit
    near 0 or 1, where the normal interval produces impossible bounds.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _z_for(confidence)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _z_for(confidence: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def two_proportion_test(s1: int, n1: int, s2: int, n2: int) -> dict[str, float]:
    """Two-proportion z-test with a difference interval.

    Used to compare a mechanism configuration against the vanilla baseline.
    """
    from statistics import NormalDist

    if n1 == 0 or n2 == 0:
        return {"diff": 0.0, "z": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))

    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    margin = 1.959963984540054 * se_unpooled
    return {
        "diff": p1 - p2,
        "z": z,
        "p_value": p_value,
        "ci_low": (p1 - p2) - margin,
        "ci_high": (p1 - p2) + margin,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of proportions.

    Conventional reading: 0.2 small, 0.5 medium, 0.8 large.
    """
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def interpret_h(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point estimate, low, high)."""
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    point = float(statistic(arr))
    if len(arr) == 1:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samples[i] = statistic(rng.choice(arr, size=len(arr), replace=True))
    alpha = (1 - confidence) / 2
    return (
        point,
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1 - alpha)),
    )


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Control the false discovery rate across the mechanism comparisons.

    The ablation makes several comparisons at once, so uncorrected p-values would
    overstate significance.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    keep = [False] * n
    threshold = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / n:
            threshold = rank
    for rank, idx in enumerate(order, start=1):
        keep[idx] = rank <= threshold
    return keep
