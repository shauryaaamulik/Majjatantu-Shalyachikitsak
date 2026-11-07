"""Failure discovery: find input regions where the model fails, and check they reproduce."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from majjatantu.types import FailureReport


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / denom)


@dataclass(frozen=True)
class FailureSlice:
    feature: int
    threshold: float  # slice is "x[:, feature] > threshold"
    n: int
    fail_rate: float
    lower_bound: float


def discover_failure_slices(
    x: np.ndarray,
    failed: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
    min_support: int = 30,
    top: int = 3,
) -> list[FailureSlice]:
    """Rank one-feature threshold slices by the Wilson lower bound of their failure rate."""
    out: list[FailureSlice] = []
    for j in range(x.shape[1]):
        for q in quantiles:
            t = float(np.quantile(x[:, j], q))
            m = x[:, j] > t
            n = int(m.sum())
            if n < min_support:
                continue
            k = int(failed[m].sum())
            out.append(FailureSlice(j, t, n, k / n, wilson_lower(k, n)))
    out.sort(key=lambda s: (s.lower_bound, s.n), reverse=True)
    return out[:top]


def reproducibility_report(
    x: np.ndarray, failed: np.ndarray, sl: FailureSlice, *, n_chunks: int = 5, min_rate: float = 0.3
) -> FailureReport:
    """A failure is eligible for repair only if it recurs across independent chunks of data."""
    in_slice = x[:, sl.feature] > sl.threshold
    hits = []
    for idx in np.array_split(np.arange(len(x)), n_chunks):
        m = in_slice[idx]
        hits.append(m.sum() > 0 and failed[idx][m].mean() >= min_rate)
    return FailureReport(
        failure_id=f"slice_f{sl.feature}_gt_{sl.threshold:.2f}",
        category="synthetic_trigger_region",
        description=f"model disagrees with clean reference when x[{sl.feature}] > {sl.threshold:.2f}",
        reproducibility_rate=float(np.mean(hits)),
        n_trials=n_chunks,
        metrics={"fail_rate": sl.fail_rate, "fail_rate_lower_bound": sl.lower_bound, "support": float(sl.n)},
    )
