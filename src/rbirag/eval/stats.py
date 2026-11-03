"""Bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel


class Estimate(BaseModel):
    mean: float
    low: float
    high: float
    n: int

    def fmt(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> Estimate:
    """Percentile bootstrap CI of ``statistic`` (default: mean) over per-item ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return Estimate(mean=float("nan"), low=float("nan"), high=float("nan"), n=0)
    stat = statistic or (lambda a: float(np.mean(a)))
    point = stat(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = np.array([stat(arr[row]) for row in idx])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(mean=float(point), low=float(low), high=float(high), n=int(arr.size))


def bootstrap_ratio_ci(
    numer: Sequence[float],
    denom: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Estimate:
    """CI for ``sum(numer) / sum(denom)`` resampling items jointly (e.g. precision)."""
    num = np.asarray(numer, dtype=np.float64)
    den = np.asarray(denom, dtype=np.float64)
    if num.size == 0 or den.sum() == 0:
        return Estimate(mean=float("nan"), low=float("nan"), high=float("nan"), n=int(num.size))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_resamples, num.size))
    d = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        samples = num[idx].sum(axis=1) / d
    samples = samples[d > 0]
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(
        mean=float(num.sum() / den.sum()), low=float(low), high=float(high), n=int(num.size)
    )
