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


class PairedComparison(BaseModel):
    """Difference ``mean(a - b)`` over paired per-question values."""

    diff: float
    low: float
    high: float
    p_value: float  # two-sided paired randomisation (sign-flip) test
    n: int
    wins: int  # questions where a > b
    losses: int  # questions where a < b


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 13,
) -> PairedComparison:
    """Bootstrap CI of the mean paired difference plus a sign-flip randomisation test
    (the standard significance test for comparing two IR systems on the same queries)."""
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if x.size == 0:
        nan = float("nan")
        return PairedComparison(diff=nan, low=nan, high=nan, p_value=nan, n=0, wins=0, losses=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    boot = x[idx].mean(axis=1)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    observed = abs(float(x.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, x.size))
    flipped = np.abs((signs * x).mean(axis=1))
    p = (np.sum(flipped >= observed - 1e-12) + 1) / (n_resamples + 1)
    return PairedComparison(
        diff=float(x.mean()),
        low=float(low),
        high=float(high),
        p_value=float(p),
        n=int(x.size),
        wins=int(np.sum(x > 1e-12)),
        losses=int(np.sum(x < -1e-12)),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), in input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
