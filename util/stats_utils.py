from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap on the mean. Returns (mean, low, high)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return float(arr.mean()), low, high


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap on paired mean-difference (a-b). Returns (mean_diff, low, high)."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError(f"paired sizes differ: {aa.size} vs {bb.size}")
    if aa.size == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = aa - bb
    if diffs.size == 1:
        v = float(diffs[0])
        return v, v, v
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        float(diffs.mean()),
        float(np.quantile(boot, alpha)),
        float(np.quantile(boot, 1.0 - alpha)),
    )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value on paired diffs, using (1+hits)/(1+n) so p>0."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError(f"paired sizes differ: {aa.size} vs {bb.size}")
    diffs = aa - bb
    n = diffs.size
    if n == 0:
        return float("nan")
    observed = float(np.abs(diffs.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(rng_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    hits = int((perm_means >= observed - 1e-12).sum())
    return (1 + hits) / (1 + n_perm)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired-sample Cohen's d: mean(diff) / sd(diff), unbiased sd."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError(f"paired sizes differ: {aa.size} vs {bb.size}")
    diffs = aa - bb
    if diffs.size < 2:
        return float("nan")
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / sd)


def holm_correction(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down FWER correction, returned in original order, clipped to [0,1]."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def pairwise_win_rate_matrix(
    per_model_paired: Dict[str, Sequence[float]],
) -> Dict[str, Dict[str, float]]:
    """Fraction of paired seeds where row-model > col-model (ties=0.5). Seed order must be aligned."""
    models = list(per_model_paired.keys())
    arrays = {m: np.asarray(per_model_paired[m], dtype=float) for m in models}
    matrix: Dict[str, Dict[str, float]] = {m: {} for m in models}
    for row in models:
        for col in models:
            if row == col:
                matrix[row][col] = float("nan")
                continue
            a = arrays[row]
            b = arrays[col]
            n = min(a.size, b.size)
            if n == 0:
                matrix[row][col] = float("nan")
                continue
            wins = float((a[:n] > b[:n]).sum())
            ties = float((a[:n] == b[:n]).sum())
            matrix[row][col] = (wins + 0.5 * ties) / n
    return matrix


def format_ci(mean: float, low: float, high: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
