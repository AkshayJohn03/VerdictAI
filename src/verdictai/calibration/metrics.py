"""Agreement metrics between judge and human scores.

Implemented by hand on numpy/python only (no scipy/sklearn) so every formula
is inspectable and unit-tested against hand-worked examples.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cohen_kappa(y1: Sequence[float | str], y2: Sequence[float | str]) -> float:
    """Cohen's kappa between two raters over categorical labels.

        kappa = (p_o - p_e) / (1 - p_e)

    where ``p_o`` is observed agreement and ``p_e`` the agreement expected by
    chance from the marginal distributions.

    Worked example::

        y1 = [0, 0, 1, 1], y2 = [0, 1, 1, 1]
        p_o = 3/4 = 0.75
        marginals y1: p(0)=0.5, p(1)=0.5 ; y2: p(0)=0.25, p(1)=0.75
        p_e = 0.5*0.25 + 0.5*0.75 = 0.5
        kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5

    Returns 1.0 for perfect agreement in the degenerate single-category case
    (``p_e == 1``) where kappa is undefined.
    """
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")
    if not y1:
        raise ValueError("need at least one label pair")
    categories = sorted(set(y1) | set(y2))
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    observed = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2, strict=False):
        observed[index[a], index[b]] += 1
    n = observed.sum()
    p_o = np.trace(observed) / n
    row = observed.sum(axis=1) / n
    col = observed.sum(axis=0) / n
    p_e = float(row @ col)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1.0 - p_e))


def quadratic_weighted_kappa(
    y1: Sequence[float], y2: Sequence[float], levels: Sequence[float] | None = None
) -> float:
    """Quadratic weighted kappa (QWK) over ordinal labels.

    Weights penalize disagreement by squared label distance::

        w_ij = (i - j)^2 / (K - 1)^2
        QWK = 1 - sum(w * O) / sum(w * E)

    with ``O`` the observed confusion counts and ``E`` the expected counts
    under independence (outer product of marginals / n).

    Worked example::

        y1 = [0, 1, 2, 2, 1, 0], y2 = [1, 1, 2, 0, 1, 0]
        O = [[1,1,0], [0,2,0], [1,0,1]]; row marg (2,2,2); col marg (2,3,1)
        sum(w*O) = 0.25*1 + 1*1 = 1.25
        sum(w*E) = 1.75   (outer product / 6 weighted by w)
        QWK = 1 - 1.25/1.75 = 2/7 = 0.285714...

    Labels must be numeric (ordinal); the sorted unique values define the
    ordinal levels unless ``levels`` is given explicitly.
    """
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")
    if not y1:
        raise ValueError("need at least one label pair")
    try:
        y1_f = [float(v) for v in y1]
        y2_f = [float(v) for v in y2]
    except (TypeError, ValueError) as exc:
        raise ValueError("QWK requires numeric ordinal labels") from exc
    if levels is None:
        levels_sorted = sorted(set(y1_f) | set(y2_f))
    else:
        levels_sorted = sorted({float(v) for v in levels})
    level_index = {v: i for i, v in enumerate(levels_sorted)}
    k = len(levels_sorted)
    if k < 2:
        return 1.0
    observed = np.zeros((k, k), dtype=float)
    for a, b in zip(y1_f, y2_f, strict=False):
        observed[level_index[a], level_index[b]] += 1
    n = observed.sum()
    weights = np.zeros((k, k), dtype=float)
    denom = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            weights[i, j] = (i - j) ** 2 / denom
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / n
    denom_obs = float((weights * observed).sum())
    denom_exp = float((weights * expected).sum())
    if denom_exp == 0:
        return 1.0
    return 1.0 - denom_obs / denom_exp


def _midranks(values: Sequence[float]) -> np.ndarray:
    """Average ("mid") ranks, 1-based, ties share the mean of their ranks."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = mean_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 2:
        return 0.0
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    denom = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denom == 0:
        return 0.0
    return float((xc * yc).sum() / denom)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with tie handling (midranks + Pearson).

    The classic ``1 - 6*sum(d^2)/(n(n^2-1))`` shortcut is only valid without
    ties; converting to midranks and taking the Pearson correlation of the
    ranks is the general tie-aware definition.

    Worked example (with ties in x)::

        x = [1, 2, 2, 3] -> ranks [1, 2.5, 2.5, 4]
        y = [1, 2, 3, 4] -> ranks [1, 2, 3, 4]
        rho = 4.5 / sqrt(4.5 * 5) = 3/sqrt(10) = 0.9486833...

    Returns 0.0 when either variable is constant (correlation undefined).
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return 0.0
    return _pearson(_midranks(x), _midranks(y))


def mean_absolute_error(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error: ``mean(|x_i - y_i|)``."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not x:
        raise ValueError("need at least one pair")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    return float(np.abs(xa - ya).mean())


def within_tolerance_accuracy(
    x: Sequence[float], y: Sequence[float], tolerance: float = 1.0
) -> float:
    """Fraction of pairs with ``|x_i - y_i| <= tolerance`` (inclusive)."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not x:
        raise ValueError("need at least one pair")
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    return float((np.abs(xa - ya) <= tolerance).mean())


def agreement_summary(judge: Sequence[float], human: Sequence[float]) -> dict[str, float]:
    """Standard agreement bundle used in calibration reports."""
    judge_rounded = [int(round(v)) for v in judge]
    human_rounded = [int(round(v)) for v in human]
    return {
        "n": float(len(judge)),
        "spearman": spearman(judge, human),
        "mae": mean_absolute_error(judge, human),
        "within_1": within_tolerance_accuracy(judge, human),
        "kappa": cohen_kappa(judge_rounded, human_rounded),
    }
