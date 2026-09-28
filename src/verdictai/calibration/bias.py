"""Judge bias diagnostics.

Three biases dominate LLM-judge miscalibration (see literature: Zheng et al.
2023 on position bias; Shen et al. 2023 / Wu & Aji 2023 on length/verbosity
bias; Panickssery et al. 2024 on self-preference):

- **Position bias** — detected directly from double-order pairwise runs.
- **Verbosity bias** — linear regression of judge score on response length;
  significance via a seeded permutation test on the slope (no distributional
  assumptions, works at small n).
- **Self-preference** — flags items where the judge's model family matches one
  candidate's family AND the judge preferred that side (from label metadata).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..judges.pairwise import PairwiseResult, position_bias_rate
from ..text import tokenize
from .labels import HumanLabel, HumanLabelSet


@dataclass
class VerbosityBiasReport:
    """Result of regressing judge score on response length."""

    slope: float
    intercept: float
    r: float
    p_value: float
    n: int
    significant: bool
    n_permutations: int


@dataclass
class SelfPreferenceReport:
    """Self-preference flags from label metadata."""

    rate: float
    flagged_item_ids: list[str]
    n_comparable: int


@dataclass
class BiasReport:
    """Aggregate bias diagnostics for a calibration report."""

    position_bias_rate: float | None = None
    verbosity: VerbosityBiasReport | None = None
    self_preference: SelfPreferenceReport | None = None


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope/intercept of y on x (closed form)."""
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    if sxx == 0:
        return 0.0, float(y.mean())
    slope = float((xc * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def verbosity_bias(
    judge_scores: Sequence[float],
    lengths: Sequence[float],
    *,
    n_permutations: int = 999,
    seed: int = 0,
    alpha: float = 0.05,
) -> VerbosityBiasReport:
    """Regression of judge score on response length + permutation p-value.

    The p-value is the two-sided permutation test of the slope: permute the
    scores against the fixed lengths ``n_permutations`` times and count how
    often the permuted slope is at least as extreme as the observed one.
    Deterministic for a fixed ``seed``.
    """
    x = np.asarray(lengths, dtype=float)
    y = np.asarray(judge_scores, dtype=float)
    if x.size != y.size:
        raise ValueError("judge_scores and lengths must have the same length")
    if x.size < 3:
        raise ValueError("verbosity bias needs at least 3 points")
    slope, intercept = _ols(x, y)
    from .metrics import _pearson

    r = _pearson(x.tolist(), y.tolist())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(y)
        perm_slope, _ = _ols(x, permuted)
        if abs(perm_slope) >= abs(slope) - 1e-15:
            exceed += 1
    p_value = (exceed + 1) / (n_permutations + 1)
    return VerbosityBiasReport(
        slope=slope,
        intercept=intercept,
        r=r,
        p_value=p_value,
        n=int(x.size),
        significant=bool(p_value < alpha),
        n_permutations=n_permutations,
    )


def self_preference(
    labels: Sequence[HumanLabel],
    *,
    judge_family_key: str = "judge_family",
    a_family_key: str = "response_a_family",
    b_family_key: str = "response_b_family",
) -> SelfPreferenceReport:
    """Detect self-preference from family metadata on pairwise labels.

    An item is *comparable* when the judge's family matches one side's family
    and a decisive preference exists. It is *flagged* when the judge preferred
    the side sharing its own family — the classic self-preference signature.
    """
    flagged: list[str] = []
    comparable = 0
    for label in labels:
        judge_family = label.meta.get(judge_family_key)
        a_family = label.meta.get(a_family_key)
        b_family = label.meta.get(b_family_key)
        if not judge_family or label.preferred not in ("a", "b"):
            continue
        own_side = None
        if judge_family == a_family:
            own_side = "a"
        elif judge_family == b_family:
            own_side = "b"
        if own_side is None:
            continue
        comparable += 1
        if label.preferred == own_side:
            flagged.append(label.item_id)
    rate = len(flagged) / comparable if comparable else 0.0
    return SelfPreferenceReport(rate=rate, flagged_item_ids=flagged, n_comparable=comparable)


def summarize_bias(
    label_set: HumanLabelSet,
    pairwise_results: Sequence[PairwiseResult] | None = None,
    *,
    verbosity_seed: int = 0,
) -> BiasReport:
    """Assemble a :class:`BiasReport` from a label set (and pairwise runs).

    Verbosity bias is computed over labels with a judge score, using the
    response_a token count (or prompt token count when no response is stored)
    as the length proxy.
    """
    report = BiasReport()
    if pairwise_results is not None:
        report.position_bias_rate = position_bias_rate(pairwise_results)

    scored = [lab for lab in label_set.labels if lab.judge_score is not None]
    if len(scored) >= 3:
        lengths = [len(tokenize(lab.response_a or lab.prompt)) for lab in scored]
        scores = [float(lab.judge_score) for lab in scored]  # type: ignore[misc]
        report.verbosity = verbosity_bias(scores, lengths, seed=verbosity_seed)

    report.self_preference = self_preference(label_set.labels)
    return report
