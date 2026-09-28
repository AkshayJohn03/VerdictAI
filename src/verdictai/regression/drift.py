"""Statistical drift detection between two score snapshots.

Why these tests (instead of a plain t-test)?

- **Paired bootstrap CI on the mean delta** — per-item score deltas are
  bounded (0-5 scale), often bimodal and non-normal, and capability slices
  can be tiny. Bootstrap percentile CIs make no normality assumption and
  preserve the item pairing by resampling delta pairs.
- **Wilcoxon signed-rank** — a rank-based paired test, robust to outliers;
  exact enumeration for small n (<= 16 non-zero deltas), normal approximation
  with tie correction otherwise.
- **Cliff's delta** — a non-parametric effect size on the deltas (fraction of
  positive minus negative deltas), used as the guard against noise alarms:
  a CI can exclude zero on a large sample of tiny, meaningless shifts, but a
  *regression* verdict additionally requires a meaningful effect size.

A delta is ``baseline_score - candidate_score``; positive means the candidate
got worse.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from math import erf, sqrt
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .runner import ScoreSnapshot

_VERDICT_REGRESSED = "regressed"
_VERDICT_IMPROVED = "improved"
_VERDICT_NO_CHANGE = "no_change"


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean of paired deltas.

    Returns ``(ci_low, ci_high, observed_mean)``. Deterministic for a seed.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("need at least one delta")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(means, [100 * tail, 100 * (1 - tail)])
    return float(lo), float(hi), float(arr.mean())


def _normal_sf(z: float) -> float:
    """Upper-tail standard normal probability P(Z > z)."""
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def _wilcoxon_normal_variance(n: int, abs_values: Sequence[float]) -> float:
    """Variance of W+ with tie correction: n(n+1)(2n+1)/24 - sum(t^3 - t)/48."""
    var = n * (n + 1) * (2 * n + 1) / 24.0
    counts: dict[float, int] = {}
    for v in abs_values:
        counts[v] = counts.get(v, 0) + 1
    correction = sum((t**3 - t) for t in counts.values() if t > 1)
    return var - correction / 48.0


def wilcoxon_signed_rank(
    deltas: Sequence[float], *, exact_max_n: int = 16, force_normal: bool = False
) -> dict[str, Any]:
    """Wilcoxon signed-rank test on paired deltas (zeros dropped).

    Returns ``{"statistic": W+, "p_value": two-sided p, "n": used pairs,
    "method": "exact" | "normal"}``. Exact = full enumeration of sign
    assignments for n <= ``exact_max_n``; otherwise a normal approximation
    with the standard tie correction (no continuity correction).
    """
    values = [float(d) for d in deltas if d != 0.0]
    n = len(values)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0, "method": "none"}

    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = mean_rank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if values[i] > 0)

    if not force_normal and n <= exact_max_n:
        # Enumerate all 2^n sign assignments of the ranks.
        dist: dict[float, int] = {0.0: 1}
        for rank in ranks:
            nxt: dict[float, int] = {}
            for w, count in dist.items():
                nxt[w + rank] = nxt.get(w + rank, 0) + count
                nxt[w] = nxt.get(w, 0) + count
            dist = nxt
        total = 2**n
        p_ge = sum(c for w, c in dist.items() if w >= w_plus) / total
        p_le = sum(c for w, c in dist.items() if w <= w_plus) / total
        p_value = min(1.0, 2.0 * min(p_ge, p_le))
        return {"statistic": w_plus, "p_value": p_value, "n": n, "method": "exact"}

    mean = n * (n + 1) / 4.0
    var = _wilcoxon_normal_variance(n, [abs(v) for v in values])
    if var <= 0:
        p_value = 1.0 if w_plus == mean else 0.0
    else:
        z = (w_plus - mean) / sqrt(var)
        p_value = min(1.0, 2.0 * _normal_sf(abs(z)))
    return {"statistic": w_plus, "p_value": p_value, "n": n, "method": "normal"}


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: ``sum(sign(x_i - y_j)) / (n_x * n_y)`` in [-1, 1].

    +1 means every x exceeds every y; 0 means dominated overlap.

    Worked example::

        x = [1, 2, 3], y = [4, 5, 6] -> every pair x < y -> delta = -1.0
        x = [1, 2, 3], y = [2, 3, 4] -> signs sum to -5 over 9 pairs
                                       -> delta = -5/9 = -0.5556
    """
    if not x or not y:
        raise ValueError("cliffs_delta needs non-empty samples")
    ys = sorted(float(v) for v in y)
    ny = len(ys)
    total = 0
    for xi in x:
        xi = float(xi)
        less = bisect_left(ys, xi)  # y < xi -> sign +1 each
        greater = ny - bisect_right(ys, xi)  # y > xi -> sign -1 each
        total += less - greater
    return total / (len(x) * ny)


def _paired_effect(deltas: Sequence[float]) -> float:
    """Cliff's delta of deltas against zero: (n_pos - n_neg) / n."""
    if not deltas:
        return 0.0
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    return (pos - neg) / len(deltas)


class DriftReport(BaseModel):
    """Verdict + evidence from comparing two snapshots."""

    verdict: str
    n_items: int
    mean_delta: float
    ci_low: float
    ci_high: float
    threshold: float
    min_effect: float
    effect_size: float
    wilcoxon_w: float
    wilcoxon_p: float
    failing_capabilities: list[str] = Field(default_factory=list)
    improving_capabilities: list[str] = Field(default_factory=list)
    slices: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_item_ids: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        direction = {"regressed": "REGRESSION", "improved": "IMPROVEMENT"}.get(
            self.verdict, "NO SIGNIFICANT CHANGE"
        )
        lines = [
            f"Verdict: {direction}",
            f"Items compared: {self.n_items} (missing from one side: {len(self.missing_item_ids)})",
            f"Mean delta (baseline - candidate): {self.mean_delta:+.4f}",
            f"Bootstrap CI: [{self.ci_low:+.4f}, {self.ci_high:+.4f}] "
            f"(threshold {self.threshold:+.4f}, min effect {self.min_effect:.2f})",
            f"Effect size (paired Cliff's delta): {self.effect_size:+.3f}",
            f"Wilcoxon: W+={self.wilcoxon_w:.1f}, p={self.wilcoxon_p:.4f}",
        ]
        if self.failing_capabilities:
            lines.append(f"Regressed capabilities: {', '.join(self.failing_capabilities)}")
        if self.improving_capabilities:
            lines.append(f"Improved capabilities: {', '.join(self.improving_capabilities)}")
        return "\n".join(lines)


class DriftDetector:
    """Compares a baseline and candidate snapshot; emits verdict + evidence."""

    def __init__(
        self,
        *,
        n_resamples: int = 2000,
        confidence: float = 0.95,
        seed: int = 12345,
        exact_wilcoxon_max_n: int = 16,
    ):
        self.n_resamples = n_resamples
        self.confidence = confidence
        self.seed = seed
        self.exact_wilcoxon_max_n = exact_wilcoxon_max_n

    def check(
        self,
        baseline: ScoreSnapshot,
        candidate: ScoreSnapshot,
        *,
        threshold: float = 0.05,
        min_effect: float = 0.1,
    ) -> DriftReport:
        base_scores = baseline.scores_by_item()
        cand_scores = candidate.scores_by_item()
        common = sorted(set(base_scores) & set(cand_scores))
        if not common:
            raise ValueError("baseline and candidate snapshots share no item_ids")
        missing = sorted((set(base_scores) | set(cand_scores)) - set(common))
        deltas = [base_scores[i] - cand_scores[i] for i in common]
        capability = baseline.capability_of()

        ci_low, ci_high, mean_delta = paired_bootstrap_ci(
            deltas, n_resamples=self.n_resamples, confidence=self.confidence, seed=self.seed
        )
        wilcoxon = wilcoxon_signed_rank(deltas, exact_max_n=self.exact_wilcoxon_max_n)
        effect = _paired_effect(deltas)
        verdict = self._verdict_for(mean_delta, ci_low, ci_high, effect, threshold, min_effect)

        slices: dict[str, dict[str, Any]] = {}
        failing: list[str] = []
        improving: list[str] = []
        by_cap: dict[str, list[float]] = {}
        for item_id, delta in zip(common, deltas, strict=False):
            by_cap.setdefault(capability.get(item_id, "unknown"), []).append(delta)
        for cap in sorted(by_cap):
            cap_deltas = by_cap[cap]
            if len(cap_deltas) >= 2:
                c_lo, c_hi, c_mean = paired_bootstrap_ci(
                    cap_deltas,
                    n_resamples=min(self.n_resamples, 1000),
                    confidence=self.confidence,
                    seed=self.seed,
                )
                cap_effect = _paired_effect(cap_deltas)
                cap_verdict = self._verdict_for(
                    c_mean, c_lo, c_hi, cap_effect, threshold, min_effect
                )
            else:
                c_lo, c_hi = None, None
                c_mean = cap_deltas[0]
                cap_effect = _paired_effect(cap_deltas)
                cap_verdict = _VERDICT_NO_CHANGE
            slices[cap] = {
                "n": len(cap_deltas),
                "mean_delta": c_mean,
                "ci_low": c_lo,
                "ci_high": c_hi,
                "effect_size": cap_effect,
                "verdict": cap_verdict,
            }
            if cap_verdict == _VERDICT_REGRESSED:
                failing.append(cap)
            elif cap_verdict == _VERDICT_IMPROVED:
                improving.append(cap)

        return DriftReport(
            verdict=verdict,
            n_items=len(common),
            mean_delta=mean_delta,
            ci_low=ci_low,
            ci_high=ci_high,
            threshold=threshold,
            min_effect=min_effect,
            effect_size=effect,
            wilcoxon_w=wilcoxon["statistic"],
            wilcoxon_p=wilcoxon["p_value"],
            failing_capabilities=failing,
            improving_capabilities=improving,
            slices=slices,
            missing_item_ids=missing,
        )

    @staticmethod
    def _verdict_for(
        mean_delta: float,
        ci_low: float,
        ci_high: float,
        effect: float,
        threshold: float,
        min_effect: float,
    ) -> str:
        if ci_low > threshold and abs(effect) >= min_effect:
            return _VERDICT_REGRESSED
        if ci_high < -threshold and abs(effect) >= min_effect:
            return _VERDICT_IMPROVED
        return _VERDICT_NO_CHANGE
