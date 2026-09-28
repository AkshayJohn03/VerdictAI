"""Isotonic calibration via pool-adjacent-violators (PAV) — pure Python.

Why isotonic regression instead of Platt scaling? Platt scaling fits a single
parametric sigmoid, which assumes a specific logistic miscalibration shape and
can only compress/stretch scores globally. Judges miscalibrate in messier,
still-monotone ways (saturation at 4-5, dead zones at 0-1, kinks). PAV fits
the best *monotone* step function directly from the data — non-parametric, no
shape assumptions, O(n) with a stack — at the cost of needing more labels and
potentially overfitting small sets (mitigated here by the monotonicity prior
and by reporting before/after agreement).

Out-of-range inputs are clamped to the end-block values; queries falling in
gaps between fitted blocks are linearly interpolated (the fit itself stays a
step function on observed ranges).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from .bias import BiasReport
from .labels import HumanLabelSet
from .metrics import agreement_summary


@dataclass
class IsotonicBlock:
    """One constant-value block of the fitted monotone function."""

    x_start: float
    x_end: float
    value: float
    weight: float


class IsotonicFit:
    """Monotone non-decreasing fit produced by PAV."""

    def __init__(self, blocks: list[IsotonicBlock]):
        if not blocks:
            raise ValueError("IsotonicFit needs at least one block")
        self.blocks = blocks

    @classmethod
    def fit(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> IsotonicFit:
        """Fit g with g(x_i) ~= y_i, g non-decreasing, minimizing weighted SSE."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if not x:
            raise ValueError("need at least one point")
        w = [1.0] * len(x) if weights is None else [float(v) for v in weights]
        if any(v < 0 for v in w):
            raise ValueError("weights must be non-negative")
        order = sorted(range(len(x)), key=lambda i: x[i])
        # Stack entries: [value, weight, x_start, x_end]
        stack: list[list[float]] = []
        for i in order:
            xi, yi, wi = float(x[i]), float(y[i]), w[i]
            stack.append([yi, wi, xi, xi])
            while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
                hi = stack.pop()
                lo = stack.pop()
                total_w = lo[1] + hi[1]
                merged_value = (lo[0] * lo[1] + hi[0] * hi[1]) / total_w if total_w > 0 else hi[0]
                stack.append([merged_value, total_w, lo[2], hi[3]])
        blocks = [
            IsotonicBlock(x_start=b[2], x_end=b[3], value=b[0], weight=b[1]) for b in stack
        ]
        return cls(blocks)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def predict(self, x: float) -> float:
        """Monotone mapping value for ``x`` (interpolated/clamped out of range)."""
        starts = [b.x_start for b in self.blocks]
        i = bisect_right(starts, x) - 1
        if i < 0:
            return self.blocks[0].value
        block = self.blocks[i]
        if x <= block.x_end:
            return block.value
        if i == len(self.blocks) - 1:
            return block.value  # beyond the last block: clamp
        nxt = self.blocks[i + 1]
        gap = nxt.x_start - block.x_end
        if gap <= 0:
            return block.value
        t = (x - block.x_end) / gap
        return block.value + t * (nxt.value - block.value)

    def predict_many(self, xs: Sequence[float]) -> list[float]:
        return [self.predict(x) for x in xs]


class CalibrationLayer:
    """Maps raw judge scores onto the human score scale via isotonic fit."""

    def __init__(self, fit: IsotonicFit, n_pairs: int):
        self.fit = fit
        self.n_pairs = n_pairs

    @classmethod
    def fit(
        cls,
        judge_scores: Sequence[float],
        human_scores: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> CalibrationLayer:
        if len(judge_scores) != len(human_scores):
            raise ValueError("judge_scores and human_scores must have the same length")
        if len(judge_scores) < 2:
            raise ValueError("isotonic calibration needs at least 2 labeled pairs")
        fit = IsotonicFit.fit(list(judge_scores), list(human_scores), weights)
        return cls(fit=fit, n_pairs=len(judge_scores))

    def calibrate(self, score: float) -> float:
        return self.fit.predict(score)

    def calibrate_many(self, scores: Sequence[float]) -> list[float]:
        return self.fit.predict_many(scores)


def render_calibration_report(
    label_set: HumanLabelSet,
    layer: CalibrationLayer | None,
    bias: BiasReport | None = None,
    *,
    title: str = "VerdictAI judge calibration report",
) -> str:
    """Markdown report: agreement before/after calibration, bias table,
    and a recommended judge configuration."""
    paired = label_set.paired()
    if not paired:
        return f"# {title}\n\nNo paired (judge + human) labels available.\n"
    judge = [float(lab.judge_score) for lab in paired]  # type: ignore[misc]
    human = [float(lab.human_score) for lab in paired]  # type: ignore[misc]

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- Paired labels: **{len(paired)}**")
    lines.append(f"- Isotonic blocks in fit: **{layer.fit.n_blocks if layer else 'n/a'}**")
    lines.append("")

    before = agreement_summary(judge, human)
    lines.append("## Agreement (before calibration)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key in ("spearman", "mae", "within_1", "kappa"):
        lines.append(f"| {key} | {before[key]:.4f} |")
    lines.append("")

    if layer is not None:
        calibrated = layer.calibrate_many(judge)
        after = agreement_summary(calibrated, human)
        lines.append("## Agreement (after isotonic calibration)")
        lines.append("")
        lines.append("| metric | before | after |")
        lines.append("|---|---|---|")
        for key in ("spearman", "mae", "within_1", "kappa"):
            lines.append(f"| {key} | {before[key]:.4f} | {after[key]:.4f} |")
        lines.append("")

    lines.append("## Bias diagnostics")
    lines.append("")
    lines.append("| bias | value |")
    lines.append("|---|---|")
    if bias is not None and bias.position_bias_rate is not None:
        lines.append(f"| position bias rate (pairwise) | {bias.position_bias_rate:.3f} |")
    if bias is not None and bias.verbosity is not None:
        v = bias.verbosity
        lines.append(
            f"| verbosity slope (score/100 tokens) | {v.slope * 100:.4f} "
            f"(p={v.p_value:.3f}, n={v.n}) |"
        )
    if bias is not None and bias.self_preference is not None:
        sp = bias.self_preference
        lines.append(
            f"| self-preference rate | {sp.rate:.3f} ({len(sp.flagged_item_ids)}"
            f"/{sp.n_comparable} comparable) |"
        )
    if bias is None or (
        bias.position_bias_rate is None and bias.verbosity is None and bias.self_preference is None
    ):
        lines.append("| (no bias diagnostics supplied) | n/a |")
    lines.append("")

    lines.append("## Recommended judge configuration")
    lines.append("")
    recommendations: list[str] = []
    if bias is not None and bias.position_bias_rate is not None and bias.position_bias_rate > 0.15:
        recommendations.append(
            f"- Position bias rate is {bias.position_bias_rate:.2f} (> 0.15): keep double-order "
            "pairwise judging ON and treat flipped items as ties."
        )
    if bias is not None and bias.verbosity is not None and bias.verbosity.significant:
        recommendations.append(
            f"- Verbosity bias is significant (slope={bias.verbosity.slope:.4f}, "
            f"p={bias.verbosity.p_value:.3f}): length-balance comparison pairs or add an "
            "explicit conciseness criterion."
        )
    if bias is not None and bias.self_preference is not None and bias.self_preference.rate > 0:
        recommendations.append(
            f"- Self-preference detected on {len(bias.self_preference.flagged_item_ids)} item(s): "
            "never let a model family judge its own outputs; use a cross-family judge."
        )
    if layer is not None:
        calibrated = layer.calibrate_many(judge)
        after = agreement_summary(calibrated, human)
        if after["mae"] > 1.0:
            recommendations.append(
                f"- MAE after calibration is still {after['mae']:.2f} (> 1.0): collect more "
                "labels via the active labeling loop before trusting automated gates."
            )
        if after["spearman"] < 0.7:
            recommendations.append(
                f"- Rank agreement is weak (Spearman {after['spearman']:.2f} < 0.7): add rubric "
                "criteria with anchors and require evidence quotes."
            )
    if not recommendations:
        recommendations.append(
            "- No red flags: judge is rank-consistent, calibrated, and shows no dominant bias "
            "on the current label set."
        )
    lines.extend(recommendations)
    lines.append("")
    return "\n".join(lines)
