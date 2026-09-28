"""CI regression gate.

``run_gate`` turns a drift comparison into a CI-friendly decision:

- exit code 1 when the candidate **regressed** beyond the threshold with at
  least the minimum effect size (the effect-size guard prevents noise alarms:
  a statistically-clean shift of 0.01 points on 5,000 items should not block
  a merge);
- a webhook-ready JSON alert payload;
- a human-readable markdown summary for the job log / PR comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..regression.drift import DriftDetector, DriftReport
from ..regression.runner import ScoreSnapshot


@dataclass
class GateResult:
    """Outcome of a regression gate check."""

    passed: bool
    exit_code: int
    verdict: str
    report_markdown: str
    alert: dict[str, Any] = field(default_factory=dict)


def run_gate(
    baseline: ScoreSnapshot,
    candidate: ScoreSnapshot,
    *,
    threshold: float = 0.05,
    min_effect: float = 0.1,
    detector: DriftDetector | None = None,
    context: dict[str, Any] | None = None,
) -> GateResult:
    """Compare snapshots and build the gate decision + artifacts."""
    detector = detector or DriftDetector()
    report = detector.check(baseline, candidate, threshold=threshold, min_effect=min_effect)
    passed = report.verdict != "regressed"
    exit_code = 0 if passed else 1

    alert: dict[str, Any] = {
        "alert_type": "verdictai.regression_gate",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": report.verdict,
        "passed": passed,
        "baseline_model_version": baseline.model_version,
        "candidate_model_version": candidate.model_version,
        "baseline_dataset_version": baseline.dataset_version,
        "candidate_dataset_version": candidate.dataset_version,
        "threshold": threshold,
        "min_effect": min_effect,
        "metrics": {
            "n_items": report.n_items,
            "mean_delta": report.mean_delta,
            "ci_low": report.ci_low,
            "ci_high": report.ci_high,
            "effect_size": report.effect_size,
            "wilcoxon_p": report.wilcoxon_p,
        },
        "failing_capabilities": report.failing_capabilities,
        "improving_capabilities": report.improving_capabilities,
        "context": context or {},
    }
    return GateResult(
        passed=passed,
        exit_code=exit_code,
        verdict=report.verdict,
        report_markdown=render_gate_markdown(report, baseline, candidate, threshold, min_effect),
        alert=alert,
    )


def render_gate_markdown(
    report: DriftReport,
    baseline: ScoreSnapshot,
    candidate: ScoreSnapshot,
    threshold: float,
    min_effect: float,
) -> str:
    header = "## VerdictAI regression gate: FAILED" if report.verdict == "regressed" else (
        "## VerdictAI regression gate: PASSED"
    )
    lines = [
        header,
        "",
        f"- Baseline: `{baseline.model_version}` (dataset {baseline.dataset_version})",
        f"- Candidate: `{candidate.model_version}` (dataset {candidate.dataset_version})",
        "",
        "```",
        report.summary(),
        "```",
        "",
        f"Gate rule: regression is claimed when the bootstrap CI lower bound of the "
        f"mean delta exceeds **{threshold}** AND |Cliff's delta| >= **{min_effect}**.",
        "",
    ]
    if report.slices:
        lines.append("| capability | n | mean delta | effect | verdict |")
        lines.append("|---|---|---|---|---|")
        for cap, s in report.slices.items():
            ci = "n/a"
            if s["ci_low"] is not None:
                ci = f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
            lines.append(
                f"| {cap} | {s['n']} | {s['mean_delta']:+.4f} | {s['effect_size']:+.2f} "
                f"| {s['verdict']} (CI {ci}) |"
            )
        lines.append("")
    return "\n".join(lines)
