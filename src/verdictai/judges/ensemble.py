"""Weighted judge ensemble with disagreement flagging.

The ensemble is the first stage of the human-calibration loop: when judges
disagree beyond a threshold the item is flagged for human review instead of
trusting any single verdict. Flagged items are exactly the highest-value
targets for the active labeling loop in ``verdictai.calibration.active``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import EnsembleResult, JudgeItem, JudgeResult


@dataclass
class EnsembleMember:
    """One ensemble member: a judge object with ``async judge(item)``."""

    name: str
    judge: Any
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


class JudgeEnsemble:
    """Weighted mean over member judges + disagreement detection."""

    judge_type = "ensemble"

    def __init__(self, members: Sequence[EnsembleMember], disagreement_threshold: float = 1.5):
        if not members:
            raise ValueError("JudgeEnsemble needs at least one member")
        self.members = list(members)
        self.disagreement_threshold = disagreement_threshold

    async def judge(self, item: JudgeItem) -> EnsembleResult:
        results: list[JudgeResult] = await asyncio.gather(
            *(member.judge.judge(item) for member in self.members)
        )
        scores = {
            member.name: float(r.overall)
            for member, r in zip(self.members, results, strict=False)
        }
        total_weight = sum(m.weight for m in self.members)
        if total_weight <= 0:
            weighted = sum(scores.values()) / len(scores)
        else:
            weighted = sum(m.weight * scores[m.name] for m in self.members) / total_weight
        values = list(scores.values())
        disagreement = (max(values) - min(values)) if len(values) > 1 else 0.0
        flagged = disagreement > self.disagreement_threshold
        reason = (
            f"score spread {disagreement:.2f} exceeds threshold "
            f"{self.disagreement_threshold:.2f}"
            if flagged
            else ""
        )
        return EnsembleResult(
            item_id=item.item_id,
            overall=round(weighted, 4),
            per_judge=scores,
            disagreement=round(disagreement, 4),
            flagged=flagged,
            reason=reason,
            meta={
                "judge_types": {
                    m.name: getattr(m.judge, "judge_type", "custom") for m in self.members
                }
            },
        )
