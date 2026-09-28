"""Active labeling loop: pick the next items for human review.

Priority per item combines two signals:

1. **Judge-human gap** — ``|judge_score - human_score|`` (normalized to [0, 1]
   by dividing by 5). Unlabeled items (no human score yet) get the maximum
   gap of 1.0, so fresh judge output is reviewed before re-checking known
   calibration failures.
2. **Ensemble disagreement** — the max score spread from ``JudgeEnsemble``,
   passed in as ``{item_id: disagreement}``.

This is uncertainty sampling: every labeling hour buys the most calibration
information per label.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from pathlib import Path

from .labels import HumanLabel, HumanLabelSet

_CSV_FIELDS = [
    "item_id",
    "priority",
    "judge_score",
    "disagreement",
    "has_human_label",
    "human_score",
    "prompt",
    "response_a",
    "response_b",
]


class ActiveLabelLoop:
    """Ranks labels for the next human review round and exports CSV queues."""

    def __init__(
        self,
        label_set: HumanLabelSet,
        disagreement: Mapping[str, float] | None = None,
        *,
        weight_gap: float = 0.7,
        weight_disagreement: float = 0.3,
    ):
        if abs(weight_gap + weight_disagreement - 1.0) > 1e-9:
            raise ValueError("weight_gap and weight_disagreement must sum to 1.0")
        self.label_set = label_set
        self.disagreement = dict(disagreement or {})
        self.weight_gap = weight_gap
        self.weight_disagreement = weight_disagreement

    def priority(self, label: HumanLabel) -> float:
        """Higher priority = review sooner, in [0, 1]."""
        if label.judge_score is None:
            gap = 1.0
        elif label.human_score is None:
            gap = 1.0
        else:
            gap = min(1.0, abs(label.judge_score - label.human_score) / 5.0)
        dis = min(1.0, self.disagreement.get(label.item_id, 0.0) / 5.0)
        return self.weight_gap * gap + self.weight_disagreement * dis

    def next_round(self, k: int = 10) -> list[tuple[HumanLabel, float]]:
        """The ``k`` highest-priority labels, descending, deterministic ties."""
        scored = [(label, self.priority(label)) for label in self.label_set.labels]
        scored.sort(key=lambda pair: (-pair[1], pair[0].item_id))
        return scored[:k]

    def export_csv(self, path: str | Path | None = None) -> str:
        """Write (or return) a labeling queue CSV with an empty human column."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for label, priority in self.next_round(k=len(self.label_set.labels)):
            writer.writerow(
                {
                    "item_id": label.item_id,
                    "priority": f"{priority:.4f}",
                    "judge_score": label.judge_score,
                    "disagreement": self.disagreement.get(label.item_id, ""),
                    "has_human_label": label.human_score is not None,
                    "human_score": label.human_score if label.human_score is not None else "",
                    "prompt": label.prompt,
                    "response_a": label.response_a,
                    "response_b": label.response_b,
                }
            )
        content = buffer.getvalue()
        if path is not None:
            Path(path).write_text(content, encoding="utf-8")
        return content
