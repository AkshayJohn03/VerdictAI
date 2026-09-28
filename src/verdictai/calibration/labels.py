"""Human label storage: the ground-truth side of judge calibration.

JSONL schema (one object per line)::

    {
      "item_id": "ticket-42",
      "prompt": "...",
      "response_a": "...",            # pointwise: the judged response
      "response_b": "...",            # optional (pairwise labels)
      "judge_score": 3.5,             # 0-5 scale
      "human_score": 4.0,             # 0-5 scale (empty while unlabeled)
      "judge_criterion_scores": {"accuracy": 4, "tone": 3},
      "human_criterion_scores": {"accuracy": 5, "tone": 3},
      "preferred": "a",               # pairwise labels: "a" | "b" | "tie"
      "meta": {"judge_family": "gpt-4o", "response_a_family": "gpt-4o", ...}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class HumanLabel(BaseModel):
    """One human-labeled judging example."""

    item_id: str
    prompt: str = ""
    response_a: str = ""
    response_b: str = ""
    judge_score: float | None = None
    human_score: float | None = None
    judge_criterion_scores: dict[str, float] = Field(default_factory=dict)
    human_criterion_scores: dict[str, float] = Field(default_factory=dict)
    preferred: str | None = None  # "a" | "b" | "tie" for pairwise labels
    meta: dict[str, Any] = Field(default_factory=dict)


class HumanLabelSet(BaseModel):
    """A collection of human labels loaded from JSONL."""

    labels: list[HumanLabel] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_jsonl(cls, text: str) -> HumanLabelSet:
        """Parse JSONL text; raises ``ValueError`` listing bad line numbers."""
        labels: list[HumanLabel] = []
        errors: list[str] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                labels.append(HumanLabel.model_validate(record))
            except (json.JSONDecodeError, ValidationError) as exc:
                errors.append(f"line {line_no}: {exc}")
        if errors:
            raise ValueError("invalid label JSONL:\n" + "\n".join(errors))
        return cls(labels=labels)

    @classmethod
    def load(cls, path: str | Path) -> HumanLabelSet:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))

    def to_jsonl(self) -> str:
        return "\n".join(label.model_dump_json() for label in self.labels)

    def paired(self) -> list[HumanLabel]:
        """Labels with both a judge score and a human score."""
        return [
            label
            for label in self.labels
            if label.judge_score is not None and label.human_score is not None
        ]

    def pairwise(self) -> list[HumanLabel]:
        """Labels carrying a pairwise preference."""
        return [label for label in self.labels if label.preferred in ("a", "b", "tie")]
