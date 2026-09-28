"""Self-consistency: sample the judge k times and measure verdict stability.

For a judge whose verdicts scatter at fixed temperature, the majority verdict
is more reliable than any single sample and the variance of the samples is a
cheap confidence signal — low-variance items need no human review, high
variance items are prime candidates for the labeling loop.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..types import ConsistencyResult, JudgeItem, JudgeResult


class SelfConsistency:
    """Run ``judge`` k times at temperature and aggregate the samples."""

    def __init__(self, k: int = 5, temperature: float = 0.7):
        if k < 2:
            raise ValueError("SelfConsistency needs k >= 2 samples")
        self.k = k
        self.temperature = temperature

    async def run(
        self,
        judge: Callable[[JudgeItem, float], Any],
        item: JudgeItem,
    ) -> ConsistencyResult:
        """``judge`` must expose ``async judge(item, temperature) -> JudgeResult``."""
        samples: list[float] = []
        for _ in range(self.k):
            result: JudgeResult = await judge.judge(item, self.temperature)
            samples.append(float(result.overall))
        rounded = [int(math.floor(s + 0.5)) for s in samples]
        counts = Counter(rounded)
        best = max(counts.values())
        majority = min(score for score, n in counts.items() if n == best)
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)  # population variance
        confidence = 1.0 / (1.0 + variance)
        return ConsistencyResult(
            item_id=item.item_id,
            majority_score=majority,
            mean=mean,
            variance=variance,
            confidence=round(confidence, 4),
            samples=samples,
        )
