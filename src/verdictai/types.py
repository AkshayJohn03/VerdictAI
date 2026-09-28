"""Shared domain models for judging, calibration and regression tracking."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Winner = Literal["a", "b", "tie"]


class Criterion(BaseModel):
    """A single rubric criterion with weight and qualitative anchors."""

    name: str
    description: str
    weight: float = 1.0
    anchors: dict[int, str] = Field(default_factory=dict)


class JudgeItem(BaseModel):
    """A single prompt/response pair to score pointwise."""

    item_id: str
    prompt: str
    response: str
    reference: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class PairwiseItem(BaseModel):
    """A single comparison between two candidate responses."""

    item_id: str
    prompt: str
    response_a: str
    response_b: str
    meta: dict[str, Any] = Field(default_factory=dict)


class CriterionScore(BaseModel):
    """Score of one rubric criterion plus supporting evidence."""

    name: str
    score: float
    evidence: str = ""
    rationale: str = ""


class JudgeResult(BaseModel):
    """Output of a pointwise judge."""

    item_id: str
    judge_type: str
    overall: float
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    raw: Any = None
    meta: dict[str, Any] = Field(default_factory=dict)


class PairwiseVerdict(BaseModel):
    """One directional verdict from a pairwise comparison (one order)."""

    item_id: str
    presentation_order: Literal["ab", "ba"]
    presentation_winner: Literal["first", "second", "tie"]
    canonical_winner: Winner
    rationale: str = ""
    raw: Any = None


class PairwiseResult(BaseModel):
    """De-biased aggregate of both presentation orders for one item.

    ``score_a`` is the de-biased win rate of response ``a`` in ``[0, 1]``:
    the mean of the two binary (0/1, tie=0.5) verdicts taken over both
    presentation orders. ``position_bias`` marks items whose winner flips
    with presentation order.
    """

    item_id: str
    score_a: float
    winner: Winner
    position_bias: bool
    verdicts: list[PairwiseVerdict] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class EnsembleResult(BaseModel):
    """Output of a weighted judge ensemble."""

    item_id: str
    overall: float
    per_judge: dict[str, float] = Field(default_factory=dict)
    disagreement: float = 0.0
    flagged: bool = False
    reason: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class ConsistencyResult(BaseModel):
    """Output of self-consistency sampling over one item."""

    item_id: str
    majority_score: int
    mean: float
    variance: float
    confidence: float
    samples: list[float] = Field(default_factory=list)
