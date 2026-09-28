"""Pairwise judging with explicit position-bias control.

Known failure mode of LLM judges: verdicts flip with presentation order
(position bias; e.g. Zheng et al. 2023 "Judging LLM-as-a-Judge", Wang et al.
2023 "Large Language Models are not Fair Evaluators"). Mitigation used here:

1. Every comparison is run TWICE — once as (A, B) and once as (B, A).
2. The de-biased score of ``a`` is the mean of the two binary verdicts
   (win=1, loss=0, tie=0.5), so a position-flipped item scores 0.5 instead
   of 0 or 1.
3. Items whose two orders disagree on a *decisive* winner are flagged; the
   fraction of such items is the ``position_bias_rate`` of a run.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import JudgeParseError
from ..llm import LLMClient, complete_json
from ..prompts import build_pairwise_prompt
from ..types import PairwiseItem, PairwiseResult, PairwiseVerdict, Winner

_PRESENTATION_ALIASES = {
    "first": "first",
    "a": "first",
    "response_first": "first",
    "second": "second",
    "b": "second",
    "response_second": "second",
    "tie": "tie",
}


def _score_for_a(canonical_winner: Winner) -> float:
    """Binary score for response ``a``: win=1.0, loss=0.0, tie=0.5."""
    if canonical_winner == "a":
        return 1.0
    if canonical_winner == "b":
        return 0.0
    return 0.5


class PairwiseJudge:
    """Runs both presentation orders and aggregates de-biased verdicts."""

    judge_type = "pairwise_llm"

    def __init__(self, client: LLMClient, max_tokens: int | None = None):
        if client is None:
            raise ValueError("PairwiseJudge requires an LLMClient (use the echo mock offline).")
        self.client = client
        self.max_tokens = max_tokens

    async def judge_pair(self, item: PairwiseItem) -> PairwiseResult:
        verdict_ab = await self._verdict(item, "ab")
        verdict_ba = await self._verdict(item, "ba")
        s_ab = _score_for_a(verdict_ab.canonical_winner)
        s_ba = _score_for_a(verdict_ba.canonical_winner)
        debiased = (s_ab + s_ba) / 2.0
        position_bias = {s_ab, s_ba} == {0.0, 1.0}
        if debiased > 0.5:
            winner: Winner = "a"
        elif debiased < 0.5:
            winner = "b"
        else:
            winner = "tie"
        return PairwiseResult(
            item_id=item.item_id,
            score_a=round(debiased, 4),
            winner=winner,
            position_bias=position_bias,
            verdicts=[verdict_ab, verdict_ba],
            meta={"s_ab": s_ab, "s_ba": s_ba, "debiased": debiased},
        )

    async def _verdict(self, item: PairwiseItem, order: str) -> PairwiseVerdict:
        prompt = build_pairwise_prompt(item, order)
        data = await complete_json(self.client, prompt, max_tokens=self.max_tokens)
        raw = str(data.get("winner", "")).strip().lower()
        if raw not in _PRESENTATION_ALIASES:
            raise JudgeParseError(f"pairwise payload has invalid winner: {data!r}")
        presentation_winner = _PRESENTATION_ALIASES[raw]
        if order == "ab":
            canonical: Winner = {"first": "a", "second": "b", "tie": "tie"}[presentation_winner]
        else:
            canonical = {"first": "b", "second": "a", "tie": "tie"}[presentation_winner]
        return PairwiseVerdict(
            item_id=item.item_id,
            presentation_order=order,  # type: ignore[arg-type]
            presentation_winner=presentation_winner,  # type: ignore[arg-type]
            canonical_winner=canonical,
            rationale=str(data.get("rationale", "")),
            raw=data,
        )


def position_bias_rate(results: Sequence[PairwiseResult]) -> float:
    """Fraction of pairwise results whose winner flipped with order."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.position_bias) / len(results)
