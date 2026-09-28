"""Reference-based judging (answer vs golden answer).

Default mode is a deterministic token-F1 heuristic (fully offline). With an
``LLMClient`` the judge instead asks the model to score the response against
the golden answer, which captures paraphrase overlap the heuristic misses.
"""

from __future__ import annotations

from ..errors import JudgeParseError
from ..llm import LLMClient, complete_json
from ..prompts import build_reference_prompt
from ..text import f1_overlap
from ..types import CriterionScore, JudgeItem, JudgeResult


class ReferenceJudge:
    """Scores a response against a golden answer (heuristic F1 or LLM)."""

    def __init__(self, client: LLMClient | None = None, name: str = "reference"):
        self.client = client
        self.name = name
        self.judge_type = "reference_llm" if client is not None else "reference_heuristic"

    async def judge(self, item: JudgeItem, temperature: float = 0.0) -> JudgeResult:
        if self.client is None:
            return self._heuristic(item)
        return await self._llm(item, temperature)

    def _heuristic(self, item: JudgeItem) -> JudgeResult:
        f1 = f1_overlap(item.response, item.reference)
        score = round(5.0 * f1, 4)
        return JudgeResult(
            item_id=item.item_id,
            judge_type=self.judge_type,
            overall=score,
            criterion_scores=[
                CriterionScore(
                    name="reference_similarity",
                    score=score,
                    evidence=f"token-F1={f1:.3f}",
                    rationale="Deterministic bag-of-words F1 vs golden answer.",
                )
            ],
            raw={"f1": f1},
            meta={"heuristic": True, "judge": self.name},
        )

    async def _llm(self, item: JudgeItem, temperature: float) -> JudgeResult:
        data = await complete_json(
            self.client, build_reference_prompt(item), temperature=temperature
        )
        try:
            score = float(data["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeParseError(f"reference payload missing numeric score: {data!r}") from exc
        score = round(min(5.0, max(0.0, score)), 4)
        return JudgeResult(
            item_id=item.item_id,
            judge_type=self.judge_type,
            overall=score,
            criterion_scores=[
                CriterionScore(
                    name="reference_match",
                    score=score,
                    rationale=str(data.get("rationale", "")),
                )
            ],
            raw=data,
            meta={"heuristic": False, "judge": self.name, "temperature": temperature},
        )
