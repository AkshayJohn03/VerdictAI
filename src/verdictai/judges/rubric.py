"""Rubric-based judging.

Two implementations share one output contract (:class:`~verdictai.types.
JudgeResult` with per-criterion scores, evidence and rationale):

- :class:`RubricJudge` — LLM-backed. The model scores each criterion 0-5 with
  an evidence quote and a one-line rationale; overall is the weighted mean.
  Rubric decomposition (many anchored criteria instead of one holistic score)
  reduces judge variance and makes verdicts auditable.

- :class:`HeuristicJudge` — a fully deterministic, clearly-labeled heuristic
  fallback (keyword coverage + length + format checks). It exists so the whole
  pipeline runs offline with zero cost; it is NOT a quality signal and its
  results carry ``meta["heuristic"] = True``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..errors import JudgeParseError
from ..llm import LLMClient, complete_json
from ..prompts import build_rubric_prompt
from ..text import content_tokens, keyword_coverage, tokenize
from ..types import Criterion, CriterionScore, JudgeItem, JudgeResult

DEFAULT_RUBRIC: list[Criterion] = [
    Criterion(
        name="relevance",
        weight=1.0,
        description="Response addresses the actual request in the prompt.",
        anchors={0: "Completely off-topic.", 5: "Directly and fully answers the ask."},
    ),
    Criterion(
        name="completeness",
        weight=1.0,
        description="Response covers the key points a strong answer would include.",
        anchors={0: "No substantive content.", 5: "All key points covered."},
    ),
    Criterion(
        name="clarity",
        weight=1.0,
        description="Response is well-structured and easy to follow.",
        anchors={0: "Rambling or unreadable.", 5: "Crisp, organized, easy to scan."},
    ),
    Criterion(
        name="format",
        weight=1.0,
        description="Response follows any output format the prompt requests.",
        anchors={0: "Ignores the requested format.", 5: "Matches the requested format exactly."},
    ),
]


def weighted_overall(criteria: list[Criterion], scores: dict[str, float]) -> float:
    """Weighted mean of criterion scores (plain mean when weights sum to 0)."""
    total_weight = sum(c.weight for c in criteria)
    if total_weight <= 0:
        return sum(scores.values()) / len(scores) if scores else 0.0
    return sum(c.weight * scores[c.name] for c in criteria) / total_weight


def _format_score(prompt: str, response: str) -> float:
    """Check whether explicitly requested output formats are honored."""
    lowered = prompt.lower()
    if "json" in lowered:
        stripped = response.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return 5.0
            except json.JSONDecodeError:
                return 3.0
        return 1.0
    if any(key in lowered for key in ("bullet", "list", "numbered", "steps")):
        if re.search(r"(?m)^\s*([-*•]|\d+[.)])\s", response):
            return 5.0
        return 1.0
    return 3.0


class HeuristicJudge:
    """Deterministic offline judge (keyword coverage + length + format checks).

    Signals per default criterion:

    - ``relevance``: fraction of prompt content-keywords appearing in response.
    - ``completeness``: fraction of reference content-keywords covered (or a
      response/prompt length ratio when no reference exists).
    - ``clarity``: sentence-length sanity + presence of structure markers.
    - ``format``: honors explicit JSON / bullet / list requests.

    Results are labeled ``meta["heuristic"] = True`` everywhere so they can
    never be mistaken for LLM verdicts.
    """

    judge_type = "heuristic"

    def __init__(self, criteria: list[Criterion] | None = None, name: str = "heuristic-rubric"):
        self.name = name
        self.criteria = list(criteria) if criteria is not None else list(DEFAULT_RUBRIC)

    async def judge(self, item: JudgeItem, temperature: float = 0.0) -> JudgeResult:
        scores = {c.name: self._score_criterion(c, item) for c in self.criteria}
        criterion_scores = [
            CriterionScore(
                name=c.name,
                score=scores[c.name],
                evidence=self._evidence(c, item),
                rationale="Deterministic heuristic signal (not an LLM verdict).",
            )
            for c in self.criteria
        ]
        overall = weighted_overall(self.criteria, scores)
        return JudgeResult(
            item_id=item.item_id,
            judge_type=self.judge_type,
            overall=round(overall, 4),
            criterion_scores=criterion_scores,
            raw={"scores": scores},
            meta={"heuristic": True, "judge": self.name, "signals": scores},
        )

    def _score_criterion(self, criterion: Criterion, item: JudgeItem) -> float:
        name = criterion.name.lower()
        if not item.response.strip():
            return 0.0
        response_tokens = tokenize(item.response)
        if "format" in name:
            # Format checks parse the response directly and must not be
            # short-circuited by the token-count guard (JSON can be tiny).
            return _format_score(item.prompt, item.response)
        if "clarit" in name:
            sentence_ends = sum(item.response.count(ch) for ch in (".", "!", "?", "\n"))
            sentences = max(1, sentence_ends)
            avg_len = len(response_tokens) / sentences
            score = 3.0
            if any(
                marker in item.response for marker in ("\n-", "\n*", "\n1.", "\n#")
            ) or "\n\n" in item.response:
                score += 1.0
            if avg_len > 40 or avg_len < 3:
                score -= 1.0
            return round(min(5.0, max(0.0, score)), 2)
        if len(response_tokens) < 3:
            # Too little content to judge relevance/completeness meaningfully.
            return 1.0
        if "relevan" in name:
            coverage = keyword_coverage(content_tokens(item.prompt), item.response)
            score = 5.0 * coverage
        elif "complete" in name:
            if item.reference:
                coverage = keyword_coverage(content_tokens(item.reference), item.response)
                score = 5.0 * coverage
            else:
                ratio = len(response_tokens) / max(1, len(tokenize(item.prompt)))
                score = 5.0 * min(1.0, ratio)
        else:
            coverage = keyword_coverage(content_tokens(criterion.description), item.response)
            score = 5.0 * coverage
        return round(min(5.0, max(0.0, score)), 2)

    @staticmethod
    def _evidence(criterion: Criterion, item: JudgeItem) -> str:
        name = criterion.name.lower()
        if "relevan" in name:
            coverage = keyword_coverage(content_tokens(item.prompt), item.response)
            return f"prompt-keyword coverage {coverage:.2f}"
        if "complete" in name and item.reference:
            coverage = keyword_coverage(content_tokens(item.reference), item.response)
            return f"reference-keyword coverage {coverage:.2f}"
        return f"response length {len(tokenize(item.response))} tokens"


class RubricJudge:
    """LLM-backed rubric judge (per-criterion score + evidence + rationale)."""

    judge_type = "rubric_llm"

    def __init__(
        self,
        client: LLMClient,
        criteria: list[Criterion] | None = None,
        name: str = "rubric-llm",
        max_tokens: int | None = None,
    ):
        if client is None:
            raise ValueError("RubricJudge requires an LLMClient; use HeuristicJudge offline.")
        self.client = client
        self.name = name
        self.max_tokens = max_tokens
        self.criteria = list(criteria) if criteria is not None else list(DEFAULT_RUBRIC)

    async def judge(self, item: JudgeItem, temperature: float = 0.0) -> JudgeResult:
        prompt = build_rubric_prompt(item, self.criteria)
        data = await complete_json(
            self.client, prompt, temperature=temperature, max_tokens=self.max_tokens
        )
        criterion_scores = self._parse_scores(data)
        overall = weighted_overall(
            self.criteria, {cs.name: cs.score for cs in criterion_scores}
        )
        return JudgeResult(
            item_id=item.item_id,
            judge_type=self.judge_type,
            overall=round(overall, 4),
            criterion_scores=criterion_scores,
            raw=data,
            meta={"judge": self.name, "temperature": temperature, "heuristic": False},
        )

    def _parse_scores(self, data: Any) -> list[CriterionScore]:
        entries = data.get("scores") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise JudgeParseError("rubric payload must contain a 'scores' list")
        by_name: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or "name" not in entry or "score" not in entry:
                raise JudgeParseError(f"malformed score entry: {entry!r}")
            by_name[str(entry["name"])] = entry
        missing = [c.name for c in self.criteria if c.name not in by_name]
        if missing:
            raise JudgeParseError(f"rubric payload missing criteria: {missing}")
        out: list[CriterionScore] = []
        for criterion in self.criteria:
            entry = by_name[criterion.name]
            try:
                score = float(entry["score"])
            except (TypeError, ValueError) as exc:
                raise JudgeParseError(f"non-numeric score for {criterion.name}: {entry!r}") from exc
            out.append(
                CriterionScore(
                    name=criterion.name,
                    score=round(min(5.0, max(0.0, score)), 4),
                    evidence=str(entry.get("evidence", "")),
                    rationale=str(entry.get("rationale", "")),
                )
            )
        return out
