"""Rubric judges: deterministic heuristic fallback + LLM rubric parsing."""

from __future__ import annotations

import json

import pytest

from verdictai.errors import JudgeParseError, LLMResponseError
from verdictai.judges.rubric import DEFAULT_RUBRIC, HeuristicJudge, RubricJudge
from verdictai.llm import EchoMockClient
from verdictai.types import Criterion, JudgeItem

GOOD_RESPONSE = (
    "The outage affected the eu-west region.\n"
    "- Root cause: a bad config rollout\n"
    "- Fix: rolled back the config\n"
    '- Action items: add a canary deploy step\n'
)
BAD_RESPONSE = "idk"

PROMPT = (
    "Summarize the incident report. Cover the outage, the affected region, and the fix. "
    "List action items as bullets."
)


def _item(response):
    return JudgeItem(item_id="r1", prompt=PROMPT, response=response)


class TestHeuristicJudge:
    def test_deterministic(self, run_async):
        first = run_async(HeuristicJudge().judge(_item(GOOD_RESPONSE)))
        second = run_async(HeuristicJudge().judge(_item(GOOD_RESPONSE)))
        assert first.overall == second.overall
        assert first.meta["heuristic"] is True

    def test_strong_beats_weak(self, run_async):
        strong = run_async(HeuristicJudge().judge(_item(GOOD_RESPONSE)))
        weak = run_async(HeuristicJudge().judge(_item(BAD_RESPONSE)))
        assert strong.overall > weak.overall
        assert weak.overall < 2.0

    def test_format_criterion_detects_bullets(self, run_async):
        result = run_async(HeuristicJudge().judge(_item(GOOD_RESPONSE)))
        format_score = next(c.score for c in result.criterion_scores if c.name == "format")
        assert format_score == 5.0

    def test_json_format_check(self, run_async):
        prompt = "Return strict JSON with key answer."
        ok = run_async(
            HeuristicJudge().judge(JudgeItem(item_id="j1", prompt=prompt, response='{"answer": 4}'))
        )
        bad = run_async(
            HeuristicJudge().judge(JudgeItem(item_id="j2", prompt=prompt, response="four"))
        )
        ok_format = next(c.score for c in ok.criterion_scores if c.name == "format")
        bad_format = next(c.score for c in bad.criterion_scores if c.name == "format")
        assert ok_format == 5.0
        assert bad_format == 1.0

    def test_empty_response_scores_zero(self, run_async):
        result = run_async(HeuristicJudge().judge(_item("")))
        assert result.overall == 0.0

    def test_scores_in_range(self, run_async):
        result = run_async(HeuristicJudge().judge(_item(GOOD_RESPONSE)))
        assert 0.0 <= result.overall <= 5.0
        for cs in result.criterion_scores:
            assert 0.0 <= cs.score <= 5.0


def _rubric_json(entries):
    return json.dumps({"scores": entries})


CRITERIA = [
    Criterion(name="accuracy", description="Is it correct?", weight=2.0),
    Criterion(name="tone", description="Is the tone right?", weight=1.0),
]


class TestRubricJudge:
    def test_weighted_overall_hand_computed(self, scripted_client, run_async):
        client = scripted_client(
            [
                _rubric_json(
                    [
                        {"name": "accuracy", "score": 4, "evidence": "q", "rationale": "r"},
                        {"name": "tone", "score": 2, "evidence": "q", "rationale": "r"},
                    ]
                )
            ]
        )
        result = run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x")))
        # (2*4 + 1*2) / 3 = 10/3 (overall stored rounded to 4 decimals)
        assert result.overall == pytest.approx(10.0 / 3.0, abs=1e-3)
        assert result.judge_type == "rubric_llm"
        assert result.meta["heuristic"] is False

    def test_scores_clamped_to_range(self, scripted_client, run_async):
        client = scripted_client(
            [
                _rubric_json(
                    [
                        {"name": "accuracy", "score": 9, "evidence": "", "rationale": ""},
                        {"name": "tone", "score": -1, "evidence": "", "rationale": ""},
                    ]
                )
            ]
        )
        result = run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x")))
        assert [c.score for c in result.criterion_scores] == [5.0, 0.0]

    def test_malformed_json_raises(self, scripted_client, run_async):
        client = scripted_client(["this is not json"])
        with pytest.raises(LLMResponseError):
            run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x")))

    def test_missing_criterion_raises(self, scripted_client, run_async):
        client = scripted_client(
            [_rubric_json([{"name": "accuracy", "score": 3, "evidence": "", "rationale": ""}])]
        )
        with pytest.raises(JudgeParseError):
            run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x")))

    def test_non_numeric_score_raises(self, scripted_client, run_async):
        client = scripted_client(
            [
                _rubric_json(
                    [
                        {"name": "accuracy", "score": "high", "evidence": "", "rationale": ""},
                        {"name": "tone", "score": 2, "evidence": "", "rationale": ""},
                    ]
                )
            ]
        )
        with pytest.raises(JudgeParseError):
            run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x")))

    def test_temperature_forwarded_to_client(self, scripted_client, run_async):
        client = scripted_client(
            [
                _rubric_json(
                    [
                        {"name": "accuracy", "score": 3, "evidence": "", "rationale": ""},
                        {"name": "tone", "score": 3, "evidence": "", "rationale": ""},
                    ]
                )
            ]
        )
        run_async(RubricJudge(client, criteria=CRITERIA).judge(_item("x"), temperature=0.4))
        assert client.calls[0]["temperature"] == 0.4

    def test_default_rubric_used_when_none_given(self, scripted_client, run_async):
        payload = _rubric_json(
            [{"name": c.name, "score": 2, "evidence": "", "rationale": ""} for c in DEFAULT_RUBRIC]
        )
        result = run_async(RubricJudge(scripted_client([payload])).judge(_item("x")))
        assert result.overall == pytest.approx(2.0)


class TestEchoMockRubric:
    def test_deterministic_baseline_scores(self, run_async):
        judge = RubricJudge(EchoMockClient(), criteria=CRITERIA)
        first = run_async(judge.judge(_item(GOOD_RESPONSE)))
        second = run_async(judge.judge(_item(GOOD_RESPONSE)))
        assert first.overall == second.overall == pytest.approx(3.0)
        for cs in first.criterion_scores:
            assert cs.score == 3.0
