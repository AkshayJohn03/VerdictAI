"""Judge ensemble (weighted + disagreement flagging) and self-consistency."""

from __future__ import annotations

import json

import pytest

from verdictai.judges.consistency import SelfConsistency
from verdictai.judges.ensemble import EnsembleMember, JudgeEnsemble
from verdictai.judges.rubric import RubricJudge
from verdictai.llm import EchoMockClient
from verdictai.types import JudgeItem, JudgeResult


class FixedJudge:
    """Test double returning a constant overall score."""

    judge_type = "fixed"

    def __init__(self, score):
        self.score = score

    async def judge(self, item, temperature=0.0):
        return JudgeResult(item_id=item.item_id, judge_type=self.judge_type, overall=self.score)


ITEM = JudgeItem(item_id="e1", prompt="p", response="r")


class TestEnsemble:
    def test_weighted_overall_and_disagreement(self, run_async):
        ensemble = JudgeEnsemble(
            [
                EnsembleMember(name="strong", judge=FixedJudge(4.0), weight=3.0),
                EnsembleMember(name="weak", judge=FixedJudge(2.0), weight=1.0),
            ]
        )
        result = run_async(ensemble.judge(ITEM))
        assert result.overall == pytest.approx(3.5)  # (3*4 + 1*2) / 4
        assert result.disagreement == pytest.approx(2.0)
        assert result.flagged is True
        assert "2.00" in result.reason

    def test_below_threshold_not_flagged(self, run_async):
        ensemble = JudgeEnsemble(
            [
                EnsembleMember(name="a", judge=FixedJudge(3.0)),
                EnsembleMember(name="b", judge=FixedJudge(3.5)),
            ],
            disagreement_threshold=1.5,
        )
        result = run_async(ensemble.judge(ITEM))
        assert result.flagged is False
        assert result.disagreement == pytest.approx(0.5)

    def test_single_member_zero_disagreement(self, run_async):
        ensemble = JudgeEnsemble([EnsembleMember(name="solo", judge=FixedJudge(4.0))])
        result = run_async(ensemble.judge(ITEM))
        assert result.disagreement == 0.0
        assert result.flagged is False

    def test_empty_ensemble_raises(self):
        with pytest.raises(ValueError):
            JudgeEnsemble([])

    def test_real_heuristic_members_smoke(self, run_async):
        from verdictai.judges.reference import ReferenceJudge
        from verdictai.judges.rubric import HeuristicJudge
        from verdictai.types import JudgeItem as JI

        item = JI(item_id="s1", prompt="explain json output", reference="json data",
                  response='{"data": 1}')
        ensemble = JudgeEnsemble(
            [
                EnsembleMember(name="rubric", judge=HeuristicJudge(), weight=2.0),
                EnsembleMember(name="reference", judge=ReferenceJudge()),
            ]
        )
        result = run_async(ensemble.judge(item))
        assert 0.0 <= result.overall <= 5.0
        assert set(result.per_judge) == {"rubric", "reference"}


class TestSelfConsistency:
    def _rubric_responses(self, scores):
        out = []
        for s in scores:
            out.append(
                json.dumps(
                    {
                        "scores": [
                            {"name": c.name, "score": s, "evidence": "", "rationale": ""}
                            for c in RubricJudge(EchoMockClient()).criteria
                        ]
                    }
                )
            )
        return out

    def test_majority_and_variance(self, scripted_client, run_async):
        judge = RubricJudge(scripted_client(self._rubric_responses([4, 4, 4, 2, 4])))
        result = run_async(SelfConsistency(k=5, temperature=0.7).run(judge, ITEM))
        assert result.majority_score == 4
        assert result.mean == pytest.approx(3.6)
        assert result.variance == pytest.approx(0.64)  # population variance
        assert result.confidence == pytest.approx(1.0 / 1.64, abs=1e-3)  # stored rounded
        assert result.samples == [4.0, 4.0, 4.0, 2.0, 4.0]

    def test_deterministic_judge_zero_variance(self, run_async):
        judge = RubricJudge(EchoMockClient())
        result = run_async(SelfConsistency(k=3).run(judge, ITEM))
        assert result.variance == 0.0
        assert result.confidence == 1.0
        assert result.majority_score == 3

    def test_temperature_forwarded(self, scripted_client, run_async):
        judge = RubricJudge(scripted_client(self._rubric_responses([3, 3])))
        run_async(SelfConsistency(k=2, temperature=0.9).run(judge, ITEM))
        assert all(call["temperature"] == 0.9 for call in judge.client.calls)

    def test_majority_tie_breaks_to_lower_score(self, scripted_client, run_async):
        judge = RubricJudge(scripted_client(self._rubric_responses([2, 3])))
        result = run_async(SelfConsistency(k=2).run(judge, ITEM))
        assert result.majority_score == 2

    def test_k_too_small_raises(self):
        with pytest.raises(ValueError):
            SelfConsistency(k=1)
