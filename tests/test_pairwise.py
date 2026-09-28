"""Pairwise double-order de-biasing math with planted position bias."""

from __future__ import annotations

import pytest

from verdictai.judges.pairwise import PairwiseJudge, position_bias_rate
from verdictai.llm import EchoMockClient
from verdictai.types import PairwiseItem

A_TEXT = "The refund is processed within 5 business days once the item is received."
B_TEXT = "Refunds take about a week after we get your return back."


def _item():
    return PairwiseItem(item_id="p1", prompt="When do I get my refund?", response_a=A_TEXT,
                        response_b=B_TEXT)


class TestDoubleOrder:
    def test_consistent_winner_no_bias(self, pairwise_rig, run_async):
        # Rig always prefers canonical A regardless of presentation order.
        client = pairwise_rig(lambda first, second: "first" if first == A_TEXT else "second")
        result = run_async(PairwiseJudge(client).judge_pair(_item()))
        assert result.score_a == pytest.approx(1.0)
        assert result.winner == "a"
        assert result.position_bias is False
        assert [v.presentation_order for v in result.verdicts] == ["ab", "ba"]

    def test_planted_position_bias_detected_and_debiased(self, pairwise_rig, run_async):
        # Rig always picks whatever is presented FIRST: order ab -> A wins,
        # order ba -> B wins. The two orders disagree decisively.
        client = pairwise_rig(lambda first, second: "first")
        result = run_async(PairwiseJudge(client).judge_pair(_item()))
        assert result.position_bias is True
        assert result.score_a == pytest.approx(0.5)  # (1 + 0) / 2
        assert result.winner == "tie"
        assert result.meta["s_ab"] == 1.0 and result.meta["s_ba"] == 0.0

    def test_position_bias_rate_counts_flipped_items(self, pairwise_rig, run_async):
        judge = PairwiseJudge(pairwise_rig(lambda first, second: "first"))
        flipped = run_async(judge.judge_pair(_item()))
        stable = run_async(
            PairwiseJudge(
                pairwise_rig(lambda first, second: "first" if first == A_TEXT else "second")
            ).judge_pair(_item())
        )
        assert position_bias_rate([flipped, stable]) == pytest.approx(0.5)

    def test_tie_handling(self, pairwise_rig, run_async):
        client = pairwise_rig(lambda first, second: "tie")
        result = run_async(PairwiseJudge(client).judge_pair(_item()))
        assert result.score_a == pytest.approx(0.5)
        assert result.winner == "tie"
        assert result.position_bias is False

    def test_half_tie_is_not_flagged_as_bias(self, pairwise_rig, run_async):
        # One decisive win + one tie: de-biased 0.75, no strict contradiction.
        client = pairwise_rig(
            lambda first, second: "first" if first == A_TEXT else "tie"
        )
        result = run_async(PairwiseJudge(client).judge_pair(_item()))
        assert result.score_a == pytest.approx(0.75)
        assert result.position_bias is False
        assert result.winner == "a"

    def test_invalid_winner_raises(self, pairwise_rig, run_async):
        from verdictai.errors import JudgeParseError

        client = pairwise_rig(lambda first, second: "both")
        with pytest.raises(JudgeParseError):
            run_async(PairwiseJudge(client).judge_pair(_item()))


class TestEchoMockPairwise:
    def test_longer_response_wins_regardless_of_order(self, run_async):
        judge = PairwiseJudge(EchoMockClient())
        result = run_async(judge.judge_pair(_item()))
        # EchoMock picks the longer response deterministically -> A wins both
        # orders, so the mock itself cannot plant position bias.
        assert result.score_a == pytest.approx(1.0)
        assert result.position_bias is False

    def test_equal_lengths_tie(self, run_async):
        item = PairwiseItem(item_id="p2", prompt="q", response_a="same", response_b="same")
        result = run_async(PairwiseJudge(EchoMockClient()).judge_pair(item))
        assert result.winner == "tie"
