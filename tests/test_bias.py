"""Bias diagnostics: position bias rate, verbosity regression + permutation
test, self-preference detection."""

from __future__ import annotations

import numpy as np
import pytest

from verdictai.calibration.bias import (
    SelfPreferenceReport,
    self_preference,
    summarize_bias,
    verbosity_bias,
)
from verdictai.calibration.labels import HumanLabel, HumanLabelSet
from verdictai.judges.pairwise import PairwiseResult, position_bias_rate


def _pw(item_id, score_a, bias):
    return PairwiseResult(
        item_id=item_id, score_a=score_a, winner="tie", position_bias=bias
    )


class TestPositionBiasRate:
    def test_rate(self):
        results = [
            _pw("a", 1.0, False),
            _pw("b", 0.5, True),
            _pw("c", 0.0, False),
            _pw("d", 0.5, True),
        ]
        assert position_bias_rate(results) == pytest.approx(0.5)

    def test_empty(self):
        assert position_bias_rate([]) == 0.0


class TestVerbosityBias:
    def test_planted_length_bias_is_detected(self):
        rng = np.random.default_rng(0)
        lengths = rng.integers(40, 1200, size=60).astype(float)
        scores = 1.0 + 0.004 * lengths + rng.normal(0, 0.3, size=60)
        report = verbosity_bias(scores.tolist(), lengths.tolist(), seed=1)
        assert report.slope == pytest.approx(0.004, abs=0.001)
        assert report.significant is True
        assert report.p_value <= 0.01  # 999 permutations, strong signal

    def test_no_bias_in_null_data(self):
        rng = np.random.default_rng(13)
        lengths = rng.integers(40, 1200, size=60).astype(float)
        scores = rng.normal(3.0, 0.4, size=60)
        report = verbosity_bias(scores.tolist(), lengths.tolist(), seed=13)
        assert report.significant is False
        assert report.p_value > 0.5

    def test_deterministic_for_fixed_seed(self):
        rng = np.random.default_rng(3)
        lengths = rng.integers(50, 900, size=30).astype(float)
        scores = (0.5 + 0.002 * lengths + rng.normal(0, 0.5, size=30)).tolist()
        a = verbosity_bias(scores, lengths.tolist(), seed=11)
        b = verbosity_bias(scores, lengths.tolist(), seed=11)
        assert a.p_value == b.p_value and a.slope == b.slope

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            verbosity_bias([1.0, 2.0], [10, 20])


class TestSelfPreference:
    def test_flags_own_family_preference(self):
        labels = [
            HumanLabel(
                item_id="i1", preferred="a",
                meta={"judge_family": "gpt-4o", "response_a_family": "gpt-4o",
                      "response_b_family": "claude"},
            ),
            HumanLabel(
                item_id="i2", preferred="b",
                meta={"judge_family": "gpt-4o", "response_a_family": "gpt-4o",
                      "response_b_family": "claude"},
            ),
            HumanLabel(
                item_id="i3", preferred="a",
                meta={"judge_family": "gpt-4o", "response_a_family": "claude",
                      "response_b_family": "mistral"},
            ),
            HumanLabel(
                item_id="i4", preferred="tie",
                meta={"judge_family": "gpt-4o", "response_a_family": "gpt-4o",
                      "response_b_family": "claude"},
            ),
            HumanLabel(item_id="i5", preferred="a", meta={}),
        ]
        report = self_preference(labels)
        # comparable: i1, i2 (judge family on a side + decisive pick);
        # i3's own family is not present, i4 is a tie, i5 has no metadata.
        assert report.n_comparable == 2
        assert report.flagged_item_ids == ["i1"]
        assert report.rate == pytest.approx(0.5)

    def test_no_metadata_no_flags(self):
        report = SelfPreferenceReport(rate=0.0, flagged_item_ids=[], n_comparable=0)
        assert report.rate == 0.0


class TestSummarizeBias:
    def test_assembles_all_available_signals(self):
        labels = HumanLabelSet(
            labels=[
                HumanLabel(item_id=f"i{i}", judge_score=s, response_a="word " * length)
                for i, (s, length) in enumerate(
                    [(1.0, 100), (2.0, 300), (4.0, 700), (5.0, 900)]
                )
            ]
        )
        report = summarize_bias(labels, pairwise_results=[_pw("x", 0.5, True)])
        assert report.position_bias_rate == pytest.approx(1.0)
        assert report.verbosity is not None
        assert report.verbosity.slope > 0
        assert report.self_preference is not None
        assert report.self_preference.n_comparable == 0
