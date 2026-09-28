"""PAV isotonic regression + calibration layer vs hand-worked cases."""

from __future__ import annotations

import pytest

from verdictai.calibration.bias import BiasReport, VerbosityBiasReport
from verdictai.calibration.isotonic import (
    CalibrationLayer,
    IsotonicFit,
    render_calibration_report,
)
from verdictai.calibration.labels import HumanLabel, HumanLabelSet


class TestPAV:
    def test_non_monotonic_sequence_pooled(self):
        fit = IsotonicFit.fit([1, 2, 3, 4], [1, 3, 2, 5])
        assert fit.predict_many([1, 2, 3, 4]) == pytest.approx([1, 2.5, 2.5, 5])

    def test_already_monotonic_unchanged(self):
        fit = IsotonicFit.fit([1, 2, 3], [1, 2, 3])
        assert fit.predict_many([1, 2, 3]) == pytest.approx([1, 2, 3])
        assert fit.n_blocks == 3

    def test_all_decreasing_collapses_to_mean(self):
        fit = IsotonicFit.fit([1, 2, 3], [3, 2, 1])
        assert fit.predict_many([1, 2, 3]) == pytest.approx([2, 2, 2])
        assert fit.n_blocks == 1

    def test_weighted_pooling(self):
        # y=[3,1] w=[3,1] -> pooled (9+1)/4 = 2.5; w=[1,3] -> (3+3)/4 = 1.5
        assert IsotonicFit.fit([1, 2], [3, 1], weights=[3, 1]).predict_many([1, 2]) == (
            pytest.approx([2.5, 2.5])
        )
        assert IsotonicFit.fit([1, 2], [3, 1], weights=[1, 3]).predict_many([1, 2]) == (
            pytest.approx([1.5, 1.5])
        )

    def test_unsorted_input_is_sorted_internally(self):
        # Same fit as IsotonicFit.fit([1,2,3],[1,2,3]) once sorted by x.
        fit = IsotonicFit.fit([2, 1, 3], [2, 1, 3])
        assert fit.predict_many([1, 2, 3]) == pytest.approx([1, 2, 3])
        assert fit.n_blocks == 3

    def test_out_of_range_clamps(self):
        fit = IsotonicFit.fit([1, 2, 3, 4], [1, 3, 2, 5])
        assert fit.predict(0.5) == pytest.approx(1)  # below range -> first block
        assert fit.predict(9.0) == pytest.approx(5)  # above range -> last block

    def test_gap_interpolation(self):
        # Blocks: [1@x=1], [2.5@x=2..3], [5@x=4]; x=3.5 is a gap -> 2.5 + 0.5*2.5
        fit = IsotonicFit.fit([1, 2, 3, 4], [1, 3, 2, 5])
        assert fit.predict(2.5) == pytest.approx(2.5)  # inside block 2
        assert fit.predict(3.5) == pytest.approx(3.75)  # halfway 2.5 -> 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            IsotonicFit.fit([], [])


class TestCalibrationLayer:
    def test_planted_miscalibration_is_fixed(self):
        # Judge over-scores: human = 0.6 * judge (monotone). Raw MAE is 40% of
        # the mean score; isotonic should recover the mapping near-exactly.
        judge = [1.0, 2.0, 3.0, 4.0, 5.0, 3.5, 2.5, 4.5, 1.5, 5.0]
        human = [0.6 * j for j in judge]
        layer = CalibrationLayer.fit(judge, human)
        calibrated = layer.calibrate_many(judge)
        raw_mae = sum(abs(j - h) for j, h in zip(judge, human, strict=False)) / len(judge)
        cal_mae = sum(abs(c - h) for c, h in zip(calibrated, human, strict=False)) / len(human)
        assert cal_mae < raw_mae
        assert cal_mae == pytest.approx(0.0, abs=0.05)

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError):
            CalibrationLayer.fit([1.0], [2.0])

    def test_monotonicity_preserved(self):
        layer = CalibrationLayer.fit([1, 2, 3, 4, 5], [1, 1, 2, 2, 5])
        scores = [0.0, 1.3, 2.2, 3.1, 4.4, 6.0]
        calibrated = layer.calibrate_many(scores)
        assert calibrated == sorted(calibrated)


class TestReport:
    def _label_set(self):
        labels = [
            HumanLabel(item_id=f"i{i}", judge_score=j, human_score=h)
            for i, (j, h) in enumerate(
                [(1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (4, 4), (2, 2), (3, 3)]
            )
        ]
        return HumanLabelSet(labels=labels)

    def test_report_contains_sections(self):
        labels = self._label_set()
        layer = CalibrationLayer.fit(
            [lab.judge_score for lab in labels.paired()],
            [lab.human_score for lab in labels.paired()],
        )
        bias = BiasReport(
            position_bias_rate=0.25,
            verbosity=VerbosityBiasReport(
                slope=0.002, intercept=2.0, r=0.5, p_value=0.01, n=8, significant=True,
                n_permutations=99,
            ),
        )
        text = render_calibration_report(labels, layer, bias)
        assert "VerdictAI judge calibration report" in text
        assert "before calibration" in text
        assert "after isotonic calibration" in text
        assert "Bias diagnostics" in text
        assert "0.250" in text  # position bias rate
        assert "double-order" in text  # recommendation triggered
        assert "length-balance" in text  # verbosity recommendation triggered

    def test_report_without_layer_or_bias(self):
        labels = self._label_set()
        text = render_calibration_report(labels, None, None)
        assert "(no bias diagnostics supplied)" in text
        assert "n/a" in text

    def test_empty_labels_message(self):
        text = render_calibration_report(HumanLabelSet(labels=[]), None, None)
        assert "No paired" in text
