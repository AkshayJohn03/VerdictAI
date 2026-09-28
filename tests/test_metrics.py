"""Agreement metrics vs hand-worked examples."""

from __future__ import annotations

import pytest

from verdictai.calibration.metrics import (
    cohen_kappa,
    mean_absolute_error,
    quadratic_weighted_kappa,
    spearman,
    within_tolerance_accuracy,
)


class TestCohenKappa:
    def test_hand_worked_example(self):
        # judge=[0,0,1,1], human=[0,1,1,1]: po=0.75, pe=0.5 -> kappa=0.5
        assert cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)

    def test_perfect_agreement(self):
        assert cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)

    def test_inverse_assignment_is_negative(self):
        # po=0, pe=0.5 -> kappa = -1
        assert cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)

    def test_string_labels(self):
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([0, 1], [0, 1, 1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohen_kappa([], [])


class TestQuadraticWeightedKappa:
    def test_hand_worked_example(self):
        # QWK = 1 - 1.25/1.75 = 2/7 (see metrics docstring)
        y1 = [0, 1, 2, 2, 1, 0]
        y2 = [1, 1, 2, 0, 1, 0]
        assert quadratic_weighted_kappa(y1, y2) == pytest.approx(2.0 / 7.0)

    def test_perfect_agreement(self):
        assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

    def test_far_miss_penalized_more_than_near_miss(self):
        # One far miss (0 vs 2) should score worse than one near miss (0 vs 1).
        far = quadratic_weighted_kappa([0, 1, 1], [2, 1, 1])
        near = quadratic_weighted_kappa([0, 1, 1], [1, 1, 1])
        assert far < near < 1.0

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            quadratic_weighted_kappa(["a", "b"], ["a", "b"])


class TestSpearman:
    def test_with_ties_matches_midrank_pearson(self):
        # x=[1,2,2,3] midranks [1,2.5,2.5,4]; rho = 3/sqrt(10)
        assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(3.0 / 10**0.5)

    def test_no_ties_matches_textbook_formula(self):
        # d = [-1,1,-1,1], sum(d^2)=4, n=4 -> 1 - 24/60 = 0.6
        assert spearman([1, 2, 3, 4], [2, 1, 4, 3]) == pytest.approx(0.6)

    def test_perfect_monotonic(self):
        assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
        assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)

    def test_constant_input_returns_zero(self):
        assert spearman([2, 2, 2], [1, 2, 3]) == 0.0


class TestSimpleMetrics:
    def test_mae(self):
        assert mean_absolute_error([1, 2, 3], [2, 2, 2]) == pytest.approx(2.0 / 3.0)

    def test_within_one_tolerance_inclusive(self):
        # |1-1.4|=0.4 ok, |2-3|=1.0 ok (inclusive), |5-2|=3 no
        assert within_tolerance_accuracy([1, 2, 5], [1.4, 3, 2]) == pytest.approx(2 / 3)

    def test_within_tolerance_custom(self):
        assert within_tolerance_accuracy([1, 5], [1.2, 4.0], tolerance=0.5) == pytest.approx(0.5)
