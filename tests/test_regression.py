"""Golden runner, snapshot store, drift statistics vs hand-worked cases."""

from __future__ import annotations

import numpy as np
import pytest

from verdictai.datasets.schema import DatasetItem
from verdictai.judges.reference import ReferenceJudge
from verdictai.regression.drift import (
    DriftDetector,
    _wilcoxon_normal_variance,
    cliffs_delta,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from verdictai.regression.runner import (
    GoldenRunner,
    OfflineReplayAdapter,
    SnapshotStore,
    degrade,
)


class ScriptedAdapter:
    """ModelAdapter double with programmed answers."""

    def __init__(self, answers, model_version="scripted-v1"):
        self.answers = answers
        self.model_version = model_version

    async def generate(self, prompt, *, item_id=""):
        return self.answers[item_id]


def _tiny_dataset(version="v1"):
    from verdictai.datasets.schema import EvalDataset

    items = [
        DatasetItem(
            item_id="d1",
            prompt="What is the refund window?",
            reference="Refunds are accepted within 30 days of purchase.",
            capability="policy",
            topic="policy",
            persona="customer",
            difficulty="easy",
            edge_case="ambiguity",
        ),
        DatasetItem(
            item_id="d2",
            prompt="Summarize the warranty terms.",
            reference="The warranty covers manufacturing defects for two years.",
            capability="policy",
            topic="policy",
            persona="customer",
            difficulty="easy",
            edge_case="ambiguity",
        ),
    ]
    return EvalDataset(
        version=version,
        generator_version="test",
        created_at="2026-01-01T00:00:00+00:00",
        config_digest="test",
        items=items,
    )


class TestDegrade:
    def test_full_quality_identity(self):
        assert degrade("one two three four", 1.0) == "one two three four"

    def test_lower_quality_drops_words(self):
        text = "one two three four five six seven eight nine ten"
        assert len(degrade(text, 0.3).split()) < len(text.split())

    def test_deterministic(self):
        text = "alpha beta gamma delta epsilon"
        assert degrade(text, 0.5) == degrade(text, 0.5)


class TestGoldenRunner:
    def test_perfect_replay_scores_five(self, run_async):
        dataset = _tiny_dataset()
        runner = GoldenRunner(OfflineReplayAdapter(dataset, quality=1.0), ReferenceJudge())
        snapshot = run_async(runner.run(dataset, persist=False))
        assert snapshot.mean() == pytest.approx(5.0)
        assert snapshot.model_version == "replay-v1"

    def test_degraded_replay_scores_lower(self, run_async):
        dataset = _tiny_dataset()
        good = run_async(
            GoldenRunner(OfflineReplayAdapter(dataset, quality=1.0), ReferenceJudge()).run(
                dataset, persist=False
            )
        )
        bad = run_async(
            GoldenRunner(OfflineReplayAdapter(dataset, quality=0.3), ReferenceJudge()).run(
                dataset, persist=False
            )
        )
        assert bad.mean() < good.mean()

    def test_snapshot_persisted_append_only(self, tmp_path, run_async):
        store = SnapshotStore(tmp_path / "runs.jsonl")
        dataset = _tiny_dataset()
        adapter = ScriptedAdapter({"d1": "Refunds within 30 days.", "d2": "Two years."},
                                  model_version="vA")
        run_async(GoldenRunner(adapter, ReferenceJudge(), store).run(dataset))
        adapter2 = ScriptedAdapter({"d1": "No idea.", "d2": "Nothing."}, model_version="vB")
        run_async(GoldenRunner(adapter2, ReferenceJudge(), store).run(dataset))
        lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # append-only: nothing rewritten
        snapshots = store.load_all()
        assert [s.model_version for s in snapshots] == ["vA", "vB"]
        scores = store.latest("vA").scores_by_item()
        assert 0.0 < scores["d1"] <= 5.0
        assert scores["d1"] > scores["d2"]  # closer to its reference

    def test_scripted_answers_scored_by_reference_overlap(self, run_async):
        dataset = _tiny_dataset()
        adapter = ScriptedAdapter(
            {"d1": "Refunds are accepted within 30 days of purchase.",
             "d2": "completely unrelated text"},
        )
        snapshot = run_async(
            GoldenRunner(adapter, ReferenceJudge(), None).run(dataset, persist=False)
        )
        scores = snapshot.scores_by_item()
        assert scores["d1"] == pytest.approx(5.0)
        assert scores["d2"] < 1.0


class TestBootstrap:
    def test_constant_delta_degenerate_ci(self):
        lo, hi, mean = paired_bootstrap_ci([0.5] * 20, seed=0)
        assert lo == hi == mean == pytest.approx(0.5)

    def test_seeded_deterministic(self):
        deltas = [0.1, -0.2, 0.3, 0.0, 0.25]
        assert paired_bootstrap_ci(deltas, seed=5) == paired_bootstrap_ci(deltas, seed=5)


class TestCliffsDelta:
    def test_hand_worked_examples(self):
        assert cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
        assert cliffs_delta([1, 2, 3], [2, 3, 4]) == pytest.approx(-5.0 / 9.0)
        assert cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)

    def test_paired_effect_counts_signs(self):
        from verdictai.regression.drift import _paired_effect

        # pos=2, neg=1, zero=1 -> (2 - 1) / 4
        assert _paired_effect([0.3, 0.3, -0.1, 0.0]) == pytest.approx(0.25)


class TestWilcoxon:
    def test_exact_small_n_all_positive(self):
        # n=4, all positive: P(W+ >= 10) = 1/16 -> two-sided p = 0.125
        result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0])
        assert result["method"] == "exact"
        assert result["statistic"] == pytest.approx(10.0)
        assert result["p_value"] == pytest.approx(0.125)

    def test_hand_worked_mixed_signs_exact(self):
        # Ranks of |d|: [4, 2, 5, 3, 1]; W+ = 3+4+5 = 12; P(W>=12) = 5/32
        result = wilcoxon_signed_rank([0.5, -0.2, 0.8, 0.3, -0.1])
        assert result["method"] == "exact"
        assert result["statistic"] == pytest.approx(12.0)
        assert result["p_value"] == pytest.approx(10.0 / 32.0)

    def test_normal_approximation_no_ties(self):
        result = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0], force_normal=True)
        # mean = 5, var = 7.5, z = 1.8257 -> p ~ 0.068
        assert result["method"] == "normal"
        assert result["p_value"] == pytest.approx(0.068, abs=2e-3)

    def test_tie_correction_variance(self):
        # n=5, |d| tie groups t=2 and t=3:
        # var = 5*6*11/24 - ((8-2)+(27-3))/48 = 13.75 - 0.625 = 13.125
        assert _wilcoxon_normal_variance(5, [0.5, 0.5, 0.3, 0.3, 0.3]) == pytest.approx(13.125)

    def test_all_zeros(self):
        result = wilcoxon_signed_rank([0.0, 0.0])
        assert result["p_value"] == 1.0 and result["n"] == 0


class TestDriftDetector:
    def test_planted_regression_detected(self, snapshot_factory):
        baseline = snapshot_factory(
            "v1", [4.0] * 10 + [3.0] * 10,
            capabilities=["reasoning"] * 10 + ["safety"] * 10,
        )
        candidate = snapshot_factory(
            "v2", [3.7] * 10 + [2.7] * 10,
            capabilities=["reasoning"] * 10 + ["safety"] * 10,
        )
        report = DriftDetector().check(baseline, candidate, threshold=0.05, min_effect=0.1)
        assert report.verdict == "regressed"
        assert report.ci_low > 0.05
        assert report.mean_delta == pytest.approx(0.3)
        assert report.effect_size == pytest.approx(1.0)
        assert report.wilcoxon_p < 0.05
        assert report.failing_capabilities == ["reasoning", "safety"]
        assert report.slices["safety"]["mean_delta"] == pytest.approx(0.3)

    def test_same_distribution_noise_not_flagged(self, snapshot_factory):
        rng = np.random.default_rng(17)
        baseline = snapshot_factory("v1", rng.normal(3.5, 0.25, 40).tolist())
        candidate = snapshot_factory("v2", rng.normal(3.5, 0.25, 40).tolist())
        report = DriftDetector().check(baseline, candidate, threshold=0.05, min_effect=0.1)
        assert report.verdict == "no_change"
        assert report.ci_low <= 0.0 <= report.ci_high

    def test_effect_size_guard_blocks_noise_alarm(self, snapshot_factory):
        # 30 items slightly worse, 10 items better: CI excludes 0 at threshold
        # 0.01 but the paired effect size (0.5) is below the 0.8 guard, so no
        # regression may be claimed.
        baseline_scores = [4.15] * 30 + [3.90] * 10
        candidate_scores = [4.0] * 40
        baseline = snapshot_factory("v1", baseline_scores)
        candidate = snapshot_factory("v2", candidate_scores)
        strict = DriftDetector().check(baseline, candidate, threshold=0.01, min_effect=0.1)
        guarded = DriftDetector().check(baseline, candidate, threshold=0.01, min_effect=0.8)
        assert strict.verdict == "regressed"
        assert guarded.verdict == "no_change"

    def test_improvement_detected(self, snapshot_factory):
        baseline = snapshot_factory("v1", [3.0] * 20)
        candidate = snapshot_factory("v2", [3.4] * 20)
        report = DriftDetector().check(baseline, candidate, threshold=0.05, min_effect=0.1)
        assert report.verdict == "improved"

    def test_missing_items_reported(self, snapshot_factory):
        baseline = snapshot_factory("v1", [4.0, 4.0, 4.0])
        candidate = snapshot_factory("v2", [3.9, 3.9])
        # snapshot_factory keys by item index; baseline has item-002 extra.
        report = DriftDetector().check(baseline, candidate)
        assert report.missing_item_ids == ["item-002"]

    def test_disjoint_snapshots_raise(self, snapshot_factory):
        baseline = snapshot_factory("v1", [4.0])
        candidate = snapshot_factory("v2", [])
        with pytest.raises(ValueError):
            DriftDetector().check(baseline, candidate)
