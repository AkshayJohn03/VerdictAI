"""CI regression gate: exit codes, alert payload, markdown, CLI wiring."""

from __future__ import annotations

import json

import pytest

from verdictai.cli import main
from verdictai.regression.gate import run_gate
from verdictai.regression.runner import SnapshotStore


@pytest.fixture
def regressed_pair(snapshot_factory):
    baseline = snapshot_factory("v1", [4.0] * 20)
    candidate = snapshot_factory("v2", [3.7] * 20)
    return baseline, candidate


@pytest.fixture
def noise_pair(snapshot_factory):
    import numpy as np

    rng = np.random.default_rng(17)
    return (
        snapshot_factory("v1", rng.normal(3.5, 0.25, 40).tolist()),
        snapshot_factory("v2", rng.normal(3.5, 0.25, 40).tolist()),
    )


class TestRunGate:
    def test_regression_blocks(self, regressed_pair):
        baseline, candidate = regressed_pair
        gate = run_gate(baseline, candidate, threshold=0.05, min_effect=0.1)
        assert gate.passed is False
        assert gate.exit_code == 1
        assert gate.verdict == "regressed"
        assert "FAILED" in gate.report_markdown
        assert gate.alert["alert_type"] == "verdictai.regression_gate"
        assert gate.alert["verdict"] == "regressed"
        assert gate.alert["candidate_model_version"] == "v2"
        assert gate.alert["metrics"]["ci_low"] == pytest.approx(0.3, abs=0.01)

    def test_noise_passes(self, noise_pair):
        baseline, candidate = noise_pair
        gate = run_gate(baseline, candidate, threshold=0.05, min_effect=0.1)
        assert gate.passed is True
        assert gate.exit_code == 0
        assert gate.alert["verdict"] == "no_change"
        assert "PASSED" in gate.report_markdown

    def test_effect_guard_prevents_noise_alarm(self, snapshot_factory):
        baseline = snapshot_factory("v1", [4.15] * 30 + [3.90] * 10)
        candidate = snapshot_factory("v2", [4.0] * 40)
        gate = run_gate(baseline, candidate, threshold=0.01, min_effect=0.8)
        assert gate.passed is True  # CI excludes 0 but effect is only 0.5
        assert gate.exit_code == 0

    def test_alert_payload_is_webhook_ready_json(self, regressed_pair):
        baseline, candidate = regressed_pair
        gate = run_gate(baseline, candidate)
        payload = json.dumps(gate.alert)  # must be JSON-serializable
        parsed = json.loads(payload)
        assert parsed["baseline_model_version"] == "v1"
        assert "failing_capabilities" in parsed


class TestGateCLI:
    def _write_snapshots(self, tmp_path, snapshot_factory, scores_v2):
        store = SnapshotStore(tmp_path / "runs.jsonl")
        store.append(snapshot_factory("v1", [4.0] * 20))
        store.append(snapshot_factory("v2", scores_v2))
        return tmp_path / "runs.jsonl"

    def test_cli_gate_exit_1_on_regression(self, tmp_path, snapshot_factory, capsys):
        path = self._write_snapshots(tmp_path, snapshot_factory, [3.5] * 20)
        code = main(
            ["regress", "--snapshots", str(path), "--baseline", "v1", "--candidate", "v2",
             "--gate", "--threshold", "0.05"]
        )
        assert code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out

    def test_cli_gate_exit_0_on_noise(self, tmp_path, snapshot_factory):
        import numpy as np

        rng = np.random.default_rng(17)
        path = self._write_snapshots(tmp_path, snapshot_factory, rng.normal(4.0, 0.2, 20).tolist())
        code = main(
            ["regress", "--snapshots", str(path), "--baseline", "v1", "--candidate", "v2",
             "--gate"]
        )
        assert code == 0

    def test_cli_report_mode_always_exits_zero(self, tmp_path, snapshot_factory, capsys):
        path = self._write_snapshots(tmp_path, snapshot_factory, [3.5] * 20)
        code = main(
            ["regress", "--snapshots", str(path), "--baseline", "v1", "--candidate", "v2"]
        )
        assert code == 0
        assert "REGRESSION" in capsys.readouterr().out

    def test_cli_missing_snapshot_errors(self, tmp_path, capsys):
        code = main(
            ["regress", "--snapshots", str(tmp_path / "none.jsonl"),
             "--baseline", "v1", "--candidate", "v2"]
        )
        assert code == 2
