"""CLI surface: judge / pair / calibrate / gen-dataset, all offline."""

from __future__ import annotations

import json

from verdictai.cli import main

LABELS = "\n".join(
    [
        json.dumps(
            {
                "item_id": f"c{i}",
                "prompt": f"question {i}",
                "response_a": "answer text " * (10 + 5 * i),
                "judge_score": judge,
                "human_score": human,
            }
        )
        for i, (judge, human) in enumerate(
            [(1.0, 1.0), (2.0, 1.5), (3.0, 2.0), (4.0, 2.5), (5.0, 3.0),
             (2.5, 2.0), (3.5, 3.0), (4.5, 4.0)]
        )
    ]
)


class TestJudgeCommand:
    def test_heuristic_judge_offline(self, capsys):
        code = main(
            ["judge", "--prompt", "Answer in JSON: what is 2+2?",
             "--response", '{"answer": "4"}']
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["judge_type"] == "heuristic"
        assert 0.0 <= payload["overall"] <= 5.0
        assert payload["meta"]["heuristic"] is True

    def test_rubric_requires_api_key(self, capsys):
        code = main(
            ["judge", "--prompt", "p", "--response", "r", "--judge", "rubric"]
        )
        assert code == 2


class TestPairCommand:
    def test_pair_offline_uses_echo_mock(self, capsys):
        code = main(
            [
                "pair",
                "--prompt", "When do I get my refund?",
                "--response-a", "A thorough answer explaining the refund policy in detail.",
                "--response-b", "short",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["winner"] == "a"
        assert payload["client"] == "EchoMockClient"
        assert payload["position_bias"] is False


class TestCalibrateCommand:
    def test_calibrate_writes_report(self, tmp_path, capsys):
        labels = tmp_path / "labels.jsonl"
        labels.write_text(LABELS, encoding="utf-8")
        report = tmp_path / "report.md"
        code = main(["calibrate", "--labels", str(labels), "--out", str(report)])
        assert code == 0
        text = report.read_text(encoding="utf-8")
        assert "spearman" in text and "after isotonic calibration" in text
        assert "Next labeling round" in capsys.readouterr().out

    def test_calibrate_needs_two_paired_labels(self, tmp_path):
        labels = tmp_path / "thin.jsonl"
        labels.write_text(
            json.dumps({"item_id": "one", "judge_score": 3.0, "human_score": 3.0}),
            encoding="utf-8",
        )
        assert main(["calibrate", "--labels", str(labels)]) == 2


class TestGenDatasetCommand:
    def test_gen_dataset_creates_artifacts(self, tmp_path):
        out = tmp_path / "data"
        code = main(
            [
                "gen-dataset",
                "--out", str(out),
                "--version", "v1",
                "--seed", "5",
                "--topics", "math-word-problems:reasoning",
                "--personas", "a busy executive",
                "--difficulties", "easy,hard",
            ]
        )
        assert code == 0
        assert (out / "dataset-v1.jsonl").exists()
        assert (out / "manifest-v1.json").exists()
        assert (out / "CHANGELOG.md").exists()
        manifest = json.loads((out / "manifest-v1.json").read_text(encoding="utf-8"))
        assert manifest["n_items"] == 2

    def test_version_flag_short_circuit(self, capsys):
        assert main(["--version"]) == 0
        assert "verdict" in capsys.readouterr().out

    def test_help_exits_zero(self, capsys):
        assert main([]) == 0
        assert "judge" in capsys.readouterr().out
