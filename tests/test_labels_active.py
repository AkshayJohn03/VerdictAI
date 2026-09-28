"""Human label loading and the active labeling loop."""

from __future__ import annotations

import csv
import io

import pytest

from verdictai.calibration.active import ActiveLabelLoop
from verdictai.calibration.labels import HumanLabelSet

L1 = '{"item_id": "l1", "prompt": "p1", "response_a": "a1", "judge_score": 3.0, "human_score": 3.0}'
L2 = '{"item_id": "l2", "prompt": "p2", "response_a": "a2", "judge_score": 4.0, "human_score": 0.0}'
L3 = '{"item_id": "l3", "prompt": "p3", "response_a": "a3", "judge_score": 2.5}'
L4 = '{"item_id": "l4", "prompt": "p4", "response_a": "a4", "judge_score": 4.0, "human_score": 3.0}'
L5 = (
    '{"item_id": "l5", "prompt": "p5", "response_a": "a5", "response_b": "b5", '
    '"judge_score": 3.0, "human_score": 1.0, "preferred": "a", '
    '"meta": {"judge_family": "gpt-4o", "response_a_family": "gpt-4o"}}'
)
LABELS_JSONL = "\n".join([L1, L2, L3, L4, L5])


class TestHumanLabelSet:
    def test_load_roundtrip(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(LABELS_JSONL, encoding="utf-8")
        label_set = HumanLabelSet.load(path)
        assert len(label_set) == 5
        assert len(label_set.paired()) == 4
        assert len(label_set.pairwise()) == 1
        assert label_set.labels[4].meta["judge_family"] == "gpt-4o"
        reloaded = HumanLabelSet.from_jsonl(label_set.to_jsonl())
        assert [lab.item_id for lab in reloaded.labels] == [lab.item_id for lab in label_set.labels]

    def test_invalid_line_reports_line_number(self):
        bad = (
            '{"item_id": "ok", "judge_score": 1.0}\n'
            '{"no_item_id": true}\n'
            'not json at all\n'
        )
        with pytest.raises(ValueError) as excinfo:
            HumanLabelSet.from_jsonl(bad)
        message = str(excinfo.value)
        assert "line 2" in message and "line 3" in message

    def test_blank_lines_skipped(self):
        label_set = HumanLabelSet.from_jsonl("\n\n" + LABELS_JSONL + "\n\n")
        assert len(label_set) == 5


class TestActiveLabelLoop:
    def _label_set(self):
        return HumanLabelSet.from_jsonl(LABELS_JSONL)

    def test_priority_ordering_hand_computed(self):
        # Priorities (w_gap=0.7, w_dis=0.3, disagreement={l1: 4.5}):
        #   l1: 0.7*|3-3|/5 + 0.3*0.9 = 0.27
        #   l2: 0.7*|4-0|/5 + 0.3*0.0 = 0.56
        #   l3: unlabeled -> gap 1.0   = 0.70
        #   l4: 0.7*|4-3|/5            = 0.14
        #   l5: 0.7*|3-1|/5            = 0.28
        loop = ActiveLabelLoop(self._label_set(), disagreement={"l1": 4.5})
        ranked = loop.next_round(k=5)
        assert [label.item_id for label, _ in ranked] == ["l3", "l2", "l5", "l1", "l4"]
        assert ranked[0][1] == pytest.approx(0.70)

    def test_unlabeled_boosted_to_front(self):
        loop = ActiveLabelLoop(self._label_set())
        top = loop.next_round(k=1)
        assert top[0][0].item_id == "l3"

    def test_k_limits_output(self):
        loop = ActiveLabelLoop(self._label_set())
        assert len(loop.next_round(k=2)) == 2

    def test_export_csv_queue(self, tmp_path):
        path = tmp_path / "round1.csv"
        loop = ActiveLabelLoop(self._label_set(), disagreement={"l1": 4.5})
        loop.export_csv(path)
        text = path.read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 5
        assert rows[0]["item_id"] == "l3"
        assert rows[0]["human_score"] == ""  # empty column for the human
        assert rows[0]["has_human_label"] == "False"
        assert float(rows[1]["priority"]) == pytest.approx(0.56, abs=1e-3)

    def test_weights_validation(self):
        with pytest.raises(ValueError):
            ActiveLabelLoop(self._label_set(), weight_gap=0.9, weight_disagreement=0.9)
