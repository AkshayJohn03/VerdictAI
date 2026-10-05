"""The Seam Auditor and the deterministic-first suite.

Everything here is offline and deterministic: planted chains with known
fidelity numbers, hand-computed geometric means and halflife positions, and
blame attribution on chains whose worst node is planted on purpose.
"""

from __future__ import annotations

import json
import math

import pytest

from verdictai.deterministic import (
    DeterministicSuite,
    HashEmbeddingClient,
    TimedOutput,
    conformity_errors,
    validate_json_schema,
)
from verdictai.report import HtmlDiffReport
from verdictai.seams import (
    AgentStep,
    ChainAudit,
    HandoffAuditor,
    audit_json,
    flatten_fields,
)

# ---------------------------------------------------------------------------
# Chain builders


def passthrough(name: str, payload: dict) -> AgentStep:
    """A step whose output equals its input (pure field carrier)."""
    return AgentStep(name=name, input=dict(payload), output=dict(payload))


def degrading_chain() -> list[AgentStep]:
    """5 fields -> 4 -> 2 -> 1 -> 1 -> 0 (the canonical halflife fixture)."""
    return [
        AgentStep(
            name="extractor",
            output={"f1": 1, "f2": 2, "f3": 3, "f4": 4, "f5": 5},
        ),
        passthrough("triage", {"f1": 1, "f2": 2, "f3": 3, "f4": 4}),
        passthrough("resolver", {"f1": 1, "f2": 2}),
        passthrough("refunds", {"f1": 1}),
        passthrough("notifier", {"f1": 1}),
        passthrough("crm", {}),
    ]


# ---------------------------------------------------------------------------
# Single-seam diffs


class TestHandoffDiff:
    def test_dropped_field_detected(self):
        steps = [
            AgentStep(name="a", output={"x": 1, "y": 2, "z": 3}),
            passthrough("b", {"x": 1, "y": 2}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.from_step == "a" and diff.to_step == "b"
        assert diff.dropped_fields == ["z"]
        assert diff.mutated == []
        assert diff.fidelity == pytest.approx(2 / 3)

    def test_mutated_field_detected(self):
        steps = [
            AgentStep(name="a", output={"status": "high"}),
            passthrough("b", {"status": "urgent"}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert len(diff.mutated) == 1
        assert diff.mutated[0].field == "status"
        assert diff.mutated[0].before == "high"
        assert diff.mutated[0].after == "urgent"
        assert diff.dropped_fields == []
        assert diff.fidelity == 0.0  # value not preserved

    def test_type_change_detected(self):
        steps = [
            AgentStep(name="a", output={"count": "5"}),
            passthrough("b", {"count": 5}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.mutated == []
        assert len(diff.type_changed) == 1
        assert diff.type_changed[0].field == "count"
        assert diff.type_changed[0].before_type == "str"
        assert diff.type_changed[0].after_type == "number"
        assert diff.fidelity == 0.0

    def test_int_and_float_are_one_number_type(self):
        steps = [
            AgentStep(name="a", output={"price": 10}),
            passthrough("b", {"price": 10.5}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.type_changed == []
        assert len(diff.mutated) == 1  # value mutation, not a type change

    def test_added_fields_do_not_reduce_fidelity(self):
        steps = [
            AgentStep(name="a", output={"x": 1, "y": 2}),
            passthrough("b", {"x": 1, "y": 2, "extra": "new", "more": True}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.fidelity == 1.0
        assert sorted(diff.added_fields) == ["extra", "more"]
        assert diff.dropped_fields == []

    def test_nested_fields_use_dot_notation(self):
        steps = [
            AgentStep(
                name="a",
                output={"user": {"name": "jo", "address": {"city": "rome", "zip": "001"}}},
            ),
            passthrough("b", {"user": {"name": "jo", "address": {"city": "rome"}}}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.dropped_fields == ["user.address.zip"]
        assert "user.address.city" not in diff.dropped_fields
        assert diff.fidelity == pytest.approx(2 / 3)

    def test_list_shrink_is_a_length_mutation(self):
        steps = [
            AgentStep(name="a", output={"items": [{"sku": "x"}, {"sku": "y"}, {"sku": "z"}]}),
            passthrough("b", {"items": [{"sku": "x"}]}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert ("items#len", 3, 1) in [(m.field, m.before, m.after) for m in diff.mutated]
        assert "items[1].sku" in diff.dropped_fields
        assert diff.fidelity < 1.0

    def test_list_growth_is_not_degradation(self):
        steps = [
            AgentStep(name="a", output={"items": [{"sku": "x"}]}),
            passthrough("b", {"items": [{"sku": "x"}, {"sku": "y"}]}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.mutated == []
        assert diff.dropped_fields == []
        assert diff.fidelity == 1.0

    def test_lists_compared_by_sampled_elements(self):
        # Elements beyond the 3-element sample are only covered by length.
        assert flatten_fields({"items": [1, 2, 3, 4, 5]})[0] == {
            "items[0]": 1,
            "items[1]": 2,
            "items[2]": 3,
        }
        assert flatten_fields({"items": [1, 2, 3, 4, 5]})[1] == {"items": 5}

    def test_renamed_field_with_recoverable_value_is_not_a_drop(self):
        steps = [
            AgentStep(name="a", output={"billing_id": "NYC-1"}),
            passthrough("b", {"city": "NYC-1"}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.dropped_fields == []
        assert diff.fidelity == 1.0
        assert diff.added_fields == ["city"]

    def test_dropped_value_nowhere_downstream_is_a_real_drop(self):
        steps = [
            AgentStep(name="a", output={"ticket_id": "T-42"}),
            passthrough("b", {"note": "nothing carried"}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.dropped_fields == ["ticket_id"]
        assert diff.fidelity == 0.0

    def test_empty_output_has_full_fidelity(self):
        steps = [
            AgentStep(name="a", output={}),
            passthrough("b", {"x": 1}),
        ]
        diff = HandoffAuditor().audit(steps).handoffs[0]
        assert diff.fidelity == 1.0
        assert diff.dropped_fields == []


# ---------------------------------------------------------------------------
# Chain-level audit


class TestChainAudit:
    def test_chain_fidelity_is_geometric_mean(self):
        steps = [
            AgentStep(name="s0", output={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}),
            passthrough("s1", {"a": 1, "b": 2, "c": 3, "d": 4}),
            passthrough("s2", {"a": 1, "b": 2}),
        ]
        audit = HandoffAuditor().audit(steps)
        assert [h.fidelity for h in audit.handoffs] == [0.8, 0.5]
        assert audit.chain_fidelity == pytest.approx(math.sqrt(0.4), abs=1e-6)

    def test_single_step_chain_is_vacuously_perfect(self):
        audit = HandoffAuditor().audit([AgentStep(name="only", output={"a": 1})])
        assert audit.handoffs == []
        assert audit.chain_fidelity == 1.0
        assert audit.fidelity_halflife is None
        assert audit.blame() == []

    def test_halflife_counts_handoffs_until_below_half(self):
        # fidelities: 0.8, 0.5, 0.5, 1.0, 0.0 -> first strictly-below-0.5 is #5
        audit = HandoffAuditor().audit(degrading_chain())
        assert [h.fidelity for h in audit.handoffs] == [0.8, 0.5, 0.5, 1.0, 0.0]
        assert audit.fidelity_halflife == 5
        assert audit.chain_fidelity == 0.0  # one seam at 0.0 collapses the mean

    def test_halflife_flags_early_collapse(self):
        steps = [
            AgentStep(name="s0", output={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}),
            passthrough("s1", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}),
            passthrough("s2", {"a": 1, "b": 2}),  # fidelity 0.4 at handoff 2
        ]
        audit = HandoffAuditor().audit(steps)
        assert audit.fidelity_halflife == 2

    def test_halflife_none_when_preservation_stays_high(self):
        steps = [
            AgentStep(name="s0", output={"a": 1, "b": 2}),
            passthrough("s1", {"a": 1, "b": 2}),
            passthrough("s2", {"a": 1, "b": 2}),
        ]
        audit = HandoffAuditor().audit(steps)
        assert audit.chain_fidelity == 1.0
        assert audit.fidelity_halflife is None

    def test_step_fidelity_aligns_with_steps(self):
        audit = HandoffAuditor().audit(degrading_chain())
        assert isinstance(audit, ChainAudit)
        assert audit.step_names == ["extractor", "triage", "resolver", "refunds",
                                    "notifier", "crm"]
        assert audit.step_fidelity == [None, 0.8, 0.5, 0.5, 1.0, 0.0]

    def test_blame_names_weighted_worst_node(self):
        # 0.5 x 4 downstream (triage->resolver) must outweigh the final 0.0
        # seam with only 1 downstream step.
        audit = HandoffAuditor().audit(degrading_chain())
        top = audit.blame()[0]
        assert top.from_step == "triage" and top.to_step == "resolver"
        assert top.severity == pytest.approx(2.0)
        assert top.downstream_steps == 4

    def test_blame_weights_by_downstream_impact(self):
        # Same two bad fidelities, different positions: the ranking flips.
        early = [
            AgentStep(name="s0", output={"a": 1, "b": 2, "c": 3, "d": 4}),
            passthrough("s1", {"a": 1, "b": 2, "c": 3}),  # 0.75, 3 downstream
            passthrough("s2", {"a": 1, "b": 2, "c": 3}),
            passthrough("s3", {}),  # 0.0, 1 downstream
        ]
        audit = HandoffAuditor().audit(early)
        assert audit.blame()[0].to_step == "s3"  # 1.0 severity beats 0.75

        late = [
            AgentStep(name="s0", output={"a": 1, "b": 2, "c": 3, "d": 4}),
            passthrough("s1", {"a": 1}),  # 0.25, 3 downstream -> 0.75
            passthrough("s2", {"a": 1}),
            passthrough("s3", {"a": 1}),
        ]
        audit = HandoffAuditor().audit(late)
        assert audit.blame()[0].to_step == "s1"  # only lossy seam, most downstream

    def test_blame_is_sorted_worst_first(self):
        entries = HandoffAuditor().audit(degrading_chain()).blame()
        severities = [e.severity for e in entries]
        assert severities == sorted(severities, reverse=True)
        assert len(entries) == 5


# ---------------------------------------------------------------------------
# Trace loading


class TestAuditJson:
    def test_bare_json_list(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                [
                    {"name": "a", "output": {"x": 1, "y": 2}},
                    {"name": "b", "input": {"x": 1}, "output": {"x": 1}},
                ]
            ),
            encoding="utf-8",
        )
        audit = audit_json(trace)
        assert audit.step_names == ["a", "b"]
        assert audit.handoffs[0].dropped_fields == ["y"]

    def test_steps_key_and_duration_passthrough(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                {
                    "steps": [
                        {"name": "a", "output": {"x": 1}, "duration_ms": 12.5},
                        {"name": "b", "input": {"x": 1}, "output": {"x": 1}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        audit = audit_json(trace)
        assert audit.steps[0].duration_ms == 12.5
        assert audit.chain_fidelity == 1.0

    def test_jsonl_trace(self, tmp_path):
        trace = tmp_path / "trace.jsonl"
        rows = [
            {"name": "a", "output": {"x": 1, "y": 2}},
            {"name": "b", "input": {"x": 1, "y": 2}, "output": {"x": 1, "y": 2}},
        ]
        trace.write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        audit = audit_json(trace)
        assert audit.chain_fidelity == 1.0

    def test_trace_key_alias(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                {
                    "trace": [
                        {"name": "a", "output": {"x": 1}},
                        {"name": "b", "input": {"x": 1}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert audit_json(trace).chain_fidelity == 1.0

    def test_rows_without_name_get_positional_names(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(json.dumps([{"output": {"x": 1}}, {"input": {"x": 1}}]),
                         encoding="utf-8")
        audit = audit_json(trace)
        assert audit.step_names == ["step-0", "step-1"]

    def test_rejects_non_object_rows_and_empty_files(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"name": "a", "output": {"x": 1}}, "junk"]),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="row 1"):
            audit_json(bad)
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            audit_json(empty)


# ---------------------------------------------------------------------------
# Deterministic-first suite


class TestDeterministicSuite:
    def test_json_schema_violation_caught(self):
        report = DeterministicSuite().run_all(
            outputs={"answer": {"ticket_id": "T-1"}},
            schema={"answer": {"type": "object", "required": ["ticket_id", "priority"]}},
        )
        assert not report.passed
        failed = {r.name for r in report.failures}
        assert "schema_conformity:answer" in failed
        assert "required_fields:answer" in failed
        assert "missing required fields: priority" in report.summary()

    def test_pydantic_model_schema_inline(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            ticket_id: str
            priority: str

        report = DeterministicSuite().run_all(
            outputs={"good": {"ticket_id": "T-1", "priority": "high"},
                     "bad": {"ticket_id": 7}},
            schema=Answer,
        )
        names = {r.name: r.passed for r in report.results}
        assert names["schema_conformity:good"] is True
        assert names["schema_conformity:bad"] is False
        assert names["required_fields:bad"] is False  # 'priority' missing

    def test_clean_output_passes_everything(self):
        report = DeterministicSuite().run_all(
            outputs={
                "answer": TimedOutput(
                    value={"ticket_id": "T-1", "priority": "high"}, latency_ms=120.0
                )
            },
            schema={"answer": {"type": "object", "required": ["ticket_id", "priority"]}},
            refs={"answer": "ticket T-1 priority high"},
            budgets={"answer": 2000.0},
        )
        assert report.passed, report.summary()
        assert report.escalations == []

    def test_content_bounds(self):
        report = DeterministicSuite().run_all(
            outputs={
                "empty_str": "",
                "blank": "   ",
                "empty_dict": {},
                "none_out": None,
                "ok": "substantive answer",
                "too_long": "x" * 201,
            },
            schema={"too_long": {"type": "string", "maxLength": 200}},
        )
        failed = {r.name for r in report.failures}
        # declaring a string schema on "too_long" also trips json_valid +
        # schema_conformity (a str payload is not parseable as a JSON object)
        assert failed == {
            "content_bounds:empty_str",
            "content_bounds:blank",
            "content_bounds:empty_dict",
            "content_bounds:none_out",
            "content_bounds:too_long",
            "schema_conformity:too_long",
            "json_valid:too_long",
        }

    def test_json_valid_with_fenced_output(self):
        report = DeterministicSuite().run_all(
            outputs={
                "fenced": '```json\n{"a": 1}\n```',
                "garbage": "not json at all {",
            },
            schema={"fenced": {"type": "object"}, "garbage": {"type": "object"}},
        )
        by_name = {r.name: r for r in report.results}
        assert by_name["json_valid:fenced"].passed
        assert not by_name["json_valid:garbage"].passed
        assert not by_name["schema_conformity:garbage"].passed

    def test_similarity_threshold_vs_reference(self):
        suite = DeterministicSuite()
        report = suite.run_all(
            outputs={
                "on_topic": "refund processed within 5 business days of return",
                "off_topic": "today the weather in paris is sunny and warm",
            },
            refs={"on_topic": "refund processed within 5 business days of return",
                  "off_topic": "refund processed within 5 business days of return"},
        )
        by_name = {r.name: r for r in report.results}
        assert by_name["similarity:on_topic"].passed
        assert not by_name["similarity:off_topic"].passed
        assert by_name["similarity:off_topic"].escalate

    def test_latency_budget(self):
        report = DeterministicSuite().run_all(
            outputs={
                "fast": TimedOutput(value={"a": 1}, latency_ms=100.0),
                "slow": TimedOutput(value={"a": 1}, latency_ms=9000.0),
                "unwrapped": {"a": 1},
            },
            schema={"fast": {"type": "object"}, "slow": {"type": "object"},
                    "unwrapped": {"type": "object"}},
            budgets={"fast": 500.0, "slow": 500.0, "unwrapped": 500.0},
        )
        by_name = {r.name: r for r in report.results}
        assert by_name["latency_budget:fast"].passed
        assert not by_name["latency_budget:slow"].passed
        assert by_name["latency_budget:slow"].severity == "warning"
        assert not by_name["latency_budget:unwrapped"].passed  # no latency reported
        assert not by_name["latency_budget:unwrapped"].escalate

    def test_escalation_list_semantics(self):
        report = DeterministicSuite().run_all(
            outputs={
                "x": "",
            },
            schema={"x": {"type": "object", "required": ["a"]}},
        )
        # content_bounds (objective) fails; schema/JSON failures escalate —
        # a judge could re-extract structured output from a mangled string
        by_name = {r.name: r for r in report.results}
        assert by_name["content_bounds:x"].escalate is False
        assert by_name["schema_conformity:x"].escalate is True
        assert {r.name for r in report.escalations} == {
            "schema_conformity:x",
            "json_valid:x",
        }

    def test_results_in_deterministic_order(self):
        report = DeterministicSuite().run_all(
            outputs={"b": {"a": 1}, "a": {"a": 1}},
            schema={"a": {"type": "object"}, "b": {"type": "object"}},
        )
        names = [r.name for r in report.results]
        assert names == [
            "schema_conformity:a",
            "content_bounds:a",
            "json_valid:a",
            "similarity:a",
            "required_fields:a",
            "latency_budget:a",
            "schema_conformity:b",
            "content_bounds:b",
            "json_valid:b",
            "similarity:b",
            "required_fields:b",
            "latency_budget:b",
        ]

    def test_validate_json_schema_subset(self):
        schema = {
            "type": "object",
            "required": ["id"],
            "properties": {
                "id": {"type": "integer"},
                "tag": {"type": "string", "enum": ["a", "b"]},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        assert validate_json_schema(
            {"id": 1, "tag": "a", "score": 0.5, "tags": ["x"]}, schema
        ) == []
        errors = validate_json_schema({"id": True, "tag": "z", "score": 2}, schema)
        assert any("missing required" not in e for e in errors)
        assert any("not in enum" in e for e in errors)
        assert any("maximum" in e for e in errors)
        assert any("expected type integer" in e for e in errors)

    def test_conformity_errors_public_helper(self):
        assert conformity_errors({"a": 1}, {"type": "object"}) == []
        assert conformity_errors([], {"type": "object"}) == [
            "$: expected type object, got list"
        ]

    def test_hash_embedding_deterministic_and_monotone(self):
        embedder = HashEmbeddingClient()
        first = embedder.embed("refund processed in five days")
        assert first == embedder.embed("refund processed in five days")
        assert len(first) == 256
        assert embedder.similarity("same", "same") == 1.0
        related = embedder.similarity(
            "refund processed in 5 days", "refund processed in 5 business days"
        )
        disjoint = embedder.similarity(
            "refund processed in 5 days", "whales migrate across oceans yearly"
        )
        assert disjoint == 0.0  # no shared content tokens -> orthogonal vectors
        assert 0.0 <= disjoint < related < 1.0
        assert embedder.similarity("", "anything") == 0.0


# ---------------------------------------------------------------------------
# HTML report


class TestHtmlDiffReport:
    def _audit(self):
        steps = [
            AgentStep(
                name="extractor",
                output={"ticket_id": "T1", "priority": "high", "customer": "acme"},
            ),
            # triage mutates priority, adds 'queue', drops 'customer'
            passthrough("triage", {"ticket_id": "T1", "priority": "urgent",
                                   "queue": "billing"}),
            passthrough("resolver", {"ticket_id": "T1"}),
        ]
        return HandoffAuditor().audit(steps)

    def test_report_is_self_contained_and_deterministic(self, tmp_path):
        out_a, out_b = tmp_path / "a.html", tmp_path / "b.html"
        html_a = HtmlDiffReport.generate(self._audit(), out_a)
        html_b = HtmlDiffReport.generate(self._audit(), out_b)
        assert html_a == html_b
        assert out_a.read_text(encoding="utf-8") == html_a
        assert "<script" not in html_a
        assert "http://" not in html_a and "https://" not in html_a

    def test_report_contains_dropped_mutated_kept_rows(self, tmp_path):
        out = tmp_path / "report.html"
        html = HtmlDiffReport.generate(self._audit(), out)
        assert 'class="cell-dropped"' in html  # 'customer' dropped at seam 1
        assert "dropped" in html
        assert 'class="cell-mutated"' in html  # priority high -> urgent
        assert "high → urgent" in html
        assert 'class="cell-kept"' in html  # ticket_id kept
        assert 'class="cell-added"' in html
        assert "user" not in html  # no phantom rows
        assert "fidelity halflife" in html
        assert "Blame attribution" in html
