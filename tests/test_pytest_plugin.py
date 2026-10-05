"""The pytest plugin: the ``verdict`` fixture, exercised through pytest's own
``pytester`` harness (in-process, offline).

The plugin is registered as a ``pytest11`` entry point, so child runs pick it
up automatically; a couple of direct unit tests cover the baseline loader.
"""

from __future__ import annotations

import json
import textwrap

import pytest

pytest_plugins = ["pytester"]

# extractor drops 'customer' at seam 1 and 'priority' at seam 2
FAILING_CHAIN = """
STEPS = [
    {"name": "extractor", "output": {"ticket_id": "T1", "priority": "high", "customer": "acme"}},
    {"name": "triage", "input": {"ticket_id": "T1", "priority": "high"},
     "output": {"ticket_id": "T1", "priority": "high"}},
    {"name": "resolver", "input": {"ticket_id": "T1"},
     "output": {"ticket_id": "T1"}},
]
"""


def _make_chain_file(pytester, test_body: str) -> None:
    """Write a child test module with the planted degrading chain at column 0.

    Both pieces are dedented *separately* before concatenation: the chain
    lives at column 0, so pytester's whole-file dedent would otherwise leave
    the test function over-indented and break collection.
    """
    pytester.makepyfile(
        textwrap.dedent(FAILING_CHAIN) + "\n" + textwrap.dedent(test_body)
    )


class TestPluginViaPytester:
    def test_audit_fixture_returns_chain_audit(self, pytester):
        _make_chain_file(
            pytester,
            """
            import pytest
            from verdictai.seams import ChainAudit


            def test_audit(verdict):
                audit = verdict.audit(STEPS)
                assert isinstance(audit, ChainAudit)
                assert audit.step_names == ["extractor", "triage", "resolver"]
                assert audit.step_fidelity == [None, pytest.approx(2 / 3), 0.5]
                assert audit.handoffs[0].dropped_fields == ["customer"]
            """,
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_assert_chain_passes_healthy_chain(self, pytester):
        pytester.makepyfile(
            """
            def test_chain(verdict):
                steps = [
                    {"name": "a", "output": {"x": 1, "y": 2}},
                    {"name": "b", "input": {"x": 1, "y": 2}, "output": {"x": 1, "y": 2}},
                ]
                audit = verdict.assert_chain(steps, min_fidelity=0.7)
                assert audit.chain_fidelity == 1.0
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_assert_chain_failure_names_dropped_field_and_seam(self, pytester):
        _make_chain_file(
            pytester,
            """
            def test_chain(verdict):
                verdict.assert_chain(STEPS, min_fidelity=0.9)
            """,
        )
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(
            [
                "*chain fidelity*< min_fidelity 0.9*",
                "*seam extractor -> triage: fidelity*",
                "*dropped fields: customer*",
                "*seam triage -> resolver: fidelity*",
                "*dropped fields: priority*",
            ]
        )

    def test_assert_chain_reports_mutations(self, pytester):
        pytester.makepyfile(
            """
            def test_chain(verdict):
                steps = [
                    {"name": "a", "output": {"status": "high"}},
                    {"name": "b", "input": {"status": "urgent"}, "output": {}},
                ]
                verdict.assert_chain(steps, min_fidelity=0.99)
            """
        )
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(
            ["*mutated: status ('high' -> 'urgent')*"]
        )

    def test_assert_conformant_failure_lists_violations(self, pytester):
        pytester.makepyfile(
            """
            def test_conformant(verdict):
                verdict.assert_conformant(
                    {"ticket_id": "T-1"},
                    {"type": "object", "required": ["ticket_id", "priority"]},
                )
            """
        )
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(
            ["*does not conform to schema*", "*missing required field 'priority'*"]
        )

    def test_assert_conformant_accepts_pydantic_models(self, pytester):
        pytester.makepyfile(
            """
            from pydantic import BaseModel

            class Answer(BaseModel):
                ticket_id: str

            def test_conformant(verdict):
                verdict.assert_conformant({"ticket_id": "T-1"}, Answer)
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_assert_no_regression_pass_and_fail(self, pytester):
        pytester.makepyfile(
            """
            import json

            def _baseline(tmp_path, payload):
                path = tmp_path / "baseline.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            def test_same(verdict, tmp_path):
                baseline = _baseline(tmp_path, {"summary": "refund processed in 5 days"})
                # case/punctuation noise only -> same tokens -> no regression
                verdict.assert_no_regression(
                    {"summary": "Refund processed in 5 days!"}, baseline
                )

            def test_missing(verdict, tmp_path):
                baseline = _baseline(tmp_path, {"summary": "s1", "payload": "s2"})
                verdict.assert_no_regression({"summary": "s1"}, baseline)
            """
        )
        result = pytester.runpytest()
        assert result.ret == 1
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(["*output 'payload' missing (present in baseline)*"])

    def test_assert_no_regression_detects_text_drift(self, pytester):
        pytester.makepyfile(
            """
            import json

            def test_drift(verdict, tmp_path):
                baseline = tmp_path / "baseline.json"
                baseline.write_text(json.dumps({"summary": "refund processed in 5 days"}))
                verdict.assert_no_regression(
                    {"summary": "the weather in paris is sunny and warm"}, baseline
                )
            """
        )
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(["*drifted from baseline: cosine*"])


class TestBaselineLoader:
    def test_loads_json_object_and_jsonl(self, tmp_path):
        from verdictai.pytest_plugin import _load_baseline

        obj = tmp_path / "obj.json"
        obj.write_text(json.dumps({"a": 1, "b": "text"}), encoding="utf-8")
        assert _load_baseline(obj) == {"a": 1, "b": "text"}

        jsonl = tmp_path / "rows.jsonl"
        jsonl.write_text(
            '{"name": "a", "value": 1}\n{"name": "b", "value": {"x": 2}}\n',
            encoding="utf-8",
        )
        assert _load_baseline(jsonl) == {"a": 1, "b": {"x": 2}}

    def test_rejects_empty_and_unrecognizable(self, tmp_path):
        from verdictai.pytest_plugin import _load_baseline

        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(AssertionError, match="empty"):
            _load_baseline(empty)

        # JSONL rows without a name carry nothing usable
        junk = tmp_path / "junk.jsonl"
        junk.write_text('{"value": 1}\n{"value": 2}\n', encoding="utf-8")
        with pytest.raises(AssertionError, match="name/value"):
            _load_baseline(junk)
