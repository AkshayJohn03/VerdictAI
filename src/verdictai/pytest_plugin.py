"""pytest integration: the ``verdict`` fixture.

Registered as a ``pytest11`` entry point (see ``pyproject.toml``), so after
``pip install -e .`` every pytest run can simply ask for the fixture::

    def test_pipeline(verdict):
        verdict.assert_chain(steps, min_fidelity=0.7)
        verdict.assert_conformant(output, schema)
        verdict.assert_no_regression(outputs, "baseline.json")

The plugin module imports nothing heavy at load time — pytest loads it in
every run via the entry point, and the VerdictAI machinery is only imported
when a test actually uses the fixture. Fixtures that are never requested
cost nothing.

Assertion messages are seam-explicit: they name the step pair, the exact
dot-notation fields that were dropped/mutated, and the fidelity numbers, so
a failure reads like a diagnosis instead of a stack trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


class VerdictClient:
    """Developer-facing assertion surface, exposed through the ``verdict`` fixture."""

    def audit(self, steps: list[Any]) -> Any:
        """Audit a chain of steps (dicts or AgentStep) -> ChainAudit."""
        from .seams import AgentStep, HandoffAuditor

        normalized: list[AgentStep] = []
        for index, step in enumerate(steps):
            if isinstance(step, AgentStep):
                normalized.append(step)
            elif isinstance(step, dict):
                payload = {**step}
                if "name" not in payload:
                    payload["name"] = payload.get("step", f"step-{index}")
                normalized.append(AgentStep.model_validate(payload))
            else:
                raise TypeError(
                    f"step {index} must be a dict or AgentStep, got {type(step).__name__}"
                )
        return HandoffAuditor().audit(normalized)

    def assert_chain(self, steps: list[Any], min_fidelity: float = 0.7) -> Any:
        """Assert a chain preserves at least ``min_fidelity`` across its seams.

        Returns the :class:`~verdictai.seams.ChainAudit` on success; raises an
        ``AssertionError`` naming every lossy seam and its dropped fields.
        """
        chain_audit = self.audit(steps)
        if chain_audit.chain_fidelity >= min_fidelity:
            return chain_audit
        lines = [
            f"chain fidelity {chain_audit.chain_fidelity:.3f} < min_fidelity "
            f"{min_fidelity} across {len(chain_audit.handoffs)} seam(s)"
        ]
        if chain_audit.fidelity_halflife is not None:
            lines.append(
                f"fidelity halflife: {chain_audit.fidelity_halflife} handoff(s) "
                "before preservation dropped below 0.5"
            )
        for diff in chain_audit.handoffs:
            if (
                not diff.dropped_fields
                and not diff.mutated
                and not diff.type_changed
            ):
                continue
            lines.append(
                f"seam {diff.from_step} -> {diff.to_step}: "
                f"fidelity {diff.fidelity:.3f}"
            )
            if diff.dropped_fields:
                lines.append(f"  dropped fields: {', '.join(diff.dropped_fields)}")
            for mutation in diff.mutated:
                lines.append(
                    f"  mutated: {mutation.field} "
                    f"({mutation.before!r} -> {mutation.after!r})"
                )
            for change in diff.type_changed:
                lines.append(
                    f"  type changed: {change.field} "
                    f"({change.before_type} -> {change.after_type})"
                )
        lines.append("HTML diff: verdict audit --steps trace.json --html report.html")
        raise AssertionError("\n".join(lines))

    def assert_conformant(self, output: Any, schema: Any) -> None:
        """Assert ``output`` conforms to a JSON-schema dict or pydantic model."""
        from .deterministic import conformity_errors

        errors = conformity_errors(output, schema)
        if errors:
            lines = [f"output does not conform to schema ({len(errors)} violation(s)):"]
            lines += [f"  - {error}" for error in errors[:10]]
            raise AssertionError("\n".join(lines))

    def assert_no_regression(
        self,
        outputs: dict[str, Any],
        baseline_path: str | Path,
        *,
        min_similarity: float = 0.85,
    ) -> None:
        """Assert ``outputs`` did not regress against a recorded baseline file.

        The baseline is a JSON object ``{name: value}`` (or JSONL rows with
        ``name``/``value``). Missing names always fail; strings are compared
        by offline cosine similarity, everything else by exact value.
        """
        from .deterministic import HashEmbeddingClient

        baseline = _load_baseline(Path(baseline_path))
        problems: list[str] = []
        for name in sorted(baseline):
            if name not in outputs:
                problems.append(f"output {name!r} missing (present in baseline)")
                continue
            expected, actual = baseline[name], outputs[name]
            if isinstance(expected, str) and isinstance(actual, str):
                score = HashEmbeddingClient().similarity(actual, expected)
                if score < min_similarity:
                    problems.append(
                        f"output {name!r} drifted from baseline: cosine {score:.3f} "
                        f"< {min_similarity} (baseline: {_short(expected)})"
                    )
            elif _stable_json(expected) != _stable_json(actual):
                problems.append(
                    f"output {name!r} differs from baseline:\n"
                    f"    baseline: {_short(expected)}\n"
                    f"    actual:   {_short(actual)}"
                )
        if problems:
            raise AssertionError(
                f"{len(problems)} regression(s) vs {Path(baseline_path)}:\n"
                + "\n".join(f"  - {problem}" for problem in problems)
            )


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _load_baseline(path: Path) -> dict[str, Any]:
    """Load a baseline written as a JSON object or JSONL name/value rows."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise AssertionError(f"baseline file is empty: {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        inner = payload.get("outputs", payload)
        if isinstance(inner, dict):
            return dict(inner)
    if isinstance(payload, list):
        rows = payload
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    baseline: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, dict) and "name" in row:
            baseline[str(row["name"])] = row.get("value", row.get("output"))
    if not baseline:
        raise AssertionError(
            f"baseline file {path} must be {{name: value}} JSON or JSONL "
            "rows with name/value"
        )
    return baseline


@pytest.fixture
def verdict() -> VerdictClient:
    """VerdictAI seam auditor + deterministic assertions for agent pipelines."""
    return VerdictClient()
