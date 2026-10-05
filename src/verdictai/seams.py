"""The Seam Auditor: deterministic fidelity analysis of multi-agent handoffs.

Single-hop evaluation (query -> context -> answer) cannot see the compounding
information loss that happens *between* agents. When Agent A hands a dict to
Agent B, every field A produced that B fails to carry forward is a silent
degradation — and five seams later the chain has "forgotten" the ticket id it
started with.

This module measures that loss with **zero LLM calls**:

- :class:`HandoffAuditor.audit` diffs every seam (``prev.output`` vs
  ``next.input``) and scores a per-handoff **fidelity** in ``[0, 1]``:
  the fraction of the previous step's leaf fields that are preserved in the
  next step's input (or are independently recoverable from it).
- :attr:`ChainAudit.chain_fidelity` is the geometric mean across seams, so a
  single catastrophic seam collapses the whole chain (0.9 * 0.9 * 0.1 < 0.3).
- :attr:`ChainAudit.fidelity_halflife` is the number of handoffs until the
  field-preservation ratio first drops below 0.5 (``None`` if it never does).
- :meth:`ChainAudit.blame` ranks the worst seam by
  ``(1 - fidelity) * downstream_steps`` — the node that both lost the most and
  poisoned the most subsequent work.

Everything is deterministic: nested dicts flatten to dot-notation keys, lists
are compared by length plus the first few sampled elements, and a "dropped"
field whose value still appears anywhere in the next input is treated as a
rename (recoverable), not a loss.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

#: How many list elements are sampled when flattening (index >= this is only
#: covered by the list-length comparison).
LIST_SAMPLE = 3

#: Pseudo-leaf suffix used to report a list-length change as a field.
LEN_SUFFIX = "#len"


def _type_name(value: Any) -> str:
    """Stable leaf-type label (int/float unify as 'number' to avoid noise)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def flatten_fields(
    value: Any, prefix: str = ""
) -> tuple[dict[str, Any], dict[str, int]]:
    """Flatten a dict/list structure into dot-notation leaf keys.

    Returns ``(leaves, list_lengths)`` where ``leaves`` maps every scalar leaf
    to its dot-notation path (nested dicts joined with ``.``, list elements
    sampled as ``path[0]``, ``path[1]``, ...) and ``list_lengths`` maps each
    list path to its length so shrink/grow is comparable without sampling.
    """
    leaves: dict[str, Any] = {}
    lengths: dict[str, int] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            sub_leaves, sub_lengths = flatten_fields(item, path)
            leaves.update(sub_leaves)
            lengths.update(sub_lengths)
    elif isinstance(value, list):
        if prefix:
            lengths[prefix] = len(value)
        for index, item in enumerate(value[:LIST_SAMPLE]):
            sub_leaves, sub_lengths = flatten_fields(item, f"{prefix}[{index}]")
            leaves.update(sub_leaves)
            lengths.update(sub_lengths)
    else:
        leaves[prefix] = value
    return leaves, lengths


class AgentStep(BaseModel):
    """One node in an agent chain: what it received and what it produced."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None


class FieldMutation(BaseModel):
    """A field carried forward but with a changed value (or list length)."""

    field: str
    before: Any
    after: Any


class TypeChange(BaseModel):
    """A field carried forward under a different scalar type."""

    field: str
    before_type: str
    after_type: str


class HandoffDiff(BaseModel):
    """Diff across one seam: what ``from_step`` produced vs what ``to_step`` took."""

    from_step: str
    to_step: str
    dropped_fields: list[str] = Field(default_factory=list)
    added_fields: list[str] = Field(default_factory=list)
    mutated: list[FieldMutation] = Field(default_factory=list)
    type_changed: list[TypeChange] = Field(default_factory=list)
    fidelity: float = 1.0


class BlameEntry(BaseModel):
    """One seam's ranked contribution to chain-level degradation."""

    from_step: str
    to_step: str
    fidelity: float
    downstream_steps: int
    severity: float


class ChainAudit(BaseModel):
    """Result of auditing a whole chain of agent steps."""

    steps: list[AgentStep] = Field(default_factory=list)
    handoffs: list[HandoffDiff] = Field(default_factory=list)
    #: Fidelity of the handoff *into* each step (``None`` for the first step).
    step_fidelity: list[float | None] = Field(default_factory=list)
    chain_fidelity: float = 1.0
    #: Handoffs survived before field preservation first drops below 0.5.
    fidelity_halflife: int | None = None

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    def blame(self) -> list[BlameEntry]:
        """Rank seams worst-first by ``(1 - fidelity) * downstream_steps``.

        The seam with the lowest fidelity weighted by how many subsequent
        steps it poisoned comes first; ties break by earlier seam (stable).
        """
        n_handoffs = len(self.handoffs)
        order = {name: i for i, name in enumerate(self.step_names)}
        entries: list[BlameEntry] = []
        for index, diff in enumerate(self.handoffs):
            downstream = n_handoffs - index  # to_step and every step after it
            entries.append(
                BlameEntry(
                    from_step=diff.from_step,
                    to_step=diff.to_step,
                    fidelity=diff.fidelity,
                    downstream_steps=downstream,
                    severity=(1.0 - diff.fidelity) * downstream,
                )
            )
        entries.sort(key=lambda e: (-e.severity, order.get(e.to_step, 0)))
        return entries


def _diff_handoff(prev: AgentStep, nxt: AgentStep) -> HandoffDiff:
    """Diff one seam deterministically (no LLM, no randomness)."""
    prev_leaves, prev_lists = flatten_fields(prev.output)
    next_leaves, next_lists = flatten_fields(nxt.input)

    # List lengths participate in the diff as pseudo-leaves ("items#len").
    for path, length in prev_lists.items():
        prev_leaves[f"{path}{LEN_SUFFIX}"] = length
    for path, length in next_lists.items():
        next_leaves[f"{path}{LEN_SUFFIX}"] = length

    # Leaves are scalars by construction (dicts/lists are always flattened).
    next_values = set(next_leaves.values())

    dropped: list[str] = []
    recoverable = 0
    mutated: list[FieldMutation] = []
    type_changed: list[TypeChange] = []
    kept = 0

    for field, before in prev_leaves.items():
        if field not in next_leaves:
            # A value that still surfaces anywhere in the next input was
            # renamed/re-derived, not lost: it does not reduce fidelity.
            if before is not None and before != "" and before in next_values:
                recoverable += 1
            else:
                dropped.append(field)
            continue
        after = next_leaves[field]
        if _type_name(before) != _type_name(after):
            type_changed.append(
                TypeChange(
                    field=field,
                    before_type=_type_name(before),
                    after_type=_type_name(after),
                )
            )
        elif before != after:
            # A list that *grew* is enrichment, not degradation: count it as
            # preserved rather than mutated.
            grew = (
                field.endswith(LEN_SUFFIX)
                and isinstance(before, int)
                and isinstance(after, int)
                and after > before
            )
            if grew:
                kept += 1
            else:
                mutated.append(FieldMutation(field=field, before=before, after=after))
        else:
            kept += 1

    total = len(prev_leaves)
    fidelity = (kept + recoverable) / total if total else 1.0

    added = [field for field in next_leaves if field not in prev_leaves]
    return HandoffDiff(
        from_step=prev.name,
        to_step=nxt.name,
        dropped_fields=dropped,
        added_fields=added,
        mutated=mutated,
        type_changed=type_changed,
        fidelity=round(fidelity, 6),
    )


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    if any(v <= 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


class HandoffAuditor:
    """Audit a chain of :class:`AgentStep` for handoff degradation."""

    def audit(self, steps: list[AgentStep]) -> ChainAudit:
        """Diff every consecutive seam and score chain-level fidelity."""
        diffs = [
            _diff_handoff(steps[i], steps[i + 1]) for i in range(len(steps) - 1)
        ]
        fidelities = [diff.fidelity for diff in diffs]
        halflife: int | None = None
        for index, fidelity in enumerate(fidelities):
            if fidelity < 0.5:
                halflife = index + 1  # handoffs survived (1-based)
                break
        return ChainAudit(
            steps=list(steps),
            handoffs=diffs,
            step_fidelity=[None, *fidelities],
            chain_fidelity=round(_geometric_mean(fidelities), 6),
            fidelity_halflife=halflife,
        )


def _steps_from_rows(rows: list[Any]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for index, row in enumerate(rows):
        if isinstance(row, AgentStep):
            steps.append(row)
        elif isinstance(row, dict):
            if "name" not in row:
                row = {**row, "name": row.get("step", f"step-{index}")}
            steps.append(AgentStep.model_validate(row))
        else:
            raise ValueError(
                f"trace row {index} must be an object with name/input/output, "
                f"got {type(row).__name__}"
            )
    return steps


def audit_json(trace_file: str | Path) -> ChainAudit:
    """Load a JSON / JSONL trace of agent steps and audit it.

    Accepts a bare list of step objects, an object with a ``"steps"`` (or
    ``"trace"``) key, or JSONL with one step object per line.
    """
    text = Path(trace_file).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"trace file is empty: {trace_file}")
    rows: list[Any] | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        inner = payload.get("steps", payload.get("trace"))
        if isinstance(inner, list):
            rows = inner
    if rows is None:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace line {line_no} is not valid JSON: {exc}") from exc
    steps = _steps_from_rows(rows)
    return HandoffAuditor().audit(steps)
