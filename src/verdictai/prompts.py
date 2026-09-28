"""Prompt builders for every LLM-backed judge and the dataset generator.

Design note: every prompt ends with two machine-readable marker lines::

    VERDICTAI_TASK_KIND: rubric
    VERDICTAI_PAYLOAD_JSON: {...}

The payload is the exact machine-readable input (criteria, responses, ...).
Real LLMs treat these as part of the instructions; :class:`~verdictai.llm.
EchoMockClient` parses them to produce deterministic offline responses, and
tests use them to script fake clients without re-implementing prompt formats.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import VerdictAIError
from .types import Criterion, JudgeItem, PairwiseItem

TASK_KIND_MARKER = "VERDICTAI_TASK_KIND:"
PAYLOAD_MARKER = "VERDICTAI_PAYLOAD_JSON:"

_RUBRIC_INSTRUCTIONS = (
    "You are a strict, calibrated evaluator of AI responses.\n"
    "Score the RESPONSE below on every criterion.\n"
    "Rules:\n"
    "- score is an integer or float in [0, 5]; use the anchors to calibrate.\n"
    "- evidence must be a short verbatim quote from the RESPONSE.\n"
    "- rationale must be one sentence.\n"
    '- Reply with JSON only: {"scores": [{"name", "score", "evidence", "rationale"}]}\n'
)

_PAIRWISE_INSTRUCTIONS = (
    "You are a strict, impartial evaluator. Two responses to the same question "
    "are given below as RESPONSE_FIRST and RESPONSE_SECOND.\n"
    "Rules:\n"
    "- Judge substance only: correctness, completeness, and instruction following.\n"
    "- Ignore presentation order, length, and formatting polish unless the question "
    "explicitly asks for that format.\n"
    '- Reply with JSON only: {"winner": "first" | "second" | "tie", "rationale": "..."}\n'
    "- Use \"tie\" only when both responses are equally good on substance.\n"
)

_REFERENCE_INSTRUCTIONS = (
    "You are a strict evaluator. Compare the RESPONSE against the GOLDEN (reference) "
    "answer.\n"
    "Score how well the response matches the golden answer in content:\n"
    '- Reply with JSON only: {"score": <0..5>, "rationale": "..."}\n'
)

_GENERATION_INSTRUCTIONS = (
    "You are building an evaluation dataset. Produce one eval item for the given "
    "spec (topic, persona, difficulty, edge case).\n"
    '- Reply with JSON only: {"question": "...", "reference": "..."}\n'
    "- The reference is the ideal answer, not a rubric.\n"
    "- Honor the edge case semantics exactly.\n"
)


def _markers(kind: str, payload: dict[str, Any]) -> str:
    return (
        f"\n{TASK_KIND_MARKER} {kind}\n"
        f"{PAYLOAD_MARKER} {json.dumps(payload, ensure_ascii=True, sort_keys=True)}\n"
    )


def build_rubric_prompt(item: JudgeItem, criteria: list[Criterion]) -> str:
    lines: list[str] = [_RUBRIC_INSTRUCTIONS, "CRITERIA:"]
    for c in criteria:
        anchors = " ".join(f"{k}:{v}" for k, v in sorted(c.anchors.items()))
        line = f"- {c.name} (weight {c.weight}): {c.description}"
        if anchors:
            line += f" Anchors: {anchors}"
        lines.append(line)
    lines.append("\nRESPONSE:")
    lines.append(item.response if item.response.strip() else "(empty response)")
    payload = {
        "item_id": item.item_id,
        "criteria": [c.model_dump() for c in criteria],
        "response": item.response,
    }
    lines.append(_markers("rubric", payload))
    return "\n".join(lines)


def build_pairwise_prompt(item: PairwiseItem, order: str) -> str:
    if order == "ab":
        first, second = item.response_a, item.response_b
    elif order == "ba":
        first, second = item.response_b, item.response_a
    else:
        raise ValueError(f"order must be 'ab' or 'ba', got {order!r}")
    lines = [
        _PAIRWISE_INSTRUCTIONS,
        "QUESTION:",
        item.prompt if item.prompt.strip() else "(empty question)",
        "\nRESPONSE_FIRST:",
        first,
        "\nRESPONSE_SECOND:",
        second,
        _markers(
            "pairwise",
            {
                "item_id": item.item_id,
                "order": order,
                "question": item.prompt,
                "response_first": first,
                "response_second": second,
            },
        ),
    ]
    return "\n".join(lines)


def build_reference_prompt(item: JudgeItem) -> str:
    lines = [
        _REFERENCE_INSTRUCTIONS,
        "GOLDEN ANSWER:",
        item.reference,
        "\nRESPONSE:",
        item.response,
        _markers(
            "reference",
            {"item_id": item.item_id, "reference": item.reference, "response": item.response},
        ),
    ]
    return "\n".join(lines)


def build_generation_prompt(spec: dict[str, Any]) -> str:
    lines = [
        _GENERATION_INSTRUCTIONS,
        "SPEC:",
        json.dumps(spec, ensure_ascii=False, sort_keys=True),
        _markers("generate", spec),
    ]
    return "\n".join(lines)


def parse_payload(prompt: str) -> tuple[str, dict[str, Any]]:
    """Extract ``(task_kind, payload_dict)`` from the trailing marker lines."""
    kind_line = None
    payload_line = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(TASK_KIND_MARKER):
            kind_line = stripped[len(TASK_KIND_MARKER) :].strip()
        elif stripped.startswith(PAYLOAD_MARKER):
            payload_line = stripped[len(PAYLOAD_MARKER) :].strip()
    if kind_line is None or payload_line is None:
        raise VerdictAIError("prompt is missing VerdictAI task markers")
    return kind_line, json.loads(payload_line)
