"""Shared test doubles and fixtures. The whole suite is offline: no test ever
touches the network (httpx.MockTransport and scripted fake clients only)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Safety net so tests import the package even without the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verdictai.regression.runner import ScoreRecord, ScoreSnapshot  # noqa: E402


def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio dependency)."""
    return asyncio.run(coro)


@pytest.fixture
def run_async():
    return run


class ScriptedJSONClient:
    """Fake LLMClient returning canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=None,
                       json_mode=False):
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "json_mode": json_mode}
        )
        if not self.responses:
            raise AssertionError("ScriptedJSONClient ran out of canned responses")
        return self.responses.pop(0)


class FunctionClient:
    """Fake LLMClient backed by a prompt -> response function."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[dict] = []

    async def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=None,
                       json_mode=False):
        self.calls.append({"temperature": temperature, "json_mode": json_mode})
        return self.fn(prompt)


class PairwiseRigClient:
    """Fake pairwise LLM: decides winners via callable(first_response, second_response)."""

    def __init__(self, decide):
        self.decide = decide

    async def complete(self, prompt, **kwargs):
        from verdictai.prompts import parse_payload

        kind, payload = parse_payload(prompt)
        assert kind == "pairwise"
        winner = self.decide(payload["response_first"], payload["response_second"])
        return json.dumps({"winner": winner, "rationale": "test rig"})


@pytest.fixture
def scripted_client():
    return ScriptedJSONClient


@pytest.fixture
def function_client():
    return FunctionClient


@pytest.fixture
def pairwise_rig():
    return PairwiseRigClient


@pytest.fixture
def snapshot_factory():
    """Build ScoreSnapshots with planted per-item scores."""

    def _make(model_version, scores, capabilities=None, dataset_version="v1", judge_type="test"):
        records = []
        for i, score in enumerate(scores):
            records.append(
                ScoreRecord(
                    item_id=f"item-{i:03d}",
                    capability=(capabilities[i] if capabilities else "general"),
                    score=float(score),
                    judge_type=judge_type,
                )
            )
        return ScoreSnapshot(
            model_version=model_version,
            dataset_version=dataset_version,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            judge_type=judge_type,
            records=records,
        )

    return _make
