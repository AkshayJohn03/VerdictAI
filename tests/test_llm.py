"""LLM layer: JSON extraction, OpenAI-compatible client over MockTransport,
echo mock determinism, and settings env handling. No network anywhere."""

from __future__ import annotations

import json

import httpx
import pytest

from verdictai.config import Settings
from verdictai.errors import LLMError, LLMResponseError
from verdictai.llm import EchoMockClient, OpenAICompatClient, complete_json, extract_json_object
from verdictai.prompts import build_rubric_prompt
from verdictai.types import Criterion, JudgeItem


class TestExtractJsonObject:
    def test_plain(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_with_prose(self):
        assert extract_json_object('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}

    def test_garbage_raises(self):
        with pytest.raises(LLMResponseError):
            extract_json_object("no json here")


class TestOpenAICompatClient:
    def _settings(self, **overrides):
        defaults = dict(
            api_key="test-key", base_url="http://testserver/v1", model="test-model",
            retry_backoff_seconds=0.0, max_retries=1,
        )
        defaults.update(overrides)
        return Settings(**defaults)

    def test_complete_builds_expected_request(self, run_async):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "hello"}}]}
            )

        client = OpenAICompatClient(self._settings(), transport=httpx.MockTransport(handler))
        text = run_async(client.complete("hi", system="be brief", temperature=0.3))
        run_async(client.aclose())
        assert text == "hello"
        assert captured["auth"] == "Bearer test-key"
        assert captured["body"]["model"] == "test-model"
        assert captured["body"]["messages"][0] == {"role": "system", "content": "be brief"}
        assert captured["body"]["temperature"] == 0.3

    def test_json_mode_sets_response_format_and_parses(self, run_async):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}
            )

        client = OpenAICompatClient(self._settings(), transport=httpx.MockTransport(handler))
        data = run_async(complete_json(client, "give me json"))
        run_async(client.aclose())
        assert data == {"ok": True}
        assert captured["body"]["response_format"] == {"type": "json_object"}

    def test_retries_on_500_then_succeeds(self, run_async):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = OpenAICompatClient(self._settings(), transport=httpx.MockTransport(handler))
        text = run_async(client.complete("hi"))
        run_async(client.aclose())
        assert text == "ok"
        assert attempts["n"] == 2

    def test_retries_exhausted_raises_llm_error(self, run_async):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        client = OpenAICompatClient(self._settings(), transport=httpx.MockTransport(handler))
        with pytest.raises(LLMError):
            run_async(client.complete("hi"))
        run_async(client.aclose())

    def test_hard_4xx_raises_without_retry(self, run_async):
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(401, text="unauthorized")

        client = OpenAICompatClient(self._settings(), transport=httpx.MockTransport(handler))
        with pytest.raises(LLMError):
            run_async(client.complete("hi"))
        run_async(client.aclose())
        assert attempts["n"] == 1


class TestEchoMockClient:
    def test_rubric_payload_deterministic(self, run_async):
        criteria = [Criterion(name="accuracy", description="d", weight=1.0)]
        prompt = build_rubric_prompt(
            JudgeItem(item_id="m1", prompt="p", response="some response text"), criteria
        )
        first = run_async(EchoMockClient().complete(prompt))
        second = run_async(EchoMockClient().complete(prompt))
        assert first == second
        payload = json.loads(first)
        assert payload["scores"][0]["name"] == "accuracy"
        assert payload["scores"][0]["score"] == 3.0

    def test_pairwise_payload(self, run_async):
        from verdictai.prompts import build_pairwise_prompt
        from verdictai.types import PairwiseItem

        prompt = build_pairwise_prompt(
            PairwiseItem(item_id="m2", prompt="p", response_a="long " * 10, response_b="short"),
            "ab",
        )
        data = json.loads(run_async(EchoMockClient().complete(prompt)))
        assert data["winner"] == "first"


class TestSettings:
    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("VERDICTAI_MODEL", "custom-model")
        monkeypatch.setenv("VERDICTAI_API_KEY", "sk-test")
        settings = Settings(_env_file=None)
        assert settings.model == "custom-model"
        assert settings.has_api_key is True

    def test_defaults_are_offline_safe(self, monkeypatch):
        for var in ("VERDICTAI_API_KEY", "VERDICTAI_MODEL"):
            monkeypatch.delenv(var, raising=False)
        settings = Settings(_env_file=None)
        assert settings.has_api_key is False
        assert settings.model == "gpt-4o-mini"
