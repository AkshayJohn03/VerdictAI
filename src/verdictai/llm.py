"""LLM access layer.

All model traffic in VerdictAI goes through the :class:`LLMClient` protocol:

- :class:`OpenAICompatClient` — any OpenAI-compatible chat completions endpoint
  (OpenAI, vLLM, Ollama, Azure-compatible gateways, ...), with retries and
  JSON mode.
- :class:`EchoMockClient` — a deterministic offline stand-in that answers from
  the prompt's VerdictAI payload markers, so the whole pipeline runs with zero
  network access and zero cost.

``complete_json`` is the JSON-mode helper: it requests JSON mode, strips code
fences and parses the first JSON object found in the response.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import LLMError, LLMResponseError
from .prompts import parse_payload

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@runtime_checkable
class LLMClient(Protocol):
    """Protocol every LLM backend must satisfy."""

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant message for ``prompt``."""
        ...


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerates markdown fences)."""
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMResponseError(f"no JSON object found in model response: {text[:200]!r}")
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"invalid JSON in model response: {exc}") from exc


async def complete_json(client: LLMClient, prompt: str, **kwargs: Any) -> dict[str, Any]:
    """Ask ``client`` for JSON and return the parsed object."""
    text = await client.complete(prompt, json_mode=True, **kwargs)
    return extract_json_object(text)


class OpenAICompatClient:
    """Async client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout_seconds,
            transport=transport,
        )

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        s = self._settings
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or s.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(s.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in _RETRYABLE_STATUS and attempt < s.max_retries:
                    await asyncio.sleep(s.retry_backoff_seconds * (2**attempt))
                    continue
                if resp.status_code >= 400:
                    raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                if attempt >= s.max_retries:
                    break
                await asyncio.sleep(s.retry_backoff_seconds * (2**attempt))
        raise LLMError(f"LLM call failed after retries: {last_error}")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _first_words(text: str, k: int) -> str:
    words = text.split()
    return " ".join(words[:k])


class EchoMockClient:
    """Deterministic offline LLM.

    Answers are derived from the VerdictAI payload markers embedded in prompts:

    - ``rubric``: scores every criterion with a fixed baseline score.
    - ``pairwise``: prefers the more substantive (longer) response; ties when
      lengths are equal. Deterministic, so it cannot plant position bias.
    - ``reference``: fixed baseline score.
    - ``generate``: synthesizes a question/reference pair from the spec.

    This exists so demos, tests and CI run without network or cost. It is NOT
    a quality signal — swap in a real client via ``VERDICTAI_API_KEY``.
    """

    def __init__(self, default_score: float = 3.0):
        self.default_score = default_score

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        kind, payload = parse_payload(prompt)
        if kind == "rubric":
            return json.dumps(
                {
                    "scores": [
                        {
                            "name": c.get("name", f"criterion-{i}"),
                            "score": self.default_score,
                            "evidence": _first_words(str(payload.get("response", "")), 12),
                            "rationale": "Deterministic mock baseline score.",
                        }
                        for i, c in enumerate(payload.get("criteria", []))
                    ]
                },
                sort_keys=True,
            )
        if kind == "pairwise":
            first = str(payload.get("response_first", ""))
            second = str(payload.get("response_second", ""))
            if len(first) > len(second):
                winner = "first"
            elif len(second) > len(first):
                winner = "second"
            else:
                winner = "tie"
            return json.dumps(
                {"winner": winner, "rationale": "Mock prefers the more substantive response."},
                sort_keys=True,
            )
        if kind == "reference":
            return json.dumps(
                {"score": self.default_score, "rationale": "Deterministic mock reference score."},
                sort_keys=True,
            )
        if kind == "generate":
            topic = str(payload.get("topic", "the topic"))
            persona = str(payload.get("persona", "a user"))
            keywords = payload.get("keywords", [])
            return json.dumps(
                {
                    "question": (
                        f"[mock] As {persona}, handle a {payload.get('difficulty', 'easy')} "
                        f"request about {topic}."
                    ),
                    "reference": f"[mock reference] {topic}: " + ", ".join(map(str, keywords)),
                },
                sort_keys=True,
            )
        return "OK"
