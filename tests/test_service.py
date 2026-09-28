"""Eval-service HTTP API: full run lifecycle, gate pass/fail exit-code
semantics, HMAC-signed webhooks, API-key auth, correlation-id echo — the
whole suite stays offline (recording webhook fake, replay/mock adapters)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from verdictai.datasets.schema import DatasetItem, EvalDataset
from verdictai.regression.runner import MockModelAdapter
from verdictai.service import (
    RunRegistry,
    WebhookRegistration,
    _execute_run,
    build_webhook_payload,
    create_app,
    hash_api_key,
    load_api_key_hashes,
    sign_payload,
)

REFERENCES = [
    "The refund posts within five business days of return receipt.",
    "Escalate billing disputes to the payments team with the invoice id.",
    "Password resets expire after sixty minutes and never unlock two-factor.",
    "Shipping upgrades apply only before the warehouse picks the order.",
    "Tier discounts stack with coupons but never with clearance prices.",
    "Support handles refunds while sales owns quotes and enterprise trials.",
    "Data exports include attachments only when the audit flag is enabled.",
    "Rate limits reset on a rolling window rather than at midnight utc.",
    "Sandbox keys cannot reach production webhooks or live payment capture.",
    "The sla clock pauses while waiting for customer-supplied information.",
    "Duplicate invoices merge automatically when the po number matches.",
    "Trial extensions require owner approval and apply once per workspace.",
]


def dataset_jsonl() -> str:
    """12 golden items across two capabilities — enough for capability slices."""
    lines = []
    for i, reference in enumerate(REFERENCES):
        item = DatasetItem(
            item_id=f"svc-{i:03d}",
            prompt=f"Answer precisely: question {i} about the policy below.\n{reference}",
            reference=reference,
            capability="accuracy" if i % 2 == 0 else "tone",
            topic="policy",
            persona="support-agent",
            difficulty="medium",
            edge_case="ambiguity",
        )
        lines.append(item.model_dump_json())
    return "\n".join(lines)


def make_dataset(version: str = "svc-v1") -> EvalDataset:
    return EvalDataset.from_jsonl(dataset_jsonl(), version=version)


class RecordingSink:
    """WebhookSink test double: captures deliveries, never touches network."""

    def __init__(self) -> None:
        self.deliveries: list[dict[str, Any]] = []

    def deliver(self, url: str, body: bytes, headers: dict[str, str]) -> None:
        self.deliveries.append({"url": url, "body": body, "headers": headers})


class FailingAdapter:
    """Adapter whose generate raises — drives the status=failed path."""

    model_version = "boom-v1"

    async def generate(self, prompt: str, *, item_id: str = "") -> str:
        raise RuntimeError("boom: candidate adapter exploded")


def submit_run(client: TestClient, quality: float, model_version: str) -> str:
    """Submit one replay-adapter run; returns the run id."""
    response = client.post(
        "/v1/eval-runs",
        json={
            "dataset": {"jsonl": dataset_jsonl(), "version": "svc-v1"},
            "candidate": {
                "adapter": "replay",
                "params": {"quality": quality, "model_version": model_version},
            },
            "metadata": {"model_version": model_version, "triggered_by": "tests"},
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def wait_terminal(client: TestClient, run_id: str, timeout: float = 10.0) -> str:
    """Poll until the run reaches a terminal status (worker thread races us)."""
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/v1/eval-runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in ("succeeded", "failed"):
            return str(payload["status"])
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s: {payload}")


def verify_signature(delivery: dict[str, Any], secret: str) -> bool:
    """Receiver-side check: HMAC-SHA256 over the raw body with the secret."""
    sent = delivery["headers"].get("X-VerdictAI-Signature", "")
    expected = "sha256=" + hmac.new(secret.encode(), delivery["body"], hashlib.sha256).hexdigest()
    return hmac.compare_digest(sent, expected)


# ---------------------------------------------------------------------------
# Health + lifecycle + reports
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self):
        with TestClient(create_app()) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}


class TestLifecycle:
    def test_submit_poll_report(self):
        with TestClient(create_app()) as client:
            run_id = submit_run(client, quality=1.0, model_version="baseline-v1")
            assert wait_terminal(client, run_id) == "succeeded"

            run = client.get(f"/v1/eval-runs/{run_id}").json()
            assert run["status"] == "succeeded"
            assert run["metadata"]["triggered_by"] == "tests"
            metrics = run["metrics"]
            assert metrics is not None
            assert metrics["n_items"] == 12
            assert metrics["mean_score"] == pytest.approx(5.0)
            assert set(metrics["capability_means"]) == {"accuracy", "tone"}
            assert metrics["model_version"] == "baseline-v1"
            assert metrics["dataset_version"] == "svc-v1"

            report_json = client.get(f"/v1/eval-runs/{run_id}/report")
            assert report_json.status_code == 200
            body = report_json.json()
            assert body["report_type"] == "run"
            assert body["metrics"]["n_items"] == 12
            assert body["snapshot"]["model_version"] == "baseline-v1"

            report_md = client.get(f"/v1/eval-runs/{run_id}/report", params={"format": "markdown"})
            assert report_md.status_code == 200
            assert report_md.headers["content-type"].startswith("text/markdown")
            assert "## VerdictAI eval run report" in report_md.text

    def test_report_before_finish_conflicts(self):
        # A queued run that has not been executed yet must 409 on /report.
        app = create_app()
        record = app.state.registry.create(metadata={}, dataset_version="svc-v1")
        with TestClient(app) as client:
            response = client.get(f"/v1/eval-runs/{record.run_id}/report")
            assert response.status_code == 409

    def test_unknown_run_404_on_every_route(self):
        with TestClient(create_app()) as client:
            assert client.get("/v1/eval-runs/missing").status_code == 404
            assert client.get("/v1/eval-runs/missing/report").status_code == 404
            gate = client.post(
                "/v1/eval-runs/missing/gate", json={"baseline_run_id": "also-missing"}
            )
            assert gate.status_code == 404
            webhook = client.post(
                "/v1/eval-runs/missing/webhook",
                json={"url": "https://hooks.example.test/x", "secret": "s"},
            )
            assert webhook.status_code == 404


# ---------------------------------------------------------------------------
# Gate semantics (exit-code equivalent)
# ---------------------------------------------------------------------------


class TestGate:
    def test_gate_passes_vs_identical_baseline(self):
        with TestClient(create_app()) as client:
            baseline_id = submit_run(client, 1.0, "baseline-v1")
            candidate_id = submit_run(client, 1.0, "candidate-v1")
            assert wait_terminal(client, baseline_id) == "succeeded"
            assert wait_terminal(client, candidate_id) == "succeeded"

            gate = client.post(
                f"/v1/eval-runs/{candidate_id}/gate", json={"baseline_run_id": baseline_id}
            )
            assert gate.status_code == 200
            payload = gate.json()
            assert payload["passed"] is True
            assert payload["exit_code"] == 0  # CI-exit-code equivalent
            assert payload["verdict"] == "no_change"
            assert payload["evidence"]["metrics"]["mean_delta"] == pytest.approx(0.0)

    def test_gate_fails_vs_degraded_baseline(self):
        with TestClient(create_app()) as client:
            baseline_id = submit_run(client, 1.0, "baseline-v1")
            degraded_id = submit_run(client, 0.4, "candidate-degraded")
            assert wait_terminal(client, baseline_id) == "succeeded"
            assert wait_terminal(client, degraded_id) == "succeeded"

            gate = client.post(
                f"/v1/eval-runs/{degraded_id}/gate",
                json={"baseline_run_id": baseline_id, "threshold": 0.05},
            )
            assert gate.status_code == 200
            payload = gate.json()
            assert payload["passed"] is False
            assert payload["exit_code"] == 1  # regression blocks, like `verdict regress --gate`
            assert payload["verdict"] == "regressed"
            assert payload["evidence"]["metrics"]["mean_delta"] > 1.0
            assert payload["evidence"]["failing_capabilities"]
            assert "regression gate: FAILED" in payload["report_markdown"]

            # Once a gate ran, /report serves the gate report (markdown/JSON).
            report_json = client.get(f"/v1/eval-runs/{degraded_id}/report").json()
            assert report_json["report_type"] == "gate"
            assert report_json["passed"] is False
            assert report_json["exit_code"] == 1
            report_md = client.get(
                f"/v1/eval-runs/{degraded_id}/report", params={"format": "markdown"}
            )
            assert "FAILED" in report_md.text

    def test_gate_on_unfinished_run_conflicts(self):
        app = create_app()
        record = app.state.registry.create(metadata={}, dataset_version="svc-v1")
        with TestClient(app) as client:
            response = client.post(
                f"/v1/eval-runs/{record.run_id}/gate", json={"baseline_run_id": record.run_id}
            )
            assert response.status_code == 409


# ---------------------------------------------------------------------------
# Webhooks (pluggable sink; signature verified locally)
# ---------------------------------------------------------------------------


class TestWebhooks:
    def test_registration_after_completion_delivers_signed_payload(self):
        sink = RecordingSink()
        secret = "whsec-tests-31337"
        with TestClient(create_app(webhook_sink=sink)) as client:
            run_id = submit_run(client, 1.0, "baseline-v1")
            assert wait_terminal(client, run_id) == "succeeded"

            response = client.post(
                f"/v1/eval-runs/{run_id}/webhook",
                json={"url": "https://hooks.example.test/verdict", "secret": secret},
            )
            assert response.status_code == 202
            assert response.json() == {
                "run_id": run_id,
                "registered": True,
                "delivered": True,
            }

            assert len(sink.deliveries) == 1
            delivery = sink.deliveries[0]
            assert delivery["url"] == "https://hooks.example.test/verdict"
            assert verify_signature(delivery, secret), "HMAC must verify with the secret"

            event = json.loads(delivery["body"])
            assert event["event"] == "eval_run.completed"
            assert event["run_id"] == run_id
            assert event["status"] == "succeeded"
            assert event["error"] is None
            assert event["metrics"]["n_items"] == 12

    def test_worker_delivers_on_completion_for_early_registration(self):
        """Registration that lands BEFORE completion is delivered by the
        worker thread right after the run reaches a terminal status."""
        sink = RecordingSink()
        app = create_app(webhook_sink=sink)
        registry: RunRegistry = app.state.registry
        record = registry.create(metadata={}, dataset_version="svc-v1")
        assert (
            registry.register_webhook(
                record.run_id,
                WebhookRegistration(url="https://hooks.example.test/early", secret="early"),
            )
            is False
        )
        _execute_run(registry, sink, make_dataset(), MockModelAdapter(quality=0.5), record.run_id)
        assert registry.get(record.run_id).status == "succeeded"  # type: ignore[union-attr]
        assert len(sink.deliveries) == 1
        assert verify_signature(sink.deliveries[0], "early")
        assert json.loads(sink.deliveries[0]["body"])["status"] == "succeeded"

    def test_failed_run_payload_carries_error(self):
        sink = RecordingSink()
        app = create_app(webhook_sink=sink)
        registry: RunRegistry = app.state.registry
        record = registry.create(metadata={}, dataset_version="svc-v1")
        registry.register_webhook(
            record.run_id, WebhookRegistration(url="https://hooks.example.test/fail", secret="k")
        )
        _execute_run(registry, sink, make_dataset(), FailingAdapter(), record.run_id)
        finished = registry.get(record.run_id)
        assert finished is not None and finished.status == "failed"
        assert "boom" in (finished.error or "")
        delivery = sink.deliveries[0]
        assert verify_signature(delivery, "k")
        event = json.loads(delivery["body"])
        assert event["status"] == "failed"
        assert "boom" in event["error"]

    def test_failed_run_report_and_gate_conflict(self):
        app = create_app()
        registry: RunRegistry = app.state.registry
        record = registry.create(metadata={}, dataset_version="svc-v1")
        _execute_run(
            registry, app.state.webhook_sink, make_dataset(), FailingAdapter(), record.run_id
        )
        with TestClient(app) as client:
            run = client.get(f"/v1/eval-runs/{record.run_id}").json()
            assert run["status"] == "failed"
            assert "boom" in run["error"]
            assert client.get(f"/v1/eval-runs/{record.run_id}/report").status_code == 409
            gate = client.post(
                f"/v1/eval-runs/{record.run_id}/gate", json={"baseline_run_id": record.run_id}
            )
            assert gate.status_code == 409


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_registered_dataset_404(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/eval-runs",
                json={"dataset": {"name": "nope"}, "candidate": {"adapter": "replay"}},
            )
            assert response.status_code == 404

    def test_registered_dataset_by_name(self):
        dataset = make_dataset(version="golden-v9")
        with TestClient(create_app(registered_datasets={"golden-smoke": dataset})) as client:
            response = client.post(
                "/v1/eval-runs",
                json={
                    "dataset": {"name": "golden-smoke"},
                    "candidate": {"adapter": "replay", "params": {"quality": 1.0}},
                    "metadata": {"model_version": "replay-v1"},
                },
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            assert wait_terminal(client, run_id) == "succeeded"
            metrics = client.get(f"/v1/eval-runs/{run_id}").json()["metrics"]
            assert metrics["dataset_version"] == "golden-v9"

    def test_dataset_needs_exactly_one_source(self):
        with TestClient(create_app()) as client:
            both = client.post(
                "/v1/eval-runs",
                json={
                    "dataset": {"name": "a", "jsonl": "x"},
                    "candidate": {"adapter": "replay"},
                },
            )
            assert both.status_code == 422
            neither = client.post(
                "/v1/eval-runs", json={"dataset": {}, "candidate": {"adapter": "replay"}}
            )
            assert neither.status_code == 422

    def test_invalid_jsonl_422(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/eval-runs",
                json={
                    "dataset": {"jsonl": '{"item_id": "x"}'},
                    "candidate": {"adapter": "replay"},
                },
            )
            assert response.status_code == 422

    def test_unknown_adapter_422(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/eval-runs",
                json={
                    "dataset": {"jsonl": dataset_jsonl()},
                    "candidate": {"adapter": "gpt-4o-live"},
                },
            )
            assert response.status_code == 422

    def test_quality_out_of_range_422(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/v1/eval-runs",
                json={
                    "dataset": {"jsonl": dataset_jsonl()},
                    "candidate": {"adapter": "replay", "params": {"quality": 1.5}},
                },
            )
            assert response.status_code == 422

    def test_bad_report_format_422(self):
        with TestClient(create_app()) as client:
            run_id = submit_run(client, 1.0, "baseline-v1")
            wait_terminal(client, run_id)
            response = client.get(f"/v1/eval-runs/{run_id}/report", params={"format": "yaml"})
            assert response.status_code == 422


# ---------------------------------------------------------------------------
# Auth (X-VerdictAI-Key, hashed storage, constant-time compare)
# ---------------------------------------------------------------------------


class TestAuth:
    def test_auth_enforced_when_env_set(self, monkeypatch):
        monkeypatch.setenv("VERDICTAI_API_KEYS", "alpha-beta, gamma-delta")
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200  # /health is exempt
            assert client.get("/v1/eval-runs/abc").status_code == 401
            wrong = client.get("/v1/eval-runs/abc", headers={"X-VerdictAI-Key": "wrong"})
            assert wrong.status_code == 401
            good = client.get("/v1/eval-runs/abc", headers={"X-VerdictAI-Key": "alpha-beta"})
            assert good.status_code == 404  # past auth; run genuinely unknown
            second = client.get("/v1/eval-runs/abc", headers={"X-VerdictAI-Key": "gamma-delta"})
            assert second.status_code == 404  # every configured key works

    def test_keys_stored_only_as_hashes(self, monkeypatch):
        monkeypatch.setenv("VERDICTAI_API_KEYS", "alpha-beta, gamma-delta, alpha-beta")
        app = create_app()
        expected = sorted(
            {hash_api_key("alpha-beta"), hash_api_key("gamma-delta")}
        )
        assert app.state.api_key_hashes == expected  # deduped + hashed
        for raw in ("alpha-beta", "gamma-delta"):
            assert raw not in app.state.api_key_hashes
            assert all(len(entry) == 64 for entry in app.state.api_key_hashes)

    def test_auth_disabled_without_env_warns(self, monkeypatch, caplog):
        monkeypatch.delenv("VERDICTAI_API_KEYS", raising=False)
        with caplog.at_level(logging.WARNING, logger="verdictai.service"):
            app = create_app()
        assert "auth is DISABLED" in caplog.text
        assert app.state.api_key_hashes == []
        with TestClient(app) as client:
            # Backward compatible: no key required when auth is disabled.
            assert client.get("/v1/eval-runs/abc").status_code == 404
            assert client.post(
                "/v1/eval-runs",
                json={"dataset": {"name": "x"}, "candidate": {"adapter": "replay"}},
            ).status_code == 404

    def test_load_api_key_hashes_parses_and_dedupes(self):
        assert load_api_key_hashes(None) == []
        assert load_api_key_hashes("") == []
        hashes = load_api_key_hashes("  k1 ,  ,k2 ,k1 ")
        assert hashes == sorted({hash_api_key("k1"), hash_api_key("k2")})


# ---------------------------------------------------------------------------
# Correlation-id echo
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def test_echo_on_success_and_404(self):
        with TestClient(create_app()) as client:
            health = client.get("/health", headers={"X-Correlation-ID": "cid-abc-123"})
            assert health.headers["X-Correlation-ID"] == "cid-abc-123"
            missing = client.get("/v1/eval-runs/none", headers={"X-Correlation-ID": "cid-xyz"})
            assert missing.status_code == 404
            assert missing.headers["X-Correlation-ID"] == "cid-xyz"

    def test_echo_on_401_auth_rejection(self, monkeypatch):
        monkeypatch.setenv("VERDICTAI_API_KEYS", "alpha-beta")
        with TestClient(create_app()) as client:
            response = client.get(
                "/v1/eval-runs/abc",
                headers={"X-Correlation-ID": "cid-401", "X-VerdictAI-Key": "nope"},
            )
            assert response.status_code == 401
            assert response.headers["X-Correlation-ID"] == "cid-401"

    def test_generated_when_missing(self):
        with TestClient(create_app()) as client:
            response = client.get("/health")
            cid = response.headers.get("X-Correlation-ID")
            assert cid  # a fresh id is minted for unmarked requests


# ---------------------------------------------------------------------------
# Registry units (TTL + cap) and signing helpers
# ---------------------------------------------------------------------------


class TestRunRegistry:
    def test_cap_evicts_oldest(self):
        registry = RunRegistry(max_runs=2, ttl_seconds=3600.0)
        first = registry.create(metadata={}, dataset_version="v")
        second = registry.create(metadata={}, dataset_version="v")
        third = registry.create(metadata={}, dataset_version="v")
        assert registry.get(first.run_id) is None  # oldest evicted
        assert registry.get(second.run_id) is not None
        assert registry.get(third.run_id) is not None

    def test_ttl_expiry(self):
        registry = RunRegistry(max_runs=10, ttl_seconds=0.0)
        record = registry.create(metadata={}, dataset_version="v")
        time.sleep(0.01)
        assert registry.get(record.run_id) is None


class TestSigning:
    def test_sign_payload_format(self):
        expected = hmac.new(b"k", b"body", hashlib.sha256).hexdigest()
        assert sign_payload(b"body", "k") == f"sha256={expected}"

    def test_payload_is_canonical_and_complete(self):
        body = build_webhook_payload("run-1", "succeeded", None, None)
        event = json.loads(body)
        assert body == json.dumps(event, sort_keys=True).encode()  # canonical bytes
        assert event["event"] == "eval_run.completed"
        assert event["run_id"] == "run-1"
        assert event["metrics"] is None
