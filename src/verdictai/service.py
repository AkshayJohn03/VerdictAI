"""Eval-service HTTP API: run the golden runner and the regression gate over
HTTP so CI systems and internal platforms can consume VerdictAI without the
CLI.

Design notes (the non-obvious "why"s):

- **No live model calls, ever.** The only adapters exposed here are the
  offline ones (``replay`` / ``mock``). An eval service that can secretly hit
  a paid API from a POST body is a cost and supply-chain hazard; callers wrap
  real models in their own adapter and register it before exposing this app.
- **Execution model.** Runs execute on a worker thread (``asyncio.to_thread``)
  and mutate an in-memory :class:`RunRegistry` (TTL + hard cap). This is a
  deliberate trade-off: CI consumers poll within minutes, so durability would
  add a storage backend without changing the API contract. Restarting the
  service loses run history — the registry TTL documents that.
- **Default judge is the offline** ``ReferenceJudge`` (deterministic token-F1
  vs the golden reference): it is strictly monotone under the replay adapter's
  word-dropping degradation, which is exactly the signal the gate's bootstrap
  CI needs. ``HeuristicJudge`` mixes keyword/length/format signals and is less
  sensitive, so it would need bigger planted regressions to gate on.
- **Webhook signing.** The payload body is canonical (sorted-key JSON) and the
  HMAC-SHA256 is computed over the exact bytes sent, so receivers can verify
  with their secret over the raw request body (standard GitHub-style scheme).
  Delivery is pluggable via :class:`WebhookSink` (httpx by default, a
  recording fake in tests) — the suite never touches the network.
- **Auth stores only SHA-256 hashes** of the ``VERDICTAI_API_KEYS`` entries
  and compares hashes in constant time, so a memory dump or log of app state
  never leaks a usable key.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from . import __version__
from .datasets.schema import EvalDataset
from .errors import DatasetValidationError
from .judges.reference import ReferenceJudge
from .regression.gate import GateResult, run_gate
from .regression.runner import (
    GoldenRunner,
    MockModelAdapter,
    ModelAdapter,
    OfflineReplayAdapter,
    ScoreSnapshot,
    snapshot_sha256,
)

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-VerdictAI-Key"
SIGNATURE_HEADER = "X-VerdictAI-Signature"
CORRELATION_HEADER = "X-Correlation-ID"

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED})

# Module-level constant so _build_adapter can reuse the adapter's canned
# default without inventing a second source of truth for it.
_DEFAULT_MOCK_ANSWER = MockModelAdapter().default_answer


# --------------------------------------------------------------------------
# Registry (in-memory run store: TTL + hard cap)
# --------------------------------------------------------------------------


@dataclass
class WebhookRegistration:
    """A URL + signing secret pair registered for one run."""

    url: str
    secret: str


@dataclass
class RunRecord:
    """Mutable state of one eval run (status flips, snapshot lands once)."""

    run_id: str
    status: str
    created_ts: float
    metadata: dict[str, Any]
    dataset_version: str
    snapshot: ScoreSnapshot | None = None
    error: str | None = None
    last_gate: GateResult | None = None
    webhooks: list[WebhookRegistration] = field(default_factory=list)


class RunRegistry:
    """Thread-safe in-memory run store with TTL eviction and a hard cap.

    All mutations happen under one lock; the worker thread and the event-loop
    request handlers share this object. Expired entries are purged lazily on
    access, so a consumer holding an id past the TTL sees a plain 404.
    """

    def __init__(self, *, max_runs: int = 100, ttl_seconds: float = 3600.0) -> None:
        self.max_runs = max_runs
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def create(self, *, metadata: dict[str, Any], dataset_version: str) -> RunRecord:
        """Insert a queued run; evict expired then oldest entries to fit cap."""
        with self._lock:
            self._purge_expired()
            while len(self._runs) >= self.max_runs and self._runs:
                oldest = min(self._runs.values(), key=lambda record: record.created_ts)
                del self._runs[oldest.run_id]
            record = RunRecord(
                run_id=str(uuid.uuid4()),
                status=QUEUED,
                created_ts=time.time(),
                metadata=dict(metadata),
                dataset_version=dataset_version,
            )
            self._runs[record.run_id] = record
            return record

    def get(self, run_id: str) -> RunRecord | None:
        """Look up a live (non-expired) run; ``None`` means unknown/gone."""
        with self._lock:
            self._purge_expired()
            return self._runs.get(run_id)

    def set_running(self, run_id: str) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.status = RUNNING

    def finish(
        self, run_id: str, *, snapshot: ScoreSnapshot | None = None, error: str | None = None
    ) -> None:
        """Move a run to a terminal status. No-op if the record was evicted."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            if error is None:
                record.status = SUCCEEDED
                record.snapshot = snapshot
            else:
                record.status = FAILED
                record.error = error

    def register_webhook(self, run_id: str, registration: WebhookRegistration) -> bool:
        """Attach a webhook to a run.

        Returns ``True`` when the run is already terminal — the caller must
        deliver immediately, because the worker cleared the pending list under
        the same lock. This keeps registration racing completion exactly-once:
        either the worker takes the registration, or the caller delivers it.
        """
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False
            if record.status in TERMINAL_STATUSES:
                return True
            record.webhooks.append(registration)
            return False

    def take_webhooks(self, run_id: str) -> list[WebhookRegistration]:
        """Atomically drain the pending webhook list (worker-side, on finish)."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return []
            pending, record.webhooks = record.webhooks, []
            return pending

    def record_gate(self, run_id: str, gate: GateResult) -> None:
        """Remember the most recent gate decision so /report can serve it."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None:
                record.last_gate = gate

    def _purge_expired(self) -> None:
        # Callers hold the lock; lazily enforcing TTL keeps the hot path free
        # of a background sweeper thread.
        now = time.time()
        expired = [
            run_id
            for run_id, record in self._runs.items()
            if now - record.created_ts > self.ttl_seconds
        ]
        for run_id in expired:
            del self._runs[run_id]


# --------------------------------------------------------------------------
# Webhooks: canonical payload, HMAC signing, pluggable delivery
# --------------------------------------------------------------------------


class WebhookSink(Protocol):
    """Delivery boundary for signed completion webhooks.

    Sync by design: deliveries run on the worker thread (or inline for the
    late-registration path), never on the event loop, so a slow receiver
    cannot stall the API. Tests substitute a recording fake.
    """

    def deliver(self, url: str, body: bytes, headers: dict[str, str]) -> None: ...


class HttpxWebhookSink:
    """Default sink: one synchronous httpx POST per delivery."""

    def __init__(self, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = timeout_seconds

    def deliver(self, url: str, body: bytes, headers: dict[str, str]) -> None:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, content=body, headers=headers)
            response.raise_for_status()


def sign_payload(body: bytes, secret: str) -> str:
    """``sha256=<hex>`` HMAC over the exact body bytes a receiver will read."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_webhook_payload(
    run_id: str, status: str, snapshot: ScoreSnapshot | None = None, error: str | None = None
) -> bytes:
    """Canonical (sorted-key) completion event — stable bytes for the HMAC."""
    payload = {
        "event": "eval_run.completed",
        "run_id": run_id,
        "status": status,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": _metrics_summary(snapshot) if snapshot is not None else None,
        "error": error,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _post_webhook(sink: WebhookSink, registration: WebhookRegistration, body: bytes) -> None:
    headers = {
        "Content-Type": "application/json",
        SIGNATURE_HEADER: sign_payload(body, registration.secret),
    }
    sink.deliver(registration.url, body, headers)


def _deliver_webhooks(
    sink: WebhookSink,
    run_id: str,
    status: str,
    snapshot: ScoreSnapshot | None,
    error: str | None,
    pending: list[WebhookRegistration],
) -> None:
    """Deliver each registered webhook; one bad receiver never fails a run."""
    if not pending:
        return
    body = build_webhook_payload(run_id, status, snapshot, error)
    for registration in pending:
        try:
            _post_webhook(sink, registration, body)
        except Exception:
            logger.warning(
                "webhook delivery to %s failed for run %s", registration.url, run_id, exc_info=True
            )


# --------------------------------------------------------------------------
# Run execution (worker-thread body)
# --------------------------------------------------------------------------


def _execute_run(
    registry: RunRegistry,
    sink: WebhookSink,
    dataset: EvalDataset,
    adapter: ModelAdapter,
    run_id: str,
) -> None:
    """Run the golden runner for one queued run. Runs in a worker thread and
    owns its own event loop (``asyncio.run``) because the adapters and judges
    are async; the thread is parked there by ``asyncio.to_thread``."""
    registry.set_running(run_id)
    try:
        runner = GoldenRunner(adapter, judge=ReferenceJudge(), store=None)
        snapshot = asyncio.run(runner.run(dataset, persist=False))
        registry.finish(run_id, snapshot=snapshot)
        status, error = SUCCEEDED, None
    except Exception as exc:  # noqa: BLE001 - any failure must land as status=failed
        logger.exception("eval run %s failed", run_id)
        registry.finish(run_id, error=str(exc))
        status, error, snapshot = FAILED, str(exc), None
    pending = registry.take_webhooks(run_id)
    record = registry.get(run_id)
    _deliver_webhooks(
        sink,
        run_id,
        status,
        record.snapshot if record else snapshot,
        error,
        pending,
    )


# --------------------------------------------------------------------------
# Metrics / report rendering
# --------------------------------------------------------------------------


def _metrics_summary(snapshot: ScoreSnapshot) -> dict[str, Any]:
    capability_means = {
        cap: round(mean, 4) for cap, mean in snapshot.capability_means().items()
    }
    return {
        "n_items": len(snapshot.records),
        "mean_score": round(snapshot.mean(), 4),
        "capability_means": capability_means,
        "model_version": snapshot.model_version,
        "dataset_version": snapshot.dataset_version,
        "dataset_sha256": snapshot.dataset_sha256,
        "judge_type": snapshot.judge_type,
        "snapshot_sha256": snapshot_sha256(snapshot),
    }


def _render_run_markdown(run_id: str, snapshot: ScoreSnapshot) -> str:
    lines = [
        "## VerdictAI eval run report",
        "",
        f"- Run: `{run_id}`",
        f"- Model: `{snapshot.model_version}` (dataset {snapshot.dataset_version}, "
        f"judge {snapshot.judge_type})",
        f"- Items: {len(snapshot.records)}, mean score {snapshot.mean():.4f}",
        "",
        "| capability | n | mean |",
        "|---|---|---|",
    ]
    counts: dict[str, int] = {}
    for record in snapshot.records:
        counts[record.capability] = counts.get(record.capability, 0) + 1
    for capability, mean in snapshot.capability_means().items():
        lines.append(f"| {capability} | {counts.get(capability, 0)} | {mean:.4f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class DatasetSpec(BaseModel):
    """Pick a dataset by registered name or inline JSONL (exactly one)."""

    name: str | None = None
    jsonl: str | None = None
    version: str | None = None


class AdapterSpec(BaseModel):
    """Offline candidate adapter. ``replay`` replays golden references with
    deterministic degradation; ``mock`` answers from a canned bank."""

    adapter: Literal["replay", "mock"]
    params: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(BaseModel):
    """Free-form run metadata; ``model_version`` / ``triggered_by`` are
    first-class so CI callers can annotate who asked for the run."""

    model_config = ConfigDict(extra="allow")

    model_version: str | None = None
    triggered_by: str | None = None


class EvalRunRequest(BaseModel):
    dataset: DatasetSpec
    candidate: AdapterSpec
    metadata: RunMetadata = Field(default_factory=RunMetadata)


class GateRequest(BaseModel):
    baseline_run_id: str
    threshold: float = Field(default=0.05, description="max tolerated mean-delta regression")
    min_effect: float = Field(default=0.1, description="minimum Cliff's delta to call a regression")


class WebhookRequest(BaseModel):
    url: HttpUrl
    secret: str = Field(min_length=1)


# --------------------------------------------------------------------------
# Resolution helpers
# --------------------------------------------------------------------------


def _resolve_dataset(spec: DatasetSpec, registered: Mapping[str, EvalDataset]) -> EvalDataset:
    if (spec.name is None) == (spec.jsonl is None):
        raise HTTPException(
            status_code=422, detail="provide exactly one of dataset.name or dataset.jsonl"
        )
    if spec.name is not None:
        dataset = registered.get(spec.name)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"unknown registered dataset {spec.name!r}")
        return dataset
    try:
        return EvalDataset.from_jsonl(spec.jsonl or "", version=spec.version or "inline")
    except (DatasetValidationError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid dataset JSONL: {exc}") from exc


def _build_adapter(
    spec: AdapterSpec, dataset: EvalDataset, metadata: RunMetadata
) -> ModelAdapter:
    params = spec.params
    try:
        quality = float(params.get("quality", 1.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="candidate.params.quality must be a number"
        ) from exc
    if not 0.0 <= quality <= 1.0:
        raise HTTPException(status_code=422, detail="candidate.params.quality must be in [0, 1]")
    model_version = str(params.get("model_version") or metadata.model_version or "")
    if spec.adapter == "replay":
        return OfflineReplayAdapter(
            dataset, quality=quality, model_version=model_version or "replay-v1"
        )
    answers_raw = params.get("answers") or {}
    if not isinstance(answers_raw, Mapping):
        raise HTTPException(status_code=422, detail="candidate.params.answers must be an object")
    answers = {str(key): str(value) for key, value in answers_raw.items()}
    default_answer = str(params.get("default_answer") or _DEFAULT_MOCK_ANSWER)
    return MockModelAdapter(
        quality=quality,
        answers=answers,
        default_answer=default_answer,
        model_version=model_version or "mock-bank-v1",
    )


def hash_api_key(raw: str) -> str:
    """SHA-256 hex of a raw API key — the only form ever kept in memory."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_api_key_hashes(env_value: str | None) -> list[str]:
    """Parse ``VERDICTAI_API_KEYS`` (comma-separated) into hashed keys.

    Hashed at load so no raw key survives in app state; comparison later is
    ``hmac.compare_digest`` over equal-length hex digests (constant time).
    """
    if not env_value:
        return []
    return sorted({hash_api_key(key.strip()) for key in env_value.split(",") if key.strip()})


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(
    *,
    registered_datasets: Mapping[str, EvalDataset] | None = None,
    webhook_sink: WebhookSink | None = None,
    api_key_hashes: list[str] | None = None,
    ttl_seconds: float = 3600.0,
    max_runs: int = 100,
) -> FastAPI:
    """Build the eval-service app.

    ``api_key_hashes`` overrides the ``VERDICTAI_API_KEYS`` env lookup (tests
    inject directly); when it is ``None`` and the env var is unset, auth is
    disabled with a startup warning so existing single-tenant deployments
    keep working unchanged.
    """
    app = FastAPI(title="VerdictAI Eval Service", version=__version__)
    app.state.registry = RunRegistry(max_runs=max_runs, ttl_seconds=ttl_seconds)
    app.state.webhook_sink: WebhookSink = webhook_sink or HttpxWebhookSink()
    app.state.registered_datasets: dict[str, EvalDataset] = dict(registered_datasets or {})
    app.state.run_tasks: set[asyncio.Task[None]] = set()

    if api_key_hashes is None:
        api_key_hashes = load_api_key_hashes(os.environ.get("VERDICTAI_API_KEYS"))
        if not api_key_hashes:
            logger.warning(
                "VERDICTAI_API_KEYS is not set; API-key auth is DISABLED "
                "(development / backward-compatible mode)"
            )
    app.state.api_key_hashes = api_key_hashes

    registry = app.state.registry
    sink = app.state.webhook_sink
    datasets = app.state.registered_datasets

    # Auth middleware first, correlation second: the last-added middleware is
    # outermost, so even 401 responses carry the caller's correlation id.
    @app.middleware("http")
    async def require_api_key(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        hashes = app.state.api_key_hashes
        if not hashes or request.url.path == "/health":
            return await call_next(request)
        provided = request.headers.get(API_KEY_HEADER, "")
        provided_hash = hash_api_key(provided)
        if any(hmac.compare_digest(provided_hash, stored) for stored in hashes):
            return await call_next(request)
        return JSONResponse(status_code=401, content={"detail": "missing or invalid API key"})

    @app.middleware("http")
    async def correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cid = request.headers.get(CORRELATION_HEADER) or uuid.uuid4().hex
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = cid
        return response

    def _get_record(run_id: str) -> RunRecord:
        record = registry.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown run id {run_id!r}")
        return record

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/eval-runs", status_code=202)
    async def submit_eval_run(request_body: EvalRunRequest) -> dict[str, Any]:
        dataset = _resolve_dataset(request_body.dataset, datasets)
        # Build the adapter up front so bad configs 422 instead of failing
        # asynchronously in the worker where nobody sees the reason.
        adapter = _build_adapter(request_body.candidate, dataset, request_body.metadata)
        record = registry.create(
            metadata=request_body.metadata.model_dump(),
            dataset_version=dataset.version,
        )
        task = asyncio.create_task(
            asyncio.to_thread(
                _execute_run, registry, sink, dataset, adapter, record.run_id
            )
        )
        # Keep a strong ref: the event loop only holds weak refs to tasks and
        # would garbage-collect a running-but-unreferenced one mid-flight.
        app.state.run_tasks.add(task)
        task.add_done_callback(app.state.run_tasks.discard)
        return {
            "run_id": record.run_id,
            "status": record.status,
            "created_at": _iso(record.created_ts),
            "metadata": record.metadata,
        }

    @app.get("/v1/eval-runs/{run_id}")
    async def get_eval_run(run_id: str) -> dict[str, Any]:
        record = _get_record(run_id)
        return {
            "run_id": record.run_id,
            "status": record.status,
            "created_at": _iso(record.created_ts),
            "metadata": record.metadata,
            "error": record.error,
            "metrics": _metrics_summary(record.snapshot) if record.snapshot else None,
        }

    @app.get("/v1/eval-runs/{run_id}/report")
    async def get_eval_run_report(run_id: str, format: str = "json") -> Response:
        record = _get_record(run_id)
        if record.status not in TERMINAL_STATUSES:
            raise HTTPException(
                status_code=409, detail=f"run {run_id} is not finished (status={record.status})"
            )
        if record.status == FAILED:
            raise HTTPException(status_code=409, detail=f"run {run_id} failed: {record.error}")
        snapshot = record.snapshot
        if snapshot is None:  # unreachable for SUCCEEDED runs; keeps types honest
            raise HTTPException(status_code=409, detail=f"run {run_id} produced no snapshot")
        fmt = format.lower()
        if fmt == "markdown":
            text = (
                record.last_gate.report_markdown
                if record.last_gate is not None
                else _render_run_markdown(record.run_id, snapshot)
            )
            return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
        if fmt == "json":
            if record.last_gate is not None:
                gate = record.last_gate
                return JSONResponse(
                    {
                        "report_type": "gate",
                        "passed": gate.passed,
                        "exit_code": gate.exit_code,
                        "verdict": gate.verdict,
                        "evidence": gate.alert,
                    }
                )
            return JSONResponse(
                {
                    "report_type": "run",
                    "run_id": record.run_id,
                    "metrics": _metrics_summary(snapshot),
                    "snapshot": snapshot.model_dump(),
                }
            )
        raise HTTPException(status_code=422, detail="format must be 'json' or 'markdown'")

    @app.post("/v1/eval-runs/{run_id}/gate")
    async def gate_eval_run(run_id: str, gate_request: GateRequest) -> dict[str, Any]:
        candidate = _get_record(run_id)
        baseline = _get_record(gate_request.baseline_run_id)
        for record in (baseline, candidate):
            if record.status != SUCCEEDED:
                raise HTTPException(
                    status_code=409,
                    detail=f"run {record.run_id} is not succeeded (status={record.status})",
                )
        baseline_snapshot = baseline.snapshot
        candidate_snapshot = candidate.snapshot
        if baseline_snapshot is None or candidate_snapshot is None:
            raise HTTPException(status_code=409, detail="run finished without a snapshot")
        gate = run_gate(
            baseline_snapshot,
            candidate_snapshot,
            threshold=gate_request.threshold,
            min_effect=gate_request.min_effect,
            context={
                "baseline_run_id": gate_request.baseline_run_id,
                "candidate_run_id": run_id,
            },
        )
        registry.record_gate(run_id, gate)
        return {
            "passed": gate.passed,
            "exit_code": gate.exit_code,
            "verdict": gate.verdict,
            "evidence": gate.alert,
            "report_markdown": gate.report_markdown,
        }

    @app.post("/v1/eval-runs/{run_id}/webhook", status_code=202)
    async def register_run_webhook(run_id: str, webhook_request: WebhookRequest) -> dict[str, Any]:
        record = _get_record(run_id)
        registration = WebhookRegistration(
            url=str(webhook_request.url), secret=webhook_request.secret
        )
        immediate = registry.register_webhook(run_id, registration)
        delivered = False
        delivery_error: str | None = None
        if immediate:
            body = build_webhook_payload(
                record.run_id, record.status, record.snapshot, record.error
            )
            try:
                _post_webhook(sink, registration, body)
                delivered = True
            except Exception as exc:  # a dead receiver must not fail registration
                delivery_error = str(exc)
        payload: dict[str, Any] = {
            "run_id": record.run_id,
            "registered": True,
            "delivered": delivered,
        }
        if delivery_error is not None:
            payload["delivery_error"] = delivery_error
        return payload

    return app
