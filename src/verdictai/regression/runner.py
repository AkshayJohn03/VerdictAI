"""Golden runner: score a candidate model against the golden dataset and
persist append-only score snapshots.

Design: snapshots are append-only JSONL keyed by ``model_version`` — every run
adds a line, nothing is ever mutated. Regression detection then pairs scores
by ``item_id`` across two snapshots, which makes the comparison paired (much
lower variance than independent samples) and fully reproducible.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..datasets.schema import EvalDataset
from ..errors import SnapshotError
from ..judges.rubric import HeuristicJudge
from ..types import JudgeItem, JudgeResult


class ModelAdapter(Protocol):
    """Protocol for systems-under-test. Wrap any real model/API here."""

    model_version: str

    async def generate(self, prompt: str, *, item_id: str = "") -> str:
        """Return the model's answer for one dataset prompt."""
        ...


def degrade(text: str, quality: float) -> str:
    """Deterministically degrade a text to a [0, 1] quality level.

    Keeps each word with probability ``quality`` using a hash of the word
    index + text prefix, so ``degrade(x, q)`` is fully deterministic and
    quality-1.0 returns the text unchanged. Used by the offline replay
    adapter to simulate a model regression without a network.
    """
    q = min(1.0, max(0.0, quality))
    if q >= 1.0:
        return text
    words = text.split(" ")
    kept: list[str] = []
    for i, word in enumerate(words):
        digest = hashlib.sha256(f"{text[:32]}|{i}".encode()).digest()
        draw = int.from_bytes(digest[:4], "big") / 2**32
        if draw < q:
            kept.append(word)
    if not kept and words:
        kept = [words[0]]
    return " ".join(kept)


class MockModelAdapter:
    """Programmable-quality offline adapter backed by a canned answer bank."""

    def __init__(
        self,
        quality: float = 1.0,
        answers: dict[str, str] | None = None,
        default_answer: str = (
            "A good answer covers the key points, states assumptions, and follows "
            "the requested format."
        ),
        model_version: str = "mock-bank-v1",
    ):
        self.quality = quality
        self.answers = answers or {}
        self.default_answer = default_answer
        self.model_version = model_version

    async def generate(self, prompt: str, *, item_id: str = "") -> str:
        base = self.answers.get(item_id) or self.answers.get(prompt) or self.default_answer
        return degrade(base, self.quality)


class OfflineReplayAdapter:
    """Replays golden references with programmable degradation.

    Not a real model — an honest offline simulation of one. Given a dataset,
    it looks up each item's reference answer and returns a deterministically
    degraded copy at ``quality`` in [0, 1]. Lower quality produces responses
    with progressively less reference overlap, so heuristic/LLM judges score
    it lower. Used for demos, tests and CI dry-runs without any network.
    """

    def __init__(
        self, dataset: EvalDataset, quality: float = 1.0, model_version: str = "replay-v1"
    ):
        self.quality = quality
        self.model_version = model_version
        self._by_id = {item.item_id: item.reference for item in dataset.items}

    async def generate(self, prompt: str, *, item_id: str = "") -> str:
        reference = self._by_id.get(item_id)
        if reference is None:
            return degrade(prompt, self.quality)
        return degrade(reference, self.quality)


class ScoreRecord(BaseModel):
    """One item's score inside a snapshot."""

    item_id: str
    capability: str
    score: float
    judge_type: str
    meta: dict[str, Any] = Field(default_factory=dict)


class ScoreSnapshot(BaseModel):
    """A complete scoring run of one model version over one dataset version."""

    model_version: str
    dataset_version: str
    dataset_sha256: str = ""
    created_at: str
    judge_type: str
    records: list[ScoreRecord] = Field(default_factory=list)

    def scores_by_item(self) -> dict[str, float]:
        return {r.item_id: r.score for r in self.records}

    def capability_of(self) -> dict[str, str]:
        return {r.item_id: r.capability for r in self.records}

    def mean(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score for r in self.records) / len(self.records)

    def capability_means(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for record in self.records:
            buckets.setdefault(record.capability, []).append(record.score)
        return {cap: sum(v) / len(v) for cap, v in sorted(buckets.items())}


class SnapshotStore:
    """Append-only JSONL store of score snapshots."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, snapshot: ScoreSnapshot) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(snapshot.model_dump_json() + "\n")

    def load_all(self) -> list[ScoreSnapshot]:
        if not self.path.exists():
            return []
        snapshots: list[ScoreSnapshot] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                snapshots.append(ScoreSnapshot.model_validate_json(stripped))
            except Exception as exc:  # pydantic ValidationError
                raise SnapshotError(f"corrupt snapshot at line {line_no}: {exc}") from exc
        return snapshots

    def latest(self, model_version: str) -> ScoreSnapshot | None:
        """Most recently appended snapshot for a model version (file order)."""
        found: ScoreSnapshot | None = None
        for snapshot in self.load_all():
            if snapshot.model_version == model_version:
                found = snapshot
        return found


class GoldenRunner:
    """Runs a dataset through a ModelAdapter, judges every response, and
    persists a score snapshot."""

    def __init__(
        self, adapter: ModelAdapter, judge: Any | None = None, store: SnapshotStore | None = None
    ):
        self.adapter = adapter
        self.judge = judge if judge is not None else HeuristicJudge()
        self.store = store

    async def run(self, dataset: EvalDataset, *, persist: bool = True) -> ScoreSnapshot:
        records: list[ScoreRecord] = []
        for item in dataset.items:
            response = await self.adapter.generate(item.prompt, item_id=item.item_id)
            judge_item = JudgeItem(
                item_id=item.item_id,
                prompt=item.prompt,
                response=response,
                reference=item.reference,
                meta={"capability": item.capability, "difficulty": item.difficulty},
            )
            result: JudgeResult = await self.judge.judge(judge_item)
            records.append(
                ScoreRecord(
                    item_id=item.item_id,
                    capability=item.capability,
                    score=float(result.overall),
                    judge_type=str(result.judge_type),
                    meta={"edge_case": item.edge_case},
                )
            )
        snapshot = ScoreSnapshot(
            model_version=self.adapter.model_version,
            dataset_version=dataset.version,
            dataset_sha256=dataset.sha256(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            judge_type=str(getattr(self.judge, "judge_type", "custom")),
            records=records,
        )
        if persist and self.store is not None:
            self.store.append(snapshot)
        return snapshot


def snapshot_sha256(snapshot: ScoreSnapshot) -> str:
    """Content hash of a snapshot (stable across runs of the same scores)."""
    payload = json.dumps(
        [[r.item_id, r.capability, r.score] for r in snapshot.records], sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
