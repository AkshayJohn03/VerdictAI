"""Versioned eval dataset container with validation and sha256 manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..errors import DatasetValidationError

Difficulty = Literal["easy", "medium", "hard", "adversarial"]

EDGE_CASES: tuple[str, ...] = (
    "empty_input",
    "unicode_stress",
    "prompt_injection",
    "refusal_bait",
    "ambiguity",
    "format_trap",
)
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "adversarial")


class DatasetItem(BaseModel):
    """One golden eval item: prompt + reference answer + slicing metadata."""

    item_id: str
    prompt: str
    reference: str
    capability: str
    topic: str
    persona: str
    difficulty: Difficulty
    edge_case: str
    meta: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A versioned, immutable-by-convention eval dataset."""

    version: str
    generator_version: str
    created_at: str
    config_digest: str
    items: list[DatasetItem] = Field(default_factory=list)
    coverage: dict[str, int] = Field(default_factory=dict)
    contaminated_ids: list[str] = Field(default_factory=list)
    changelog_note: str = ""

    def validate_items(self) -> None:
        """Structural validation; raises :class:`DatasetValidationError`."""
        errors: list[str] = []
        seen: set[str] = set()
        for item in self.items:
            where = item.item_id or "<missing id>"
            if not item.item_id:
                errors.append(f"{where}: item_id is empty")
            if item.item_id in seen:
                errors.append(f"{where}: duplicate item_id")
            seen.add(item.item_id)
            if not item.reference.strip():
                errors.append(f"{where}: reference answer is empty")
            if not item.prompt.strip() and item.edge_case != "empty_input":
                errors.append(f"{where}: empty prompt is only allowed for edge_case=empty_input")
        if errors:
            raise DatasetValidationError("; ".join(errors))

    def to_jsonl(self) -> str:
        """Canonical JSONL text (deterministic; the manifest hashes this)."""
        return "\n".join(item.model_dump_json() for item in self.items)

    @classmethod
    def from_jsonl(cls, text: str, *, version: str = "unknown") -> EvalDataset:
        items = [
            DatasetItem.model_validate_json(line) for line in text.splitlines() if line.strip()
        ]
        dataset = cls(
            version=version,
            generator_version="unknown",
            created_at="unknown",
            config_digest="unknown",
            items=items,
        )
        dataset.validate_items()
        return dataset

    def sha256(self) -> str:
        return hashlib.sha256(self.to_jsonl().encode("utf-8")).hexdigest()

    def capability_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.capability] = counts.get(item.capability, 0) + 1
        return dict(sorted(counts.items()))


def save_dataset(dataset: EvalDataset, out_dir: str | Path) -> dict[str, Path]:
    """Write dataset JSONL + sha256 manifest + append a changelog entry."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_text = dataset.to_jsonl()
    jsonl_path = out / f"dataset-{dataset.version}.jsonl"
    jsonl_path.write_text(jsonl_text + ("\n" if jsonl_text else ""), encoding="utf-8")
    sha = hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest()

    manifest = {
        "dataset_version": dataset.version,
        "generator_version": dataset.generator_version,
        "created_at": dataset.created_at,
        "sha256": sha,
        "n_items": len(dataset.items),
        "config_digest": dataset.config_digest,
        "capability_counts": dataset.capability_counts(),
        "coverage": dataset.coverage,
        "contaminated_ids": dataset.contaminated_ids,
    }
    manifest_path = out / f"manifest-{dataset.version}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    changelog_path = out / "CHANGELOG.md"
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = (
        f"## [{dataset.version}] - {date}\n\n"
        f"- Items: {len(dataset.items)} by capability {dataset.capability_counts()}\n"
        f"- Coverage cells filled: {len(dataset.coverage)}\n"
        f"- Contaminated items flagged: {len(dataset.contaminated_ids)}\n"
        f"- SHA256: `{sha}`\n"
    )
    if dataset.changelog_note:
        entry += f"- Note: {dataset.changelog_note}\n"
    entry += "\n"
    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
        if not existing.startswith("# Changelog"):
            existing = "# Changelog\n\n" + existing
        changelog_path.write_text(existing + entry, encoding="utf-8")
    else:
        changelog_path.write_text("# Changelog\n\n" + entry, encoding="utf-8")

    return {"dataset": jsonl_path, "manifest": manifest_path, "changelog": changelog_path}


def load_dataset(out_dir: str | Path, version: str) -> EvalDataset:
    """Load a dataset + its manifest back from disk."""
    out = Path(out_dir)
    manifest = json.loads((out / f"manifest-{version}.json").read_text(encoding="utf-8"))
    text = (out / f"dataset-{version}.jsonl").read_text(encoding="utf-8")
    dataset = EvalDataset(
        version=manifest["dataset_version"],
        generator_version=manifest["generator_version"],
        created_at=manifest["created_at"],
        config_digest=manifest["config_digest"],
        items=[DatasetItem.model_validate_json(line) for line in text.splitlines() if line.strip()],
        coverage=manifest.get("coverage", {}),
        contaminated_ids=manifest.get("contaminated_ids", []),
    )
    dataset.validate_items()
    return dataset
