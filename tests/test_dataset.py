"""Dataset generator: coverage matrix, versioned artifacts, decontamination."""

from __future__ import annotations

import hashlib
import json

import pytest

from verdictai.datasets.generator import (
    EvalSetGenerator,
    GeneratorConfig,
    TopicSeed,
    build_corpus_ngrams,
    ngram_contamination,
)
from verdictai.datasets.schema import (
    EDGE_CASES,
    DatasetItem,
    EvalDataset,
    load_dataset,
    save_dataset,
)
from verdictai.errors import DatasetValidationError

SMALL_CONFIG = dict(
    topics=[
        TopicSeed(name="math-word-problems", capability="reasoning",
                  keywords=["solve", "total", "steps"], blurb="word problems"),
        TopicSeed(name="document-summarization", capability="summarization",
                  keywords=["summary", "key", "points"], blurb="summaries"),
    ],
    personas=["a busy executive", "a junior engineer"],
    difficulties=["easy", "hard"],
    seed=7,
    version="v1",
)


class TestCoverage:
    def test_full_grid_covered_and_edges_cycled(self):
        dataset = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        # 2 topics x 2 personas x 2 difficulties = 8 cells
        assert len(dataset.items) == 8
        assert len(dataset.coverage) == 8
        assert all(count == 1 for count in dataset.coverage.values())
        # 8 items cycle the 6 edge classes -> all present
        assert {item.edge_case for item in dataset.items} == set(EDGE_CASES)
        capabilities = {item.capability for item in dataset.items}
        assert capabilities == {"reasoning", "summarization"}

    def test_deterministic_for_fixed_seed(self):
        first = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        second = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        assert first.sha256() == second.sha256()

    def test_empty_input_edge_has_empty_prompt(self):
        dataset = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        empty = [i for i in dataset.items if i.edge_case == "empty_input"]
        assert len(empty) >= 1  # 8 items cycle 6 edge classes, so it recurs
        for item in empty:
            assert item.prompt.strip() == ""


class TestVersionedArtifacts:
    def test_save_manifest_changelog(self, tmp_path):
        dataset = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        paths = save_dataset(dataset, tmp_path)
        jsonl_text = paths["dataset"].read_text(encoding="utf-8").rstrip("\n")
        expected_sha = hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest()
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        assert manifest["sha256"] == expected_sha
        assert manifest["n_items"] == 8
        assert manifest["dataset_version"] == "v1"
        changelog = paths["changelog"].read_text(encoding="utf-8")
        assert "## [v1]" in changelog and expected_sha in changelog

    def test_load_roundtrip(self, tmp_path):
        dataset = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        save_dataset(dataset, tmp_path)
        loaded = load_dataset(tmp_path, "v1")
        assert loaded.sha256() == dataset.sha256()
        assert [i.item_id for i in loaded.items] == [i.item_id for i in dataset.items]

    def test_changelog_appends_new_version(self, tmp_path):
        dataset = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG)).generate()
        save_dataset(dataset, tmp_path)
        dataset.version = "v2"
        save_dataset(dataset, tmp_path)
        text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [v1]" in text and "## [v2]" in text


class TestSchemaValidation:
    def _dataset(self, items):
        return EvalDataset(
            version="t", generator_version="t", created_at="t", config_digest="t", items=items
        )

    def test_duplicate_ids_rejected(self):
        item = DatasetItem(
            item_id="dup", prompt="p", reference="r", capability="c", topic="t",
            persona="p", difficulty="easy", edge_case="ambiguity",
        )
        with pytest.raises(DatasetValidationError):
            self._dataset([item, item]).validate_items()

    def test_empty_reference_rejected(self):
        item = DatasetItem(
            item_id="x", prompt="p", reference="  ", capability="c", topic="t",
            persona="p", difficulty="easy", edge_case="ambiguity",
        )
        with pytest.raises(DatasetValidationError):
            self._dataset([item]).validate_items()

    def test_empty_prompt_allowed_only_for_empty_input(self):
        base = dict(
            item_id="x", reference="r", capability="c", topic="t", persona="p",
            difficulty="easy",
        )
        ok = DatasetItem(**base, prompt="   ", edge_case="empty_input")
        self._dataset([ok]).validate_items()  # no raise
        bad = DatasetItem(**base, prompt="   ", edge_case="ambiguity")
        with pytest.raises(DatasetValidationError):
            self._dataset([bad]).validate_items()

    def test_invalid_difficulty_rejected_by_pydantic(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DatasetItem(
                item_id="x", prompt="p", reference="r", capability="c", topic="t",
                persona="p", difficulty="impossible", edge_case="ambiguity",
            )


class TestDecontamination:
    def test_ngram_unit(self):
        corpus = build_corpus_ngrams(["the quick brown fox jumps over the lazy dog"], n=4)
        assert ngram_contamination("please jump! over the lazy dog now", corpus, n=4)
        assert ngram_contamination("totally unrelated sentence about ships", corpus, n=4) == []

    def test_planted_overlap_is_flagged(self):
        generator = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG))
        clean = generator.generate()
        target = clean.items[3]
        # Plant an 8-gram window of the target prompt into a fake corpus.
        words = target.prompt.split()
        planted = " ".join(words[:12])
        contaminated = generator.generate(corpus=planted)
        assert target.item_id in contaminated.contaminated_ids
        flagged_item = next(i for i in contaminated.items if i.item_id == target.item_id)
        assert flagged_item.meta["contaminated_ngrams"]

    def test_drop_mode_removes_items(self):
        generator = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG))
        clean = generator.generate()
        target = clean.items[3]
        config = GeneratorConfig(**{**SMALL_CONFIG, "on_contamination": "drop"})
        dropped = EvalSetGenerator(config).generate(corpus=" ".join(target.prompt.split()[:12]))
        assert target.item_id not in [i.item_id for i in dropped.items]
        assert len(dropped.items) < len(clean.items)


class TestLLMGenerationPath:
    def test_generate_async_uses_llm_output(self, function_client, run_async):
        counter = {"n": 0}

        def respond(prompt):
            counter["n"] += 1
            return json.dumps(
                {"question": f"[llm] question {counter['n']}",
                 "reference": f"[llm] reference {counter['n']}"}
            )

        generator = EvalSetGenerator(
            GeneratorConfig(**SMALL_CONFIG), client=function_client(respond)
        )
        dataset = run_async(generator.generate_async())
        assert len(dataset.items) == 8
        assert all(item.meta["generator"] == "llm" for item in dataset.items)
        assert dataset.items[0].prompt.startswith("[llm]")

    def test_generate_async_requires_client(self, run_async):
        generator = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG))
        with pytest.raises(ValueError):
            run_async(generator.generate_async())

    def test_generate_async_falls_back_on_bad_payload(self, function_client, run_async):
        generator = EvalSetGenerator(
            GeneratorConfig(**SMALL_CONFIG), client=function_client(lambda p: "garbage")
        )
        dataset = run_async(generator.generate_async())
        assert all(item.meta["generator"] == "template" for item in dataset.items)
        assert all(item.meta["llm_fallback"] is True for item in dataset.items)

    def test_sync_generate_never_touches_client(self, function_client, run_async):
        def boom(prompt):
            raise AssertionError("client must not be called in sync template mode")

        generator = EvalSetGenerator(GeneratorConfig(**SMALL_CONFIG), client=function_client(boom))
        dataset = generator.generate()
        assert len(dataset.items) == 8
