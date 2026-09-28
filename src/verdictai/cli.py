"""VerdictAI command line interface.

Subcommands (all run fully offline by default):

- ``verdict judge``      — score one prompt/response with a judge
- ``verdict pair``       — double-order pairwise comparison of two responses
- ``verdict calibrate``  — fit isotonic calibration and emit a markdown report
- ``verdict gen-dataset``— generate a versioned golden dataset + manifest
- ``verdict regress``    — compare two model-version snapshots; ``--gate``
                           makes it a CI gate (exit 1 on regression)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .calibration.active import ActiveLabelLoop
from .calibration.bias import summarize_bias
from .calibration.isotonic import CalibrationLayer, render_calibration_report
from .calibration.labels import HumanLabelSet
from .config import Settings
from .datasets.generator import DEFAULT_TOPICS, EvalSetGenerator, GeneratorConfig, TopicSeed
from .datasets.schema import save_dataset
from .judges.pairwise import PairwiseJudge
from .judges.rubric import Criterion, HeuristicJudge, RubricJudge
from .llm import EchoMockClient, LLMClient, OpenAICompatClient
from .regression.drift import DriftDetector
from .regression.gate import run_gate
from .regression.runner import SnapshotStore
from .types import JudgeItem, PairwiseItem


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="verdict",
        description="VerdictAI: LLM-as-judge with human calibration, eval datasets, "
        "and regression detection.",
    )
    parser.add_argument("--version", action="version", version=f"verdict {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_judge = sub.add_parser("judge", help="score one prompt/response pair")
    p_judge.add_argument("--prompt", required=True)
    p_judge.add_argument("--response", required=True)
    p_judge.add_argument("--reference", default="", help="optional golden answer")
    p_judge.add_argument("--item-id", default="cli-item")
    p_judge.add_argument("--judge", choices=["heuristic", "rubric"], default="heuristic")
    p_judge.add_argument("--rubric-file", default=None, help="YAML file with criteria")
    p_judge.set_defaults(func=cmd_judge)

    p_pair = sub.add_parser("pair", help="double-order pairwise comparison")
    p_pair.add_argument("--prompt", required=True)
    p_pair.add_argument("--response-a", required=True)
    p_pair.add_argument("--response-b", required=True)
    p_pair.add_argument("--item-id", default="cli-pair")
    p_pair.add_argument("--offline", action="store_true", help="force the echo mock client")
    p_pair.set_defaults(func=cmd_pair)

    p_cal = sub.add_parser("calibrate", help="isotonic calibration + report")
    p_cal.add_argument("--labels", required=True, help="human labels JSONL")
    p_cal.add_argument("--out", default=None, help="write markdown report here (default stdout)")
    p_cal.set_defaults(func=cmd_calibrate)

    p_gen = sub.add_parser("gen-dataset", help="generate a versioned golden dataset")
    p_gen.add_argument("--out", required=True, help="output directory")
    p_gen.add_argument("--version", default="v1")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument(
        "--topics",
        default=None,
        help="comma list 'name:capability' (default: built-in topics)",
    )
    p_gen.add_argument("--personas", default=None, help="comma list of personas")
    p_gen.add_argument("--difficulties", default=None, help="comma list of difficulties")
    p_gen.add_argument("--items-per-cell", type=int, default=1)
    p_gen.add_argument("--corpus", default=None, help="text file for n-gram decontamination")
    p_gen.add_argument("--contamination-n", type=int, default=8)
    p_gen.add_argument("--on-contamination", choices=["flag", "drop"], default="flag")
    p_gen.set_defaults(func=cmd_gen_dataset)

    p_reg = sub.add_parser("regress", help="compare two model-version snapshots")
    p_reg.add_argument("--snapshots", required=True, help="append-only snapshot JSONL")
    p_reg.add_argument("--baseline", required=True, help="baseline model_version")
    p_reg.add_argument("--candidate", required=True, help="candidate model_version")
    p_reg.add_argument("--threshold", type=float, default=0.05)
    p_reg.add_argument("--min-effect", type=float, default=0.1)
    p_reg.add_argument("--gate", action="store_true", help="CI gate mode: exit 1 on regression")
    p_reg.set_defaults(func=cmd_regress)
    return parser


def _client_from_settings(settings: Settings, offline: bool) -> LLMClient:
    if offline or not settings.has_api_key:
        return EchoMockClient()
    return OpenAICompatClient(settings)


def _criteria_from_yaml(path: str) -> list[Criterion]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("criteria", []) if isinstance(raw, dict) else raw
    criteria: list[Criterion] = []
    for entry in entries or []:
        criteria.append(
            Criterion(
                name=str(entry["name"]),
                description=str(entry.get("description", "")),
                weight=float(entry.get("weight", 1.0)),
                anchors={int(k): str(v) for k, v in (entry.get("anchors") or {}).items()},
            )
        )
    return criteria or None  # type: ignore[return-value]


def cmd_judge(args: argparse.Namespace) -> int:
    settings = Settings()
    item = JudgeItem(
        item_id=args.item_id, prompt=args.prompt, response=args.response, reference=args.reference
    )
    if args.judge == "rubric":
        if not settings.has_api_key:
            print(
                "error: --judge rubric requires VERDICTAI_API_KEY "
                "(use --judge heuristic offline)",
                file=sys.stderr,
            )
            return 2
        client = OpenAICompatClient(settings)
        criteria = _criteria_from_yaml(args.rubric_file) if args.rubric_file else None
        judge = RubricJudge(client, criteria=criteria)
    else:
        judge = HeuristicJudge()
    result = asyncio.run(judge.judge(item))
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return 0


def cmd_pair(args: argparse.Namespace) -> int:
    settings = Settings()
    client = _client_from_settings(settings, offline=args.offline)
    judge = PairwiseJudge(client)
    item = PairwiseItem(
        item_id=args.item_id, prompt=args.prompt, response_a=args.response_a,
        response_b=args.response_b,
    )
    result = asyncio.run(judge.judge_pair(item))
    out: dict[str, Any] = result.model_dump()
    out["client"] = type(client).__name__
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    label_set = HumanLabelSet.load(args.labels)
    paired = label_set.paired()
    if len(paired) < 2:
        print(
            "error: need at least 2 labels with both judge_score and human_score",
            file=sys.stderr,
        )
        return 2
    judge_scores = [float(lab.judge_score) for lab in paired]  # type: ignore[misc]
    human_scores = [float(lab.human_score) for lab in paired]  # type: ignore[misc]
    layer = CalibrationLayer.fit(judge_scores, human_scores)
    bias = summarize_bias(label_set)
    report = render_calibration_report(label_set, layer, bias)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote calibration report to {args.out}")
    else:
        print(report)
    loop = ActiveLabelLoop(label_set)
    top = loop.next_round(k=5)
    print(f"\nNext labeling round (top {min(5, len(top))} priorities):")
    for label, priority in top:
        if label.human_score is None:
            state = "unlabeled"
        else:
            gap = label.human_score - (label.judge_score or 0)
            state = f"gap={gap:+.2f}"
        print(f"  {label.item_id}: priority={priority:.3f} ({state})")
    return 0


def cmd_gen_dataset(args: argparse.Namespace) -> int:
    if args.topics:
        topics = []
        for chunk in args.topics.split(","):
            name, _, capability = chunk.strip().partition(":")
            known = next((t for t in DEFAULT_TOPICS if t.name == name), None)
            if known is not None:
                topics.append(known)
            else:
                topics.append(
                    TopicSeed(
                        name=name,
                        capability=capability or "general",
                        keywords=[w for w in name.split("-") if len(w) > 3],
                        blurb=name.replace("-", " "),
                    )
                )
    else:
        topics = list(DEFAULT_TOPICS)
    personas = args.personas.split(",") if args.personas else None
    difficulties = args.difficulties.split(",") if args.difficulties else None
    config = GeneratorConfig(
        topics=topics,
        personas=personas or GeneratorConfig().personas,
        difficulties=difficulties or list(GeneratorConfig().difficulties),
        items_per_cell=args.items_per_cell,
        seed=args.seed,
        contamination_n=args.contamination_n,
        on_contamination=args.on_contamination,
        version=args.version,
    )
    corpus_text = None
    if args.corpus:
        corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    generator = EvalSetGenerator(config)
    dataset = generator.generate(corpus=corpus_text)
    paths = save_dataset(dataset, Path(args.out))
    print(f"dataset {dataset.version}: {len(dataset.items)} items")
    print(f"coverage cells filled: {len(dataset.coverage)}")
    print(f"capability counts: {json.dumps(dataset.capability_counts())}")
    print(f"contaminated items flagged: {len(dataset.contaminated_ids)}")
    for label, path in sorted(paths.items()):
        print(f"{label}: {path}")
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshots)
    baseline = store.latest(args.baseline)
    candidate = store.latest(args.candidate)
    if baseline is None or candidate is None:
        print(
            f"error: snapshots not found for baseline={args.baseline!r} "
            f"candidate={args.candidate!r}",
            file=sys.stderr,
        )
        return 2
    if args.gate:
        gate = run_gate(
            baseline, candidate, threshold=args.threshold, min_effect=args.min_effect
        )
        print(gate.report_markdown)
        print(json.dumps(gate.alert, indent=2))
        return gate.exit_code
    report = DriftDetector().check(
        baseline, candidate, threshold=args.threshold, min_effect=args.min_effect
    )
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse --version / usage errors
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
