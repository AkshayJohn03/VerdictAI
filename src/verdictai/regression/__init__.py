"""Regression detection: golden runner, drift detection, CI gate."""

from .drift import (
    DriftDetector,
    DriftReport,
    cliffs_delta,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from .gate import GateResult, run_gate
from .runner import (
    GoldenRunner,
    MockModelAdapter,
    ModelAdapter,
    OfflineReplayAdapter,
    ScoreRecord,
    ScoreSnapshot,
    SnapshotStore,
    degrade,
)

__all__ = [
    "DriftDetector",
    "DriftReport",
    "GateResult",
    "GoldenRunner",
    "MockModelAdapter",
    "ModelAdapter",
    "OfflineReplayAdapter",
    "ScoreRecord",
    "ScoreSnapshot",
    "SnapshotStore",
    "cliffs_delta",
    "degrade",
    "paired_bootstrap_ci",
    "run_gate",
    "wilcoxon_signed_rank",
]
