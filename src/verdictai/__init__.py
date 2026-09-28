"""VerdictAI: LLM-as-judge with human calibration, eval dataset generation,
and regression detection."""

from .config import Settings
from .errors import (
    DatasetValidationError,
    JudgeParseError,
    LLMError,
    LLMResponseError,
    SnapshotError,
    VerdictAIError,
)
from .llm import EchoMockClient, LLMClient, OpenAICompatClient, complete_json
from .types import (
    Criterion,
    CriterionScore,
    JudgeItem,
    JudgeResult,
    PairwiseItem,
    PairwiseResult,
    PairwiseVerdict,
    Winner,
)

__version__ = "0.1.0"

__all__ = [
    "Criterion",
    "CriterionScore",
    "DatasetValidationError",
    "EchoMockClient",
    "JudgeItem",
    "JudgeParseError",
    "JudgeResult",
    "LLMClient",
    "LLMError",
    "LLMResponseError",
    "OpenAICompatClient",
    "PairwiseItem",
    "PairwiseResult",
    "PairwiseVerdict",
    "Settings",
    "SnapshotError",
    "VerdictAIError",
    "Winner",
    "__version__",
    "complete_json",
]
