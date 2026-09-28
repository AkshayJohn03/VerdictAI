"""Exception hierarchy for VerdictAI."""


class VerdictAIError(Exception):
    """Base class for all VerdictAI errors."""


class LLMError(VerdictAIError):
    """Raised when an LLM HTTP call fails (after retries) or is misconfigured."""


class LLMResponseError(LLMError):
    """Raised when a model response cannot be parsed as a JSON object."""


class JudgeParseError(VerdictAIError):
    """Raised when a judge cannot interpret the LLM verdict payload."""


class DatasetValidationError(VerdictAIError):
    """Raised when an eval dataset container fails schema validation."""


class SnapshotError(VerdictAIError):
    """Raised for corrupt or unreadable score snapshot stores."""
