"""Deterministic-first assertions: fast, zero-token checks before any LLM spend.

Most eval "failures" are not matters of taste. A payload that violates its
schema, an answer that came back empty, JSON with a broken fence, a latency
budget blown by 10x — these need no model to adjudicate. :class:`DeterministicSuite`
runs an ordered battery of such checks over a batch of outputs and returns a
:class:`SuiteReport`. Only failures that genuinely benefit from semantic
judgment (schema mismatch the model might talk its way out of, similarity
below threshold, unparseable-but-maybe-fenced JSON) are placed on the
**escalation list** for an optional LLM judge; everything else is a hard,
objective verdict.

The suite runs fully offline: text similarity uses :class:`HashEmbeddingClient`,
a deterministic hashed bag-of-words embedder (no network, no model weights).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import LLMResponseError
from .llm import extract_json_object
from .text import tokenize

#: Default cosine threshold for the reference-similarity check.
DEFAULT_MIN_SIMILARITY = 0.5

#: Fallback character bounds when the schema does not specify lengths.
DEFAULT_MAX_CHARS = 100_000

#: Checks that fail deterministically but still merit an LLM judge's opinion.
ESCALATABLE_CHECKS = frozenset({"schema_conformity", "json_valid", "similarity"})


class TimedOutput(BaseModel):
    """An output wrapped with its measured latency (ms) for budget checks."""

    value: Any
    latency_ms: float


class CheckResult(BaseModel):
    """Outcome of one deterministic check."""

    name: str
    passed: bool
    severity: str = "error"  # "error" | "warning"
    detail: str = ""
    #: True when a failure should be queued for LLM-judge review.
    escalate: bool = False


class SuiteReport(BaseModel):
    """All check results for one run of the suite."""

    results: list[CheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def escalations(self) -> list[CheckResult]:
        """Failed checks an LLM judge could meaningfully re-adjudicate."""
        return [r for r in self.results if not r.passed and r.escalate]

    def summary(self) -> str:
        total = len(self.results)
        n_passed = sum(1 for r in self.results if r.passed)
        lines = [f"deterministic suite: {n_passed}/{total} checks passed"]
        for result in self.results:
            if result.passed:
                continue
            tag = "ESCALATE" if result.escalate else "FAIL"
            lines.append(f"{tag} {result.name} [{result.severity}] - {result.detail}")
        if self.passed:
            lines.append("verdict: PASS")
        elif self.escalations and not [
            r for r in self.failures if not r.escalate
        ]:
            lines.append(
                "verdict: ESCALATE (all failures are judge-reviewable; "
                "run an LLM judge on the escalation list)"
            )
        else:
            lines.append("verdict: FAIL")
        return "\n".join(lines)


class HashEmbeddingClient:
    """Deterministic offline text embedder.

    Projects unigrams + adjacent bigrams into a fixed-size vector via SHA-1
    bucketing, term-count weighting, and L2 normalization. No network, no
    weights, bit-for-bit reproducible — good enough for "is this answer still
    about the same thing" thresholds, not a semantic oracle.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.sha1(feature.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        counts: dict[int, float] = {}
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            bucket = self._bucket(feature)
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm == 0:
            return [0.0] * self.dim
        return [counts.get(i, 0.0) / norm for i in range(self.dim)]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of the two embeddings in [0, 1]-ish range."""
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return round(
            sum(x * y for x, y in zip(vec_a, vec_b, strict=True)), 6
        )


def _is_bare_schema(candidate: dict[str, Any]) -> bool:
    """True when a dict *is* a JSON schema rather than a name->schema mapping."""
    return any(key in candidate for key in ("type", "properties", "$schema", "items"))


def _schema_for(
    schema: dict[str, Any] | type[BaseModel] | None, name: str
) -> dict[str, Any] | type[BaseModel] | None:
    """Resolve the schema that applies to output ``name``."""
    if schema is None:
        return None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    if isinstance(schema, dict):
        if _is_bare_schema(schema):
            return schema
        return schema.get(name)
    return None


def _required_fields(schema: dict[str, Any] | type[BaseModel]) -> list[str]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return [
            field_name
            for field_name, info in schema.model_fields.items()
            if info.is_required()
        ]
    if isinstance(schema, dict):
        required = schema.get("required", [])
        return [str(field_name) for field_name in required] if isinstance(required, list) else []
    return []


_JSON_TYPES: dict[str, tuple[type, ...] | None] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": None,  # handled specially below
}


def _json_type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected in ("number", "integer") and isinstance(instance, bool):
        return False  # bool is an int subclass but not a JSON number
    types = _JSON_TYPES.get(expected)
    if types is None:
        return True  # unknown type keyword: do not fail on it
    if expected == "integer" and isinstance(instance, float):
        return instance.is_integer()
    return isinstance(instance, types)


def validate_json_schema(
    instance: Any, schema: dict[str, Any], path: str = "$"
) -> list[str]:
    """Validate ``instance`` against the supported JSON-schema subset.

    Supports: ``type``, ``properties``, ``required``, ``items``, ``enum``,
    ``minLength``/``maxLength``, ``minimum``/``maximum``. Returns a list of
    human-readable violation strings (empty when conformant).
    """
    errors: list[str] = []
    expected = schema.get("type")
    if isinstance(expected, str) and not _json_type_matches(instance, expected):
        errors.append(f"{path}: expected type {expected}, got {type(instance).__name__}")
        return errors  # shape is wrong; deeper checks would cascade
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']!r}")
    if isinstance(instance, dict):
        for required in schema.get("required", []):
            if required not in instance:
                errors.append(f"{path}: missing required field {required!r}")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                if key in instance and isinstance(sub_schema, dict):
                    errors.extend(
                        validate_json_schema(instance[key], sub_schema, f"{path}.{key}")
                    )
    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, element in enumerate(instance):
                errors.extend(
                    validate_json_schema(element, items, f"{path}[{index}]")
                )
    if isinstance(instance, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(instance) < min_length:
            errors.append(f"{path}: length {len(instance)} < minLength {min_length}")
        if isinstance(max_length, int) and len(instance) > max_length:
            errors.append(f"{path}: length {len(instance)} > maxLength {max_length}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            errors.append(f"{path}: {instance} < minimum {minimum}")
        if isinstance(maximum, (int, float)) and instance > maximum:
            errors.append(f"{path}: {instance} > maximum {maximum}")
    return errors


def conformity_errors(
    value: Any, schema: dict[str, Any] | type[BaseModel]
) -> list[str]:
    """Validate ``value`` against a JSON-schema dict or pydantic model class.

    Returns human-readable violation strings (empty when conformant); the
    public entry point used by the pytest plugin's ``assert_conformant``.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if not isinstance(value, dict):
            return [f"expected object for {schema.__name__}, got {type(value).__name__}"]
        try:
            schema.model_validate(value)
        except ValidationError as exc:
            return [
                f"{'.'.join(str(loc) for loc in err['loc']) or '$'}: {err['msg']}"
                for err in exc.errors()
            ]
        return []
    if isinstance(schema, dict):
        return validate_json_schema(value, schema)
    return []


def _coerce_structured(value: Any) -> tuple[Any, str | None]:
    """Parse string outputs into structures for schema checks.

    Returns ``(parsed, error)``: dict/list payloads pass through; strings are
    JSON-parsed with markdown-fence tolerance.
    """
    if isinstance(value, str):
        try:
            return json.loads(value), None
        except json.JSONDecodeError:
            pass
        try:
            return extract_json_object(value), None
        except LLMResponseError as exc:
            return None, f"not parseable as JSON: {exc}"
    return value, None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


class DeterministicSuite:
    """Ordered battery of zero-token checks over a batch of outputs.

    Per output, checks run in this fixed order: ``schema_conformity``,
    ``content_bounds`` (non-empty + length), ``json_valid``, ``similarity``,
    ``required_fields``, ``latency_budget``. Outputs are processed in sorted
    name order, so a report is bit-for-bit reproducible.

    Usage::

        suite = DeterministicSuite()
        report = suite.run_all(
            outputs={"answer": '{"ticket_id": "T-1"}'},
            schema={"answer": {"type": "object", "required": ["ticket_id"]}},
            refs={"answer": "ticket T-1 resolved"},
            budgets={"answer": 2000.0},
        )
        assert report.passed, report.summary()
    """

    def __init__(self, min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self.min_similarity = min_similarity
        self._embedder = HashEmbeddingClient()

    # Individual checks (each returns one CheckResult) --------------------

    def check_schema_conformity(
        self, name: str, value: Any, schema: dict[str, Any] | type[BaseModel] | None
    ) -> CheckResult:
        if schema is None:
            return CheckResult(
                name=f"schema_conformity:{name}", passed=True, detail="no schema declared"
            )
        parsed, parse_error = _coerce_structured(value)
        if parse_error is not None:
            return CheckResult(
                name=f"schema_conformity:{name}",
                passed=False,
                detail=f"{parse_error} (vs schema)",
                escalate=True,
            )
        errors = conformity_errors(parsed, schema)
        return CheckResult(
            name=f"schema_conformity:{name}",
            passed=not errors,
            detail="conformant" if not errors else "; ".join(errors[:5]),
            escalate=True,
        )

    def check_content_bounds(
        self, name: str, value: Any, schema: dict[str, Any] | None
    ) -> CheckResult:
        if value is None:
            return CheckResult(
                name=f"content_bounds:{name}", passed=False, detail="output is None/empty"
            )
        detail = "non-empty"
        if isinstance(value, str):
            if not value.strip():
                return CheckResult(
                    name=f"content_bounds:{name}",
                    passed=False,
                    detail="string output is empty/whitespace",
                )
            min_length, max_length = 1, DEFAULT_MAX_CHARS
            if isinstance(schema, dict):
                min_length = schema.get("minLength", min_length)
                max_length = schema.get("maxLength", max_length)
            if len(value) < min_length or len(value) > max_length:
                return CheckResult(
                    name=f"content_bounds:{name}",
                    passed=False,
                    detail=f"length {len(value)} outside [{min_length}, {max_length}]",
                )
            detail = f"non-empty, {len(value)} chars"
        elif isinstance(value, (dict, list)) and not value:
            return CheckResult(
                name=f"content_bounds:{name}",
                passed=False,
                detail=f"empty {type(value).__name__} output",
            )
        return CheckResult(name=f"content_bounds:{name}", passed=True, detail=detail)

    def check_json_valid(self, name: str, value: Any, structured: bool) -> CheckResult:
        if not structured:
            return CheckResult(
                name=f"json_valid:{name}", passed=True, detail="not declared structured"
            )
        if isinstance(value, (dict, list)):
            return CheckResult(
                name=f"json_valid:{name}", passed=True, detail="already structured"
            )
        parsed, parse_error = _coerce_structured(value)
        if parse_error is not None or parsed is None:
            return CheckResult(
                name=f"json_valid:{name}",
                passed=False,
                detail="structured output is not valid JSON",
                escalate=True,
            )
        return CheckResult(
            name=f"json_valid:{name}", passed=True, detail="valid JSON"
        )

    def check_similarity(self, name: str, value: Any, ref: str | dict[str, Any]) -> CheckResult:
        threshold = self.min_similarity
        reference = ref
        if isinstance(ref, dict):
            reference = ref.get("text", "")
            threshold = float(ref.get("min_similarity", threshold))
        score = self._embedder.similarity(_as_text(value), _as_text(reference))
        passed = score >= threshold
        return CheckResult(
            name=f"similarity:{name}",
            passed=passed,
            detail=f"cosine {score:.3f} vs threshold {threshold:.3f}",
            escalate=True,
        )

    def check_required_fields(
        self, name: str, value: Any, schema: dict[str, Any] | type[BaseModel] | None
    ) -> CheckResult:
        if schema is None:
            return CheckResult(
                name=f"required_fields:{name}", passed=True, detail="no schema declared"
            )
        required = _required_fields(schema)
        if not required:
            return CheckResult(
                name=f"required_fields:{name}", passed=True, detail="no required fields"
            )
        parsed, parse_error = _coerce_structured(value)
        if parse_error is not None or not isinstance(parsed, dict):
            return CheckResult(
                name=f"required_fields:{name}",
                passed=False,
                detail="cannot check required fields on a non-object output",
            )
        missing = [field_name for field_name in required if field_name not in parsed]
        return CheckResult(
            name=f"required_fields:{name}",
            passed=not missing,
            detail=(
                "all required fields present"
                if not missing
                else f"missing required fields: {', '.join(missing)}"
            ),
        )

    def check_latency_budget(self, name: str, value: Any, budget: float) -> CheckResult:
        latency = value.latency_ms if isinstance(value, TimedOutput) else None
        if latency is None:
            return CheckResult(
                name=f"latency_budget:{name}",
                passed=False,
                severity="warning",
                detail=f"budget {budget:g}ms declared but output reports no latency "
                "(wrap it in TimedOutput to enable the check)",
            )
        passed = latency <= budget
        return CheckResult(
            name=f"latency_budget:{name}",
            passed=passed,
            severity="warning",
            detail=f"{latency:g}ms measured vs {budget:g}ms budget",
        )

    # The suite -----------------------------------------------------------

    def _check_output(
        self,
        name: str,
        value: Any,
        out_schema: dict[str, Any] | type[BaseModel] | None,
        refs: dict[str, str | dict[str, Any]],
        budgets: dict[str, float],
    ) -> list[CheckResult]:
        """Run the ordered battery for one output (schema, bounds, JSON,
        similarity, required fields, latency — in that order)."""
        # Timing metadata rides along in TimedOutput; all non-latency checks
        # see the wrapped payload, the latency check sees the wrapper.
        payload = value.value if isinstance(value, TimedOutput) else value
        json_schema = out_schema if isinstance(out_schema, dict) else None
        structured = out_schema is not None
        results = [
            self.check_schema_conformity(name, payload, out_schema),
            self.check_content_bounds(name, payload, json_schema),
            self.check_json_valid(name, payload, structured),
        ]
        if name in refs:
            results.append(self.check_similarity(name, payload, refs[name]))
        else:
            results.append(
                CheckResult(name=f"similarity:{name}", passed=True, detail="no reference")
            )
        results.append(self.check_required_fields(name, payload, out_schema))
        if name in budgets:
            results.append(self.check_latency_budget(name, value, budgets[name]))
        else:
            results.append(
                CheckResult(
                    name=f"latency_budget:{name}",
                    passed=True,
                    detail="no budget declared",
                    severity="warning",
                )
            )
        return results

    def run_all(
        self,
        outputs: dict[str, Any],
        schema: dict[str, Any] | type[BaseModel] | None = None,
        refs: dict[str, str | dict[str, Any]] | None = None,
        budgets: dict[str, float] | None = None,
    ) -> SuiteReport:
        """Run the ordered battery over ``outputs`` (deterministic order).

        - ``outputs``: name -> value (``str``, ``dict``, ``list``, or
          :class:`TimedOutput` when a latency budget applies).
        - ``schema``: a bare JSON-schema dict (applies to every output), a
          pydantic model class, or a per-output ``{name: schema}`` mapping.
        - ``refs``: name -> reference text (or ``{"text": ..., "min_similarity": ...}``)
          enabling the cosine-similarity check for that output.
        - ``budgets``: name -> maximum latency in ms.
        """
        refs = refs or {}
        budgets = budgets or {}
        results: list[CheckResult] = []
        for name in sorted(outputs):
            out_schema = _schema_for(schema, name)
            results.extend(self._check_output(name, outputs[name], out_schema, refs, budgets))
        return SuiteReport(results=results)
