"""Evidence-based error classification for the Indic NLP pipeline.

The classifier does not infer error causes from arbitrary text differences.
Callers must supply an explicit category or unambiguous category evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ErrorCategory(str, Enum):
    """Supported error categories."""

    OCR = "ocr"
    TRANSLATION = "translation"
    TERMINOLOGY = "terminology"
    SEGMENTATION = "segmentation"


class ErrorSeverity(str, Enum):
    """Impact-based error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    """Supported pipeline stages where an error can be observed."""

    OCR = "ocr"
    NORMALIZATION = "normalization"
    SEGMENTATION = "segmentation"
    TRANSLATION = "translation"


ERROR_RECORD_FIELDS: tuple[str, ...] = (
    "error_id",
    "segment_id",
    "stage",
    "category",
    "system_output",
    "reference_text",
    "severity",
)


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of classifying a suspected pipeline error.

    A missing category indicates that the evidence is ambiguous or
    insufficient and the case requires review.
    """

    stage: PipelineStage
    category: ErrorCategory | None
    severity: ErrorSeverity | None
    requires_review: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Validated error record matching the required export schema."""

    error_id: str
    segment_id: str
    stage: PipelineStage
    category: ErrorCategory
    system_output: str
    reference_text: str
    severity: ErrorSeverity

    def __post_init__(self) -> None:
        """Reject empty identifiers and non-string text fields."""
        if not self.error_id.strip():
            raise ValueError("error_id must not be empty.")
        if not self.segment_id.strip():
            raise ValueError("segment_id must not be empty.")
        if not isinstance(self.system_output, str):
            raise TypeError("system_output must be a string.")
        if not isinstance(self.reference_text, str):
            raise TypeError("reference_text must be a string.")

    def to_dict(self) -> dict[str, str]:
        """Return fields in the canonical CSV column order."""
        return {
            "error_id": self.error_id,
            "segment_id": self.segment_id,
            "stage": self.stage.value,
            "category": self.category.value,
            "system_output": self.system_output,
            "reference_text": self.reference_text,
            "severity": self.severity.value,
        }


def _enum_value(
    enum_type: type[Enum],
    value: Enum | str,
    field_name: str,
) -> Enum:
    """Validate and convert a string or enum to its canonical enum."""
    if isinstance(value, enum_type):
        return value

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or {enum_type.__name__}.")

    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(
            f"Invalid {field_name} {value!r}. Allowed values: {allowed}."
        ) from exc


def classify_error(
    stage: PipelineStage | str,
    evidence: Mapping[str, object],
    *,
    category: ErrorCategory | str | None = None,
    severity: ErrorSeverity | str | None = None,
) -> ClassificationResult:
    """Classify a suspected error using explicit, auditable evidence.

    Args:
        stage: Pipeline stage where the suspected error was observed.
        evidence: Evidence mapping. When category is omitted, the optional
            ``category_evidence`` mapping must contain exactly one category
            with a non-empty textual rationale.
        category: Explicit category supported by annotation or review.
        severity: Explicit impact-based severity supported by evidence.

    Returns:
        A deterministic result. Insufficient or ambiguous category evidence
        produces ``requires_review=True`` and no category.

    Raises:
        TypeError: If the stage or evidence has an invalid type.
        ValueError: If a supplied stage, category, or severity is invalid.
    """
    validated_stage = _enum_value(PipelineStage, stage, "stage")

    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping.")

    validated_severity = (
        _enum_value(ErrorSeverity, severity, "severity")
        if severity is not None
        else None
    )

    if category is not None:
        validated_category = _enum_value(
            ErrorCategory, category, "category"
        )
        rationale = evidence.get("rationale", "Explicit category supplied.")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                "An explicit category requires a non-empty rationale."
            )
        return ClassificationResult(
            stage=validated_stage,
            category=validated_category,
            severity=validated_severity,
            requires_review=False,
            rationale=rationale.strip(),
        )

    category_evidence = evidence.get("category_evidence", {})
    if not isinstance(category_evidence, Mapping):
        raise TypeError("category_evidence must be a mapping.")

    supported: list[tuple[ErrorCategory, str]] = []
    for raw_category, rationale in category_evidence.items():
        validated_category = _enum_value(
            ErrorCategory, raw_category, "category_evidence key"
        )
        if not isinstance(rationale, str):
            raise TypeError("Each category evidence rationale must be a string.")
        if rationale.strip():
            supported.append((validated_category, rationale.strip()))

    if len(supported) != 1:
        return ClassificationResult(
            stage=validated_stage,
            category=None,
            severity=validated_severity,
            requires_review=True,
            rationale=(
                "Insufficient or conflicting category evidence; human review required."
            ),
        )

    selected_category, rationale = supported[0]
    return ClassificationResult(
        stage=validated_stage,
        category=selected_category,
        severity=validated_severity,
        requires_review=False,
        rationale=rationale,
    )


def make_error_id(
    segment_id: str,
    stage: PipelineStage | str,
    category: ErrorCategory | str,
    system_output: str,
    reference_text: str,
) -> str:
    """Generate a deterministic identifier for an unchanged error record.

    The identifier is derived from the segment, stage, category, and text.
    Identical records receive identical IDs across repeated runs.
    """
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise ValueError("segment_id must be a non-empty string.")
    if not isinstance(system_output, str):
        raise TypeError("system_output must be a string.")
    if not isinstance(reference_text, str):
        raise TypeError("reference_text must be a string.")

    validated_stage = _enum_value(PipelineStage, stage, "stage")
    validated_category = _enum_value(ErrorCategory, category, "category")

    payload = json.dumps(
        [
            segment_id,
            validated_stage.value,
            validated_category.value,
            system_output,
            reference_text,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"ERR-{digest}"