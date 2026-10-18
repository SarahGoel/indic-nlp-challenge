import pytest

from indicnlp_pipeline.error_analysis.classifier import (
    ERROR_RECORD_FIELDS,
    ClassificationResult,
    ErrorCategory,
    ErrorRecord,
    ErrorSeverity,
    PipelineStage,
    classify_error,
    make_error_id,
)


@pytest.mark.parametrize(
    ("category", "stage"),
    [
        (ErrorCategory.OCR, PipelineStage.OCR),
        (ErrorCategory.TRANSLATION, PipelineStage.TRANSLATION),
        (ErrorCategory.TERMINOLOGY, PipelineStage.TRANSLATION),
        (ErrorCategory.SEGMENTATION, PipelineStage.SEGMENTATION),
    ],
)
def test_explicit_category_is_classified(category, stage):
    result = classify_error(
        stage,
        {"rationale": "Supported by reviewed evidence."},
        category=category,
        severity=ErrorSeverity.HIGH,
    )

    assert isinstance(result, ClassificationResult)
    assert result.category is category
    assert result.stage is stage
    assert result.severity is ErrorSeverity.HIGH
    assert result.requires_review is False
    assert result.rationale == "Supported by reviewed evidence."


@pytest.mark.parametrize(
    "severity",
    [
        ErrorSeverity.LOW,
        ErrorSeverity.MEDIUM,
        ErrorSeverity.HIGH,
        ErrorSeverity.CRITICAL,
    ],
)
def test_all_severity_levels_are_supported(severity):
    result = classify_error(
        "ocr",
        {"rationale": "Reviewed OCR evidence."},
        category="ocr",
        severity=severity.value,
    )

    assert result.severity is severity


def test_single_supported_category_evidence_is_classified():
    result = classify_error(
        "translation",
        {
            "category_evidence": {
                "terminology": "A domain expert confirmed the term is incorrect."
            }
        },
        severity="high",
    )

    assert result.category is ErrorCategory.TERMINOLOGY
    assert result.severity is ErrorSeverity.HIGH
    assert result.requires_review is False


def test_missing_category_evidence_requires_review():
    result = classify_error("ocr", {})

    assert result.category is None
    assert result.requires_review is True
    assert result.severity is None


def test_empty_category_evidence_requires_review():
    result = classify_error("translation", {"category_evidence": {}})

    assert result.category is None
    assert result.requires_review is True


def test_conflicting_category_evidence_requires_review():
    result = classify_error(
        "translation",
        {
            "category_evidence": {
                "translation": "Possible mistranslation.",
                "terminology": "Possible terminology error.",
            }
        },
    )

    assert result.category is None
    assert result.requires_review is True


@pytest.mark.parametrize(
    ("stage", "evidence", "kwargs", "exception"),
    [
        ("invalid-stage", {}, {}, ValueError),
        ("ocr", {}, {"category": "invalid-category"}, ValueError),
        ("ocr", {}, {"severity": "extreme"}, ValueError),
        ("ocr", [], {}, TypeError),
        ("ocr", {"category_evidence": []}, {}, TypeError),
        (
            "ocr",
            {"category_evidence": {"ocr": 123}},
            {},
            TypeError,
        ),
        (
            "ocr",
            {"category_evidence": {"invalid-category": "Evidence."}},
            {},
            ValueError,
        ),
    ],
)
def test_invalid_classification_inputs_are_rejected(
    stage, evidence, kwargs, exception
):
    with pytest.raises(exception):
        classify_error(stage, evidence, **kwargs)


def test_explicit_category_requires_nonempty_rationale():
    with pytest.raises(ValueError):
        classify_error(
            "ocr",
            {"rationale": "   "},
            category="ocr",
        )


def test_devanagari_text_is_preserved_in_error_record():
    system_output = "ग्रहों का फल"
    reference_text = "ग्रहों के फल"

    record = ErrorRecord(
        error_id="ERR-TEST-001",
        segment_id="BRIGU_V5_P0001_S001",
        stage=PipelineStage.TRANSLATION,
        category=ErrorCategory.TERMINOLOGY,
        system_output=system_output,
        reference_text=reference_text,
        severity=ErrorSeverity.HIGH,
    )

    assert record.to_dict()["system_output"] == system_output
    assert record.to_dict()["reference_text"] == reference_text


def test_error_record_uses_canonical_field_order():
    record = ErrorRecord(
        error_id="ERR-TEST-002",
        segment_id="BRIGU_V5_P0001_S002",
        stage=PipelineStage.OCR,
        category=ErrorCategory.OCR,
        system_output="श्लोक",
        reference_text="श्लोक",
        severity=ErrorSeverity.LOW,
    )

    assert tuple(record.to_dict()) == ERROR_RECORD_FIELDS


@pytest.mark.parametrize(
    ("error_id", "segment_id"),
    [
        ("", "SEG-001"),
        ("   ", "SEG-001"),
        ("ERR-001", ""),
        ("ERR-001", "   "),
    ],
)
def test_error_record_rejects_empty_identifiers(error_id, segment_id):
    with pytest.raises(ValueError):
        ErrorRecord(
            error_id=error_id,
            segment_id=segment_id,
            stage=PipelineStage.OCR,
            category=ErrorCategory.OCR,
            system_output="text",
            reference_text="reference",
            severity=ErrorSeverity.LOW,
        )


def test_error_record_rejects_non_string_text_fields():
    with pytest.raises(TypeError):
        ErrorRecord(
            error_id="ERR-003",
            segment_id="SEG-003",
            stage=PipelineStage.OCR,
            category=ErrorCategory.OCR,
            system_output=None,
            reference_text="reference",
            severity=ErrorSeverity.LOW,
        )


def test_error_id_is_deterministic():
    args = (
        "BRIGU_V5_P0001_S001",
        PipelineStage.OCR,
        ErrorCategory.OCR,
        "श्लोक",
        "श्लोक",
    )

    assert make_error_id(*args) == make_error_id(*args)


def test_error_id_changes_when_record_content_changes():
    original = make_error_id(
        "SEG-001", "ocr", "ocr", "श्लोक", "श्लोक"
    )
    changed = make_error_id(
        "SEG-001", "ocr", "ocr", "श्लोक", "श्लोकः"
    )

    assert original != changed


@pytest.mark.parametrize(
    ("segment_id", "stage", "category", "output", "reference", "exception"),
    [
        ("", "ocr", "ocr", "text", "text", ValueError),
        ("SEG-001", "invalid-stage", "ocr", "text", "text", ValueError),
        ("SEG-001", "ocr", "invalid-category", "text", "text", ValueError),
        ("SEG-001", "ocr", "ocr", None, "text", TypeError),
        ("SEG-001", "ocr", "ocr", "text", None, TypeError),
    ],
)
def test_error_id_rejects_invalid_inputs(
    segment_id, stage, category, output, reference, exception
):
    with pytest.raises(exception):
        make_error_id(segment_id, stage, category, output, reference)