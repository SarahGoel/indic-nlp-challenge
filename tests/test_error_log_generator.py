import csv
import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/06_generate_error_logs.py")
spec = importlib.util.spec_from_file_location("error_log_generator", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_csv_uses_canonical_field_order(tmp_path):
    output = tmp_path / "errors.csv"
    module.write_csv(output, [])

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == [
            "error_id",
            "segment_id",
            "stage",
            "category",
            "system_output",
            "reference_text",
            "severity",
        ]
        assert list(reader) == []


def test_whitespace_only_difference_is_not_an_error():
    assert (
        module.normalized_for_comparison("पहला   पाठ\nदूसरा")
        == module.normalized_for_comparison("पहला पाठ दूसरा")
    )


def test_severity_is_deterministic():
    reference = "यह एक परीक्षण वाक्य है।"
    output = "यह एक परीक्षण वाक्य है।"

    assert module.severity_for(reference, output) == "low"
    assert module.severity_for(reference, "पूर्णतः अलग") == "critical"


def test_make_record_preserves_devanagari():
    record = module.make_record(
        segment_id="BRIGU_V5_P0005_SEG001",
        stage="ocr",
        category="ocr",
        severity="medium",
        system_output="परीक्षण पाठ",
        reference_text="सही पाठ",
    )

    assert list(record) == [
        "error_id",
        "segment_id",
        "stage",
        "category",
        "system_output",
        "reference_text",
        "severity",
    ]
    assert record["system_output"] == "परीक्षण पाठ"
    assert record["reference_text"] == "सही पाठ"
    assert record["error_id"].startswith("ERR-")


def test_error_ids_are_deterministic():
    first = module.make_record(
        segment_id="BRIGU_V5_P0005_SEG001",
        stage="ocr",
        category="ocr",
        severity="low",
        system_output="गलत",
        reference_text="सही",
    )
    second = module.make_record(
        segment_id="BRIGU_V5_P0005_SEG001",
        stage="ocr",
        category="ocr",
        severity="low",
        system_output="गलत",
        reference_text="सही",
    )
    assert first["error_id"] == second["error_id"]


def test_ocr_comparison_requires_verified_reference(tmp_path, monkeypatch):
    comparisons = tmp_path / "ocr_comparisons"
    comparisons.mkdir()
    monkeypatch.setattr(module, "COMPARISON_DIR", comparisons)

    (comparisons / "BRIGU_V5_P0005__grayscale.json").write_text(
        '{"source_page":"BRIGU_V5_P0005",'
        '"preprocessing_variant":"grayscale",'
        '"comparison":{"tesseract_text":"गलत पाठ",'
        '"paddleocr_text":"गलत पाठ"}}',
        encoding="utf-8",
    )

    rows, provenance, stats = module.build_ocr_errors({})
    assert rows == []
    assert provenance == []
    assert stats["unmatched_reference_files"] == 1



def test_page_level_ocr_is_skipped_for_segment_reference(
    tmp_path, monkeypatch
):
    comparisons = tmp_path / "ocr_comparisons"
    comparisons.mkdir()
    monkeypatch.setattr(module, "COMPARISON_DIR", comparisons)

    (comparisons / "BRIGU_V5_P0005__grayscale.json").write_text(
        '{"source_page":"BRIGU_V5_P0005",'
        '"preprocessing_variant":"grayscale",'
        '"comparison":{"tesseract_text":"पृष्ठ-स्तरीय OCR",'
        '"paddleocr_text":"पृष्ठ-स्तरीय OCR"}}',
        encoding="utf-8",
    )

    reference = {
        "segment_id": "BRIGU_V5_P0005_SEG001",
        "ground_truth": "यह सत्यापित खंड है।",
    }

    rows, provenance, stats = module.build_ocr_errors(
        {"BRIGU_V5_P0005": reference}
    )

    assert rows == []
    assert provenance == []
    assert stats["matched_reference_files"] == 1
    assert stats["unaligned_reference_files"] == 1
    assert stats["mismatching_engine_outputs"] == 0
