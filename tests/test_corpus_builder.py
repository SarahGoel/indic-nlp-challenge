import json
from pathlib import Path

import pytest

from indicnlp_pipeline.utils.corpus_builder import (
    CorpusValidationError,
    build_corpus_records,
    validate_corpus_record,
    write_corpus_jsonl,
)


def _valid_record():
    return {
        "id": "brigu_v5_p0005_seg001",
        "source_language": "hin",
        "source_script": "Devanagari",
        "page": 5,
        "segment_id": "BRIGU_V5_P0005_SEG001",
        "source_page": "BRIGU_V5_P0005.png",
        "ocr": {
            "tesseract": {},
            "paddleocr": {},
        },
        "ocr_corrected": None,
        "ground_truth": "परीक्षण पाठ।",
        "ground_truth_metadata": {
            "annotation_id": "BRIGU_V5_P0005_SEG001",
            "page_id": "BRIGU_V5_P0005",
            "annotation_status": "verified",
            "confidence": 1.0,
            "annotator": "human",
            "notes": None,
        },
        "human_verified": True,
        "translations": {
            "hin": {
                "indictrans2": None,
                "gemini": None,
                "human": None,
            }
        },
    }


def test_valid_record_passes_validation():
    validate_corpus_record(_valid_record())


def test_missing_required_field_is_rejected():
    record = _valid_record()
    del record["ground_truth"]

    with pytest.raises(CorpusValidationError, match="ground_truth"):
        validate_corpus_record(record)


def test_invalid_page_type_is_rejected():
    record = _valid_record()
    record["page"] = "five"

    with pytest.raises(CorpusValidationError, match="integer"):
        validate_corpus_record(record)


def test_jsonl_preserves_devanagari_and_writes_one_record_per_line(tmp_path):
    destination = tmp_path / "corpus.jsonl"
    records = [_valid_record(), {**_valid_record(), "id": "second_record"}]

    write_corpus_jsonl(records, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["ground_truth"] == "परीक्षण पाठ।"
    assert not destination.read_bytes().startswith(b"\xef\xbb\xbf")


def test_existing_output_is_not_overwritten(tmp_path):
    destination = tmp_path / "corpus.jsonl"
    destination.write_text("keep this\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_corpus_jsonl([_valid_record()], destination)

    assert destination.read_text(encoding="utf-8") == "keep this\n"


def test_builder_uses_ground_truth_records_and_ocr_metadata(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data" / "ground_truth").mkdir(parents=True)
    (root / "data" / "raw_scans").mkdir(parents=True)
    for engine in ("tesseract", "paddleocr"):
        (root / "data" / "ocr_raw" / engine).mkdir(parents=True)

    (root / "configs" / "config.yaml").write_text(
        "project:\n"
        "  target_languages: [hin, ben]\n",
        encoding="utf-8",
    )
    (root / "data" / "raw_scans" / "page_manifest.csv").write_text(
        "page_id,pdf_page_number\nBRIGU_V5_P0005,5\n",
        encoding="utf-8",
    )

    annotation = {
        "annotation_id": "BRIGU_V5_P0005_SEG001",
        "page_id": "BRIGU_V5_P0005",
        "source_page": "BRIGU_V5_P0005.png",
        "language": "hin",
        "text": "मूल सत्यापित पाठ।",
        "annotation_status": "verified",
        "confidence": 1.0,
        "annotator": "human",
        "notes": None,
    }
    (root / "data" / "ground_truth" / "ground_truth.jsonl").write_text(
        json.dumps(annotation, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tesseract = {
        "source_page": "BRIGU_V5_P0005",
        "preprocessing_variant": "adaptive_threshold",
        "ocr_engine": "tesseract",
        "language": "san+hin",
        "text": "raw tesseract text",
    }
    paddleocr = {
        "source_page": "BRIGU_V5_P0005",
        "preprocessing_variant": "adaptive_threshold",
        "ocr_engine": "paddleocr",
        "text": ["raw", "paddle", "text"],
        "confidence": [0.8, 0.9, 0.7],
        "bounding_boxes": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
    }

    (root / "data" / "ocr_raw" / "tesseract" /
     "BRIGU_V5_P0005__adaptive_threshold.json").write_text(
        json.dumps(tesseract, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "data" / "ocr_raw" / "paddleocr" /
     "BRIGU_V5_P0005__adaptive_threshold.json").write_text(
        json.dumps(paddleocr, ensure_ascii=False),
        encoding="utf-8",
    )

    records = build_corpus_records(root)

    assert len(records) == 1
    record = records[0]
    assert record["source_language"] == "hin"
    assert record["ground_truth"] == "मूल सत्यापित पाठ।"
    assert record["human_verified"] is True
    assert record["ocr"]["tesseract"]["adaptive_threshold"]["text"] == (
        "raw tesseract text"
    )
    assert record["ocr"]["paddleocr"]["adaptive_threshold"]["confidence"] == [
        0.8, 0.9, 0.7
    ]
    assert record["translations"]["ben"]["gemini"] is None
