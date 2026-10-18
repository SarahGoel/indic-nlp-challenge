import csv
import json

import pytest

from indicnlp_pipeline.evaluation.ocr_evaluator import (
    evaluate_ocr_outputs,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_ground_truth(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in records
        ) + "\n",
        encoding="utf-8",
    )


def test_evaluate_ocr_outputs_writes_rows_for_each_variant(tmp_path):
    ocr_root = tmp_path / "ocr_raw"
    gt_path = tmp_path / "ground_truth.jsonl"
    output_path = tmp_path / "metrics" / "ocr_evaluation.csv"

    _write_ground_truth(gt_path, [
        {
            "annotation_id": "PAGE1_SEG001",
            "page_id": "PAGE1",
            "language": "hin",
            "text": "नमस्ते दुनिया",
            "annotation_status": "verified",
        }
    ])

    for variant, text in [
        ("grayscale", "नमस्ते दुनिया"),
        ("otsu", "नमस्ते दुनया"),
    ]:
        _write_json(
            ocr_root / "tesseract" / f"PAGE1__{variant}.json",
            {
                "source_page": "PAGE1",
                "preprocessing_variant": variant,
                "ocr_engine": "tesseract",
                "text": text,
            },
        )

    count = evaluate_ocr_outputs(ocr_root, gt_path, output_path)

    assert count == 2
    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert {row["preprocessing_variant"] for row in rows} == {
        "grayscale",
        "otsu",
    }
    assert all(row["page_id"] == "PAGE1" for row in rows)
    assert all(row["annotation_status"] == "verified" for row in rows)
    assert float(next(
        row["cer"] for row in rows
        if row["preprocessing_variant"] == "grayscale"
    )) == 0.0


def test_paddleocr_snippets_are_joined_in_order(tmp_path):
    ocr_root = tmp_path / "ocr_raw"
    gt_path = tmp_path / "ground_truth.jsonl"
    output_path = tmp_path / "ocr_evaluation.csv"

    _write_ground_truth(gt_path, [
        {
            "annotation_id": "PAGE1_SEG001",
            "page_id": "PAGE1",
            "language": "hin",
            "text": "नमस्ते दुनिया",
            "annotation_status": "verified",
        }
    ])

    _write_json(
        ocr_root / "paddleocr" / "PAGE1__grayscale.json",
        {
            "source_page": "PAGE1",
            "preprocessing_variant": "grayscale",
            "ocr_engine": "paddleocr",
            "text": ["नमस्ते", "दुनिया"],
            "confidence": [0.99, 0.98],
            "bounding_boxes": [],
        },
    )

    evaluate_ocr_outputs(ocr_root, gt_path, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))

    assert row["hypothesis_text"] == "नमस्ते दुनिया"
    assert float(row["cer"]) == 0.0
    assert float(row["wer"]) == 0.0


def test_unverified_annotations_are_not_used(tmp_path):
    ocr_root = tmp_path / "ocr_raw"
    gt_path = tmp_path / "ground_truth.jsonl"
    output_path = tmp_path / "ocr_evaluation.csv"

    _write_ground_truth(gt_path, [
        {
            "annotation_id": "PAGE1_SEG001",
            "page_id": "PAGE1",
            "language": "hin",
            "text": "unverified text",
            "annotation_status": "draft",
        }
    ])

    _write_json(
        ocr_root / "tesseract" / "PAGE1__grayscale.json",
        {
            "source_page": "PAGE1",
            "preprocessing_variant": "grayscale",
            "ocr_engine": "tesseract",
            "text": "unverified text",
        },
    )

    with pytest.raises(ValueError, match="No verified ground-truth"):
        evaluate_ocr_outputs(ocr_root, gt_path, output_path)


def test_invalid_ocr_text_schema_raises_value_error(tmp_path):
    ocr_root = tmp_path / "ocr_raw"
    gt_path = tmp_path / "ground_truth.jsonl"
    output_path = tmp_path / "ocr_evaluation.csv"

    _write_ground_truth(gt_path, [
        {
            "annotation_id": "PAGE1_SEG001",
            "page_id": "PAGE1",
            "language": "hin",
            "text": "नमस्ते",
            "annotation_status": "verified",
        }
    ])

    _write_json(
        ocr_root / "tesseract" / "PAGE1__grayscale.json",
        {
            "source_page": "PAGE1",
            "preprocessing_variant": "grayscale",
            "ocr_engine": "tesseract",
            "text": None,
        },
    )

    with pytest.raises(ValueError, match="Unsupported or invalid OCR"):
        evaluate_ocr_outputs(ocr_root, gt_path, output_path)
