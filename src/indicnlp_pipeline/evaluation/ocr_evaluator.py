"""Evaluate OCR outputs against verified page-level ground truth."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from indicnlp_pipeline.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
)


CSV_COLUMNS = [
    "annotation_ids",
    "page_id",
    "language",
    "annotation_status",
    "ocr_engine",
    "preprocessing_variant",
    "reference_characters",
    "hypothesis_characters",
    "cer",
    "wer",
    "reference_text",
    "hypothesis_text",
]


def _load_verified_ground_truth(
    ground_truth_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load verified annotations and group their text by page ID."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with ground_truth_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}."
                )

            if record.get("annotation_status") != "verified":
                continue

            page_id = record.get("page_id")
            text = record.get("text")
            annotation_id = record.get("annotation_id")
            language = record.get("language")

            if not isinstance(page_id, str) or not page_id:
                raise ValueError(
                    f"Missing or invalid page_id at line {line_number}."
                )
            if not isinstance(text, str):
                raise ValueError(
                    f"Missing or invalid text at line {line_number}."
                )
            if not isinstance(annotation_id, str) or not annotation_id:
                raise ValueError(
                    f"Missing or invalid annotation_id at line {line_number}."
                )
            if not isinstance(language, str) or not language:
                raise ValueError(
                    f"Missing or invalid language at line {line_number}."
                )

            grouped[page_id].append(record)

    ground_truth: dict[str, dict[str, Any]] = {}

    for page_id, records in grouped.items():
        records.sort(key=lambda record: record["annotation_id"])
        languages = {record["language"] for record in records}

        if len(languages) != 1:
            raise ValueError(
                f"Multiple languages found in verified annotations "
                f"for page {page_id}: {sorted(languages)}"
            )

        ground_truth[page_id] = {
            "annotation_ids": ";".join(
                record["annotation_id"] for record in records
            ),
            "language": records[0]["language"],
            "annotation_status": "verified",
            "reference_text": " ".join(
                record["text"] for record in records
            ),
        }

    if not ground_truth:
        raise ValueError(
            f"No verified ground-truth records found in {ground_truth_path}."
        )

    return ground_truth


def _extract_hypothesis(data: dict[str, Any], path: Path) -> str:
    """Extract OCR text while preserving Tesseract text verbatim."""
    text = data.get("text")

    if isinstance(text, str):
        return text

    if isinstance(text, list) and all(
        isinstance(item, str) for item in text
    ):
        # PaddleOCR returns individual text snippets. Join them in their
        # existing order; this does not reconstruct the original layout.
        return " ".join(text)

    raise ValueError(
        f"Unsupported or invalid OCR 'text' field in {path}."
    )


def evaluate_ocr_outputs(
    ocr_root: Path,
    ground_truth_path: Path,
    output_csv_path: Path,
) -> int:
    """Evaluate matching OCR files and write a UTF-8 CSV.

    Args:
        ocr_root: Directory containing one subdirectory per OCR engine.
        ground_truth_path: Verified ground-truth JSONL file.
        output_csv_path: Destination CSV path.

    Returns:
        Number of evaluation rows written.

    Raises:
        FileNotFoundError: If required input paths do not exist.
        ValueError: If input records or OCR output schemas are invalid.
    """
    ocr_root = Path(ocr_root)
    ground_truth_path = Path(ground_truth_path)
    output_csv_path = Path(output_csv_path)

    if not ocr_root.is_dir():
        raise FileNotFoundError(f"OCR directory not found: {ocr_root}")
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {ground_truth_path}"
        )

    ground_truth = _load_verified_ground_truth(ground_truth_path)
    rows: list[dict[str, Any]] = []
    matched_pages: set[str] = set()

    for engine_dir in sorted(
        path for path in ocr_root.iterdir() if path.is_dir()
    ):
        for ocr_path in sorted(engine_dir.glob("*.json")):
            try:
                with ocr_path.open(
                    "r", encoding="utf-8-sig"
                ) as file:
                    data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid OCR JSON in {ocr_path}: {exc}"
                ) from exc

            if not isinstance(data, dict):
                raise ValueError(
                    f"Expected a JSON object in {ocr_path}."
                )

            page_id = data.get("source_page")
            if not isinstance(page_id, str) or not page_id:
                raise ValueError(
                    f"Missing or invalid source_page in {ocr_path}."
                )

            reference = ground_truth.get(page_id)
            if reference is None:
                logger.warning(
                    "Skipping OCR file without verified ground truth: {}",
                    ocr_path,
                )
                continue

            engine = data.get("ocr_engine", engine_dir.name)
            variant = data.get("preprocessing_variant")

            if not isinstance(engine, str) or not engine:
                raise ValueError(
                    f"Missing or invalid ocr_engine in {ocr_path}."
                )
            if not isinstance(variant, str) or not variant:
                raise ValueError(
                    f"Missing or invalid preprocessing_variant in {ocr_path}."
                )

            hypothesis = _extract_hypothesis(data, ocr_path)
            reference_text = reference["reference_text"]

            row = {
                "annotation_ids": reference["annotation_ids"],
                "page_id": page_id,
                "language": reference["language"],
                "annotation_status": reference["annotation_status"],
                "ocr_engine": engine,
                "preprocessing_variant": variant,
                "reference_characters": len(reference_text),
                "hypothesis_characters": len(hypothesis),
                "cer": calculate_cer(reference_text, hypothesis),
                "wer": calculate_wer(reference_text, hypothesis),
                "reference_text": reference_text,
                "hypothesis_text": hypothesis,
            }
            rows.append(row)
            matched_pages.add(page_id)

    if not rows:
        raise ValueError(
            "No OCR files matched verified ground-truth pages."
        )

    missing_pages = set(ground_truth) - matched_pages
    if missing_pages:
        logger.warning(
            "Ground-truth pages without any matching OCR output: {}",
            sorted(missing_pages),
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Wrote {} OCR evaluation rows to {}",
        len(rows),
        output_csv_path,
    )
    return len(rows)
