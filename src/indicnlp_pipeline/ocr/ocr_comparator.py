from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from rapidfuzz.distance import Levenshtein


class OCRComparisonError(RuntimeError):
    """Raised when OCR comparison processing fails."""


def load_json_output(json_path: Path) -> dict[str, Any]:
    """
    Load a structured OCR JSON output.

    Args:
        json_path: Path to the OCR JSON file.

    Returns:
        Parsed OCR output dictionary.

    Raises:
        TypeError: If json_path is not a pathlib.Path.
        FileNotFoundError: If the file does not exist.
        OCRComparisonError: If the JSON is malformed or invalid.
    """
    if not isinstance(json_path, Path):
        raise TypeError("json_path must be a pathlib.Path instance.")

    if not json_path.is_file():
        raise FileNotFoundError(f"OCR output not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.exception("Failed to load OCR output: {}", json_path)
        raise OCRComparisonError(
            f"Unable to load OCR output: {json_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise OCRComparisonError(
            f"OCR output must contain a JSON object: {json_path}"
        )

    return payload


def extract_tesseract_text(payload: dict[str, Any]) -> str:
    """
    Extract Tesseract text from a raw OCR payload.

    Args:
        payload: Tesseract OCR result.

    Returns:
        Recognized Tesseract text.

    Raises:
        OCRComparisonError: If the expected text field is invalid.
    """
    text = payload.get("text")

    if not isinstance(text, str):
        raise OCRComparisonError(
            "Tesseract OCR output contains an invalid text field."
        )

    return text


def extract_paddleocr_text(payload: dict[str, Any]) -> str:
    """
    Convert PaddleOCR recognized text regions into comparison text.

    Args:
        payload: PaddleOCR result.

    Returns:
        Recognized PaddleOCR text joined in detection order.

    Raises:
        OCRComparisonError: If the expected text field is invalid.
    """
    text = payload.get("text")

    if not isinstance(text, list):
        raise OCRComparisonError(
            "PaddleOCR output contains an invalid text field."
        )

    if not all(isinstance(item, str) for item in text):
        raise OCRComparisonError(
            "PaddleOCR text entries must all be strings."
        )

    return "\n".join(text)


def build_diff_operations(
    source: str,
    target: str,
) -> list[dict[str, Any]]:
    """
    Generate character or token-level Levenshtein operations.

    Args:
        source: Source sequence.
        target: Target sequence.

    Returns:
        Structured edit operations.
    """
    operations: list[dict[str, Any]] = []

    for tag, source_start, source_end, target_start, target_end in (
        Levenshtein.opcodes(source, target)
    ):
        if tag == "equal":
            continue

        operations.append(
            {
                "operation": tag,
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "source": source[source_start:source_end],
                "target": target[target_start:target_end],
            }
        )

    return operations


def build_word_diff(
    tesseract_text: str,
    paddle_text: str,
) -> list[dict[str, Any]]:
    """
    Generate word-level differences.

    Args:
        tesseract_text: Tesseract OCR text.
        paddle_text: PaddleOCR OCR text.

    Returns:
        Structured word-level differences.
    """
    tesseract_words = tesseract_text.split()
    paddle_words = paddle_text.split()

    operations: list[dict[str, Any]] = []

    for (
        tag,
        source_start,
        source_end,
        target_start,
        target_end,
    ) in Levenshtein.opcodes(
        tesseract_words,
        paddle_words,
    ):
        if tag == "equal":
            continue

        operations.append(
            {
                "operation": tag,
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "source": tesseract_words[source_start:source_end],
                "target": paddle_words[target_start:target_end],
            }
        )

    return operations


def calculate_similarity(
    tesseract_text: str,
    paddle_text: str,
) -> dict[str, float]:
    """
    Calculate OCR text similarity and edit-distance information.

    Args:
        tesseract_text: Tesseract OCR text.
        paddle_text: PaddleOCR OCR text.

    Returns:
        Similarity and distance statistics.
    """
    character_distance = Levenshtein.distance(
        tesseract_text,
        paddle_text,
    )

    character_max_length = max(
        len(tesseract_text),
        len(paddle_text),
    )

    if character_max_length == 0:
        character_similarity = 1.0
    else:
        character_similarity = (
            1.0
            - character_distance / character_max_length
        )

    tesseract_words = tesseract_text.split()
    paddle_words = paddle_text.split()

    word_distance = Levenshtein.distance(
        tesseract_words,
        paddle_words,
    )

    word_max_length = max(
        len(tesseract_words),
        len(paddle_words),
    )

    if word_max_length == 0:
        word_similarity = 1.0
    else:
        word_similarity = (
            1.0
            - word_distance / word_max_length
        )

    return {
        "character_similarity": character_similarity,
        "character_distance": float(character_distance),
        "word_similarity": word_similarity,
        "word_distance": float(word_distance),
    }


def compare_ocr_outputs(
    tesseract_path: Path,
    paddleocr_path: Path,
) -> dict[str, Any]:
    """
    Compare one Tesseract output against one PaddleOCR output.

    Args:
        tesseract_path: Tesseract JSON output path.
        paddleocr_path: PaddleOCR JSON output path.

    Returns:
        Structured OCR comparison result.

    Raises:
        OCRComparisonError: If outputs are malformed or mismatched.
    """
    tesseract_payload = load_json_output(tesseract_path)
    paddle_payload = load_json_output(paddleocr_path)

    tesseract_page = tesseract_payload.get("source_page")
    paddle_page = paddle_payload.get("source_page")

    tesseract_variant = tesseract_payload.get(
        "preprocessing_variant"
    )
    paddle_variant = paddle_payload.get(
        "preprocessing_variant"
    )

    if tesseract_page != paddle_page:
        raise OCRComparisonError(
            "OCR outputs refer to different source pages: "
            f"{tesseract_page!r} vs {paddle_page!r}"
        )

    if tesseract_variant != paddle_variant:
        raise OCRComparisonError(
            "OCR outputs use different preprocessing variants: "
            f"{tesseract_variant!r} vs {paddle_variant!r}"
        )

    tesseract_text = extract_tesseract_text(
        tesseract_payload
    )
    paddle_text = extract_paddleocr_text(
        paddle_payload
    )

    character_differences = build_diff_operations(
        tesseract_text,
        paddle_text,
    )

    word_differences = build_word_diff(
        tesseract_text,
        paddle_text,
    )

    similarity = calculate_similarity(
        tesseract_text,
        paddle_text,
    )

    return {
        "source_page": tesseract_page,
        "preprocessing_variant": tesseract_variant,
        "comparison": {
            "tesseract_file": str(tesseract_path),
            "paddleocr_file": str(paddleocr_path),
            "tesseract_text": tesseract_text,
            "paddleocr_text": paddle_text,
            "similarity": similarity,
            "character_differences": character_differences,
            "word_differences": word_differences,
            "difference_counts": {
                "character": len(character_differences),
                "word": len(word_differences),
            },
        },
    }


def compare_ocr_directories(
    tesseract_dir: Path,
    paddleocr_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """
    Compare matching Tesseract and PaddleOCR outputs.

    Args:
        tesseract_dir: Directory containing Tesseract JSON files.
        paddleocr_dir: Directory containing PaddleOCR JSON files.
        output_dir: Directory for structured comparison outputs.

    Returns:
        Summary counts for matched, failed, and missing pairs.
    """
    if not tesseract_dir.is_dir():
        raise FileNotFoundError(
            f"Tesseract directory not found: {tesseract_dir}"
        )

    if not paddleocr_dir.is_dir():
        raise FileNotFoundError(
            f"PaddleOCR directory not found: {paddleocr_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tesseract_files = {
        path.name: path
        for path in tesseract_dir.glob("*.json")
    }

    paddleocr_files = {
        path.name: path
        for path in paddleocr_dir.glob("*.json")
    }

    all_filenames = sorted(
        set(tesseract_files) | set(paddleocr_files)
    )

    matched = 0
    failed = 0
    missing = 0

    for filename in all_filenames:
        tesseract_path = tesseract_files.get(filename)
        paddleocr_path = paddleocr_files.get(filename)

        if tesseract_path is None or paddleocr_path is None:
            missing += 1

            logger.warning(
                "Missing OCR pair for filename: {}",
                filename,
            )
            continue

        try:
            comparison = compare_ocr_outputs(
                tesseract_path,
                paddleocr_path,
            )

            output_path = output_dir / filename

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    comparison,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            matched += 1

            logger.info(
                "OCR comparison written: {}",
                output_path,
            )

        except (
            OCRComparisonError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as exc:
            failed += 1

            logger.error(
                "OCR comparison failed for {}: {}",
                filename,
                exc,
            )

    summary = {
        "matched": matched,
        "failed": failed,
        "missing": missing,
        "total_pairs_considered": len(all_filenames),
    }

    logger.info(
        "OCR comparison completed: matched={}, failed={}, missing={}",
        matched,
        failed,
        missing,
    )

    return summary