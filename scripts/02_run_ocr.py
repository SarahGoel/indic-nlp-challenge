from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indicnlp_pipeline.ocr.paddle_engine import (  # noqa: E402
    PaddleOCRError,
    create_paddleocr_engine,
    run_paddleocr,
)
from src.indicnlp_pipeline.ocr.tesseract_engine import (  # noqa: E402
    TesseractOCRError,
    run_tesseract_ocr,
)
from src.indicnlp_pipeline.utils.config_loader import load_config  # noqa: E402


PREPROCESSING_VARIANTS = (
    "grayscale",
    "denoised",
    "contrast",
    "otsu",
    "adaptive_threshold",
    "deskewed",
)


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a structured OCR result as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def discover_preprocessed_images(
    project_root: Path,
) -> list[tuple[str, Path]]:
    """
    Discover experimental preprocessed images.

    Returns:
        List of (preprocessing_variant, image_path) tuples.
    """
    discovered: list[tuple[str, Path]] = []

    preprocessing_root = project_root / "data" / "preprocessed"

    for variant in PREPROCESSING_VARIANTS:
        variant_dir = preprocessing_root / variant

        if not variant_dir.is_dir():
            logger.warning(
                "Preprocessing directory does not exist: {}",
                variant_dir,
            )
            continue

        images = sorted(variant_dir.glob("*.png"))

        logger.info(
            "Discovered {} images for preprocessing variant '{}'.",
            len(images),
            variant,
        )

        for image_path in images:
            discovered.append((variant, image_path))

    return discovered


def build_output_path(
    output_root: Path,
    image_path: Path,
    preprocessing_variant: str,
) -> Path:
    """Build a deterministic JSON output path."""
    page_name = image_path.stem

    filename = (
        f"{page_name}__{preprocessing_variant}.json"
    )

    return output_root / filename


def run_tesseract_batch(
    images: list[tuple[str, Path]],
    output_root: Path,
) -> tuple[int, int]:
    """
    Run Tesseract across all discovered images.

    Returns:
        Tuple containing successful and failed counts.
    """
    successful = 0
    failed = 0

    for variant, image_path in images:
        logger.info(
            "Tesseract processing: variant={}, image={}",
            variant,
            image_path,
        )

        try:
            result = run_tesseract_ocr(
                image_path=image_path,
                psm=6,
                oem=3,
                language="san+hin",
            )

            output = {
                "source_page": image_path.stem,
                "preprocessing_variant": variant,
                "ocr_engine": "tesseract",
                "image_path": str(image_path),
                "language": result["language"],
                "psm": result["psm"],
                "oem": result["oem"],
                "text": result["text"],
            }

            output_path = build_output_path(
                output_root,
                image_path,
                variant,
            )

            write_json(output_path, output)

            successful += 1

            logger.info(
                "Tesseract output written: {}",
                output_path,
            )

        except (TesseractOCRError, FileNotFoundError, ValueError) as exc:
            failed += 1

            logger.error(
                "Tesseract failed: variant={}, image={}, error={}",
                variant,
                image_path,
                exc,
            )

    return successful, failed


def run_paddleocr_batch(
    images: list[tuple[str, Path]],
    output_root: Path,
) -> tuple[int, int]:
    """
    Run PaddleOCR across all discovered images.

    Returns:
        Tuple containing successful and failed counts.
    """
    successful = 0
    failed = 0

    try:
        engine = create_paddleocr_engine()
    except PaddleOCRError as exc:
        logger.error(
            "PaddleOCR engine initialization failed: {}",
            exc,
        )
        return 0, len(images)

    for variant, image_path in images:
        logger.info(
            "PaddleOCR processing: variant={}, image={}",
            variant,
            image_path,
        )

        try:
            result = run_paddleocr(
                image_path=image_path,
                engine=engine,
            )

            output = {
                "source_page": image_path.stem,
                "preprocessing_variant": variant,
                "ocr_engine": "paddleocr",
                "image_path": str(image_path),
                "text": result["text"],
                "confidence": result["confidence"],
                "bounding_boxes": result["bounding_boxes"],
            }

            output_path = build_output_path(
                output_root,
                image_path,
                variant,
            )

            write_json(output_path, output)

            successful += 1

            logger.info(
                "PaddleOCR output written: {}",
                output_path,
            )

        except (
            PaddleOCRError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            failed += 1

            logger.error(
                "PaddleOCR failed: variant={}, image={}, error={}",
                variant,
                image_path,
                exc,
            )

    return successful, failed


def main() -> int:
    """Run the dual-OCR experimental pipeline."""
    logger.info("Starting Step 18 dual-OCR driver.")

    config = load_config()

    logger.info(
        "Configuration loaded successfully: {}",
        config,
    )

    project_root = PROJECT_ROOT

    tesseract_output_root = (
        project_root / "data" / "ocr_raw" / "tesseract"
    )

    paddle_output_root = (
        project_root / "data" / "ocr_raw" / "paddleocr"
    )

    images = discover_preprocessed_images(project_root)

    if not images:
        logger.error(
            "No preprocessed experimental images were discovered."
        )
        return 1

    logger.info(
        "Total preprocessing/image combinations discovered: {}",
        len(images),
    )

    tesseract_success, tesseract_failed = run_tesseract_batch(
        images,
        tesseract_output_root,
    )

    paddle_success, paddle_failed = run_paddleocr_batch(
        images,
        paddle_output_root,
    )

    logger.info(
        "Tesseract summary: successful={}, failed={}",
        tesseract_success,
        tesseract_failed,
    )

    logger.info(
        "PaddleOCR summary: successful={}, failed={}",
        paddle_success,
        paddle_failed,
    )

    total_failures = tesseract_failed + paddle_failed

    if total_failures:
        logger.warning(
            "Dual-OCR run completed with {} failures.",
            total_failures,
        )
        return 1

    logger.info("Step 18 dual-OCR driver completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())