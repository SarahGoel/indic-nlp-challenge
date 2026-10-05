from pathlib import Path
from typing import Any

import pytesseract
from loguru import logger
from PIL import Image


class TesseractOCRError(RuntimeError):
    """Raised when Tesseract OCR processing fails."""


def run_tesseract_ocr(
    image_path: Path,
    psm: int = 6,
    oem: int = 3,
    language: str = "san+hin",
) -> dict[str, Any]:
    """
    Run Tesseract OCR on a single image.

    Args:
        image_path: Path to the input image.
        psm: Tesseract Page Segmentation Mode.
        oem: Tesseract OCR Engine Mode.
        language: Tesseract language configuration.

    Returns:
        Structured OCR result containing the input image path,
        OCR configuration, and recognized text.

    Raises:
        FileNotFoundError: If the input image does not exist.
        TesseractOCRError: If OCR processing fails.
        ValueError: If OCR parameters are invalid.
    """
    if not isinstance(image_path, Path):
        raise TypeError("image_path must be a pathlib.Path instance.")

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if psm < 0:
        raise ValueError("PSM must be a non-negative integer.")

    if oem < 0:
        raise ValueError("OEM must be a non-negative integer.")

    logger.info(
        "Running Tesseract OCR: image={}, language={}, psm={}, oem={}",
        image_path,
        language,
        psm,
        oem,
    )

    config = f"--psm {psm} --oem {oem}"

    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(
                image,
                lang=language,
                config=config,
            )

    except Exception as exc:
        logger.exception("Tesseract OCR failed for {}", image_path)
        raise TesseractOCRError(
            f"Tesseract OCR failed for {image_path}"
        ) from exc

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "language": language,
        "psm": psm,
        "oem": oem,
        "text": text,
    }

    logger.info(
        "Tesseract OCR completed successfully: image={}",
        image_path,
    )

    return result