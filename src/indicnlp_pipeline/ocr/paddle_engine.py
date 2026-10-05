from pathlib import Path
from typing import Any

from loguru import logger
from paddleocr import PaddleOCR


class PaddleOCRError(RuntimeError):
    """Raised when PaddleOCR processing fails."""


def create_paddleocr_engine() -> PaddleOCR:
    """
    Create a PaddleOCR engine configured for Devanagari OCR.

    Returns:
        Initialized PaddleOCR engine using the PP-OCRv5 mobile
        detector and Devanagari recognition model.

    Raises:
        PaddleOCRError: If PaddleOCR initialization fails.
    """
    logger.info("Initializing PaddleOCR Devanagari engine.")

    try:
        engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="devanagari_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:
        logger.exception("Failed to initialize PaddleOCR.")
        raise PaddleOCRError(
            "Failed to initialize PaddleOCR Devanagari engine."
        ) from exc

    logger.info("PaddleOCR Devanagari engine initialized successfully.")

    return engine


def run_paddleocr(
    image_path: Path,
    engine: PaddleOCR,
) -> dict[str, Any]:
    """
    Run PaddleOCR on a single image.

    Args:
        image_path: Path to the input image.
        engine: Initialized PaddleOCR engine.

    Returns:
        Structured OCR result containing image identity,
        recognized text, confidence scores, and bounding boxes.

    Raises:
        TypeError: If image_path is not a pathlib.Path.
        FileNotFoundError: If the input image does not exist.
        PaddleOCRError: If OCR processing fails.
    """
    if not isinstance(image_path, Path):
        raise TypeError("image_path must be a pathlib.Path instance.")

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    logger.info("Running PaddleOCR: image={}", image_path)

    try:
        results = engine.predict(str(image_path))
    except Exception as exc:
        logger.exception("PaddleOCR failed for {}", image_path)
        raise PaddleOCRError(
            f"PaddleOCR failed for {image_path}"
        ) from exc

    if not results:
        logger.warning("PaddleOCR returned no results for {}", image_path)

        return {
            "image_path": str(image_path),
            "text": [],
            "confidence": [],
            "bounding_boxes": [],
        }

    result = results[0]

    try:
        recognized_text = list(result["rec_texts"])
        confidence_scores = [
            float(score) for score in result["rec_scores"]
        ]

        bounding_boxes = [
            box.tolist() for box in result["rec_polys"]
        ]

    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.exception(
            "Unexpected PaddleOCR result structure for {}",
            image_path,
        )
        raise PaddleOCRError(
            f"Unable to parse PaddleOCR result for {image_path}"
        ) from exc

    if not (
        len(recognized_text)
        == len(confidence_scores)
        == len(bounding_boxes)
    ):
        raise PaddleOCRError(
            "PaddleOCR result fields have inconsistent lengths: "
            f"text={len(recognized_text)}, "
            f"confidence={len(confidence_scores)}, "
            f"bounding_boxes={len(bounding_boxes)}"
        )

    output: dict[str, Any] = {
        "image_path": str(image_path),
        "text": recognized_text,
        "confidence": confidence_scores,
        "bounding_boxes": bounding_boxes,
    }

    logger.info(
        "PaddleOCR completed successfully: image={}, detections={}",
        image_path,
        len(recognized_text),
    )

    return output