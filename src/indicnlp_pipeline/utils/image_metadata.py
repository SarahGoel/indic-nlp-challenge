"""Image metadata extraction utilities for manuscript scans."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from loguru import logger


SOURCE_RENDER_DPI = 300


def extract_image_metadata(image_path: Path) -> dict[str, object]:
    """Extract metadata from a single manuscript image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]

    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    file_size = image_path.stat().st_size

    with Image.open(image_path) as pil_image:
        dpi = pil_image.info.get("dpi")

    embedded_dpi_x = None
    embedded_dpi_y = None

    if dpi is not None and len(dpi) >= 2:
        embedded_dpi_x = dpi[0]
        embedded_dpi_y = dpi[1]

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": file_size,
        "embedded_dpi_x": embedded_dpi_x,
        "embedded_dpi_y": embedded_dpi_y,
        "estimated_dpi": SOURCE_RENDER_DPI,
    }


def collect_image_metadata(
    input_directory: Path,
) -> pd.DataFrame:
    """Collect metadata for all manuscript PNG scans."""
    image_paths = sorted(input_directory.glob("BRIGU_V5_P*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No manuscript PNG scans found in {input_directory}"
        )

    records: list[dict[str, object]] = []

    for image_path in image_paths:
        try:
            records.append(extract_image_metadata(image_path))
        except (OSError, ValueError) as exc:
            logger.error(
                "Failed to inspect {}: {}",
                image_path.name,
                exc,
            )
            raise

    return pd.DataFrame(records)


def save_metadata(
    metadata: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save image metadata as a UTF-8 CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Image metadata written to {}",
        output_path,
    )