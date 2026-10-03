"""Inspect raw manuscript scans and generate image metadata."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from indicnlp_pipeline.utils.image_metadata import (  # noqa: E402
    collect_image_metadata,
    save_metadata,
)


def main() -> None:
    """Inspect all raw manuscript scans."""
    raw_scans_directory = PROJECT_ROOT / "data" / "raw_scans"
    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "metrics"
        / "image_metadata.csv"
    )

    logger.info("Starting raw scan metadata inspection")
    logger.info("Input directory: {}", raw_scans_directory)

    metadata = collect_image_metadata(raw_scans_directory)

    save_metadata(metadata, output_path)

    logger.info(
        "Inspected {} manuscript images",
        len(metadata),
    )


if __name__ == "__main__":
    main()