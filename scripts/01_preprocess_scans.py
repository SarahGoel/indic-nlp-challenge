from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = REPOSITORY_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


import cv2
from loguru import logger

from indicnlp_pipeline.preprocessing.image_prep import (
    deskew,
    denoise_gaussian,
    enhance_contrast,
    grayscale,
    load_image,
    threshold_adaptive,
    threshold_otsu,
)
from indicnlp_pipeline.utils.config_loader import (
    load_config,
    resolve_config_paths,
)
from indicnlp_pipeline.utils.logger import configure_logger


def build_page_filename(page_number: int) -> str:
    """Build the source-compatible filename for a manuscript page."""
    return f"BRIGU_V5_P{page_number:04d}.png"


def save_image(image, output_path: Path) -> None:
    """Save an image and raise an error if OpenCV cannot write it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise IOError(f"OpenCV failed to write image: {output_path}")


def process_page(
    input_path: Path,
    output_paths: dict[str, Path],
    adaptive_block_size: int,
    adaptive_constant: float,
    adaptive_method: str,
) -> None:
    """Run all preprocessing stages for one manuscript page."""
    logger.info("Processing page: {}", input_path.name)

    image = load_image(input_path)

    gray = grayscale(image)
    save_image(gray, output_paths["grayscale"])

    denoised = denoise_gaussian(
        gray,
        kernel_size=5,
        sigma=0,
    )
    save_image(denoised, output_paths["denoised"])

    contrast = enhance_contrast(
        denoised,
        clip_limit=2.0,
        tile_grid_size=8,
    )
    save_image(contrast, output_paths["contrast"])

    otsu = threshold_otsu(contrast)
    save_image(otsu, output_paths["otsu"])

    adaptive = threshold_adaptive(
        contrast,
        block_size=adaptive_block_size,
        constant=adaptive_constant,
        method=adaptive_method,
    )
    save_image(adaptive, output_paths["adaptive_threshold"])

    deskewed = deskew(contrast)
    save_image(deskewed, output_paths["deskewed"])

    logger.info(
        "Successfully processed page: {}",
        input_path.name,
    )


def main() -> None:
    """Run preprocessing on the configured experimental page subset."""
    config = resolve_config_paths(load_config())

    configure_logger(
        log_directory=config["paths"]["logs"],
    )

    paths = config["paths"]
    preprocessing_config = config["preprocessing"]

    raw_scans = paths["raw_scans"]
    preprocessed = paths["preprocessed"]

    start_page = int(
        preprocessing_config["experimental_start_page"]
    )
    end_page = int(
        preprocessing_config["experimental_end_page"]
    )

    adaptive_config = preprocessing_config["threshold"]

    adaptive_block_size = int(
        adaptive_config["block_size"]
    )
    adaptive_constant = float(
        adaptive_config["constant"]
    )
    adaptive_method = str(
        adaptive_config["adaptive_method"]
    )

    output_directories = {
        "grayscale": preprocessed / "grayscale",
        "denoised": preprocessed / "denoised",
        "contrast": preprocessed / "contrast",
        "otsu": preprocessed / "otsu",
        "adaptive_threshold": preprocessed / "adaptive_threshold",
        "deskewed": preprocessed / "deskewed",
    }

    for directory in output_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting experimental preprocessing: pages {}-{}",
        start_page,
        end_page,
    )

    successful_pages = 0
    failed_pages = 0

    for page_number in range(start_page, end_page + 1):
        filename = build_page_filename(page_number)
        input_path = raw_scans / filename

        output_paths = {
            stage: directory / filename
            for stage, directory in output_directories.items()
        }

        try:
            process_page(
                input_path=input_path,
                output_paths=output_paths,
                adaptive_block_size=adaptive_block_size,
                adaptive_constant=adaptive_constant,
                adaptive_method=adaptive_method,
            )
            successful_pages += 1

        except Exception as exc:
            failed_pages += 1
            logger.exception(
                "Failed to process page {}: {}",
                filename,
                exc,
            )

    logger.info(
        "Experimental preprocessing completed: successful={}, failed={}",
        successful_pages,
        failed_pages,
    )


if __name__ == "__main__":
    main()