from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or BGRA image to grayscale.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Grayscale image as a NumPy array.

    Raises:
        ValueError: If the input image has an unsupported shape.
    """
    if image.ndim == 2:
        return image.copy()

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image array, got shape {image.shape}."
        )

    channels = image.shape[2]

    if channels == 3:
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channels == 4:
        result = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(
            f"Unsupported number of image channels: {channels}."
        )

    logger.debug(
        "Converted image to grayscale: shape {} -> {}",
        image.shape,
        result.shape,
    )

    return result


def denoise_gaussian(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian blur for noise reduction.

    Args:
        image: Input image as a NumPy array.
        kernel_size: Size of the Gaussian kernel. Must be a positive odd integer.
        sigma: Gaussian kernel standard deviation. Zero lets OpenCV calculate it.

    Returns:
        Denoised image with the same dimensions as the input.

    Raises:
        ValueError: If kernel_size is not a positive odd integer.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    result = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma,
    )

    logger.debug(
        "Applied Gaussian denoising: kernel_size={}, sigma={}",
        kernel_size,
        sigma,
    )

    return result


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Enhance local image contrast using CLAHE.

    Args:
        image: Input grayscale image as a NumPy array.
        clip_limit: CLAHE contrast clipping limit.
        tile_grid_size: Number of tiles along each image dimension.

    Returns:
        Contrast-enhanced image with the same dimensions as the input.

    Raises:
        ValueError: If the input is not a grayscale image or parameters are invalid.
    """
    if image.ndim != 2:
        raise ValueError(
            "enhance_contrast expects a grayscale 2D image."
        )

    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than zero.")

    if tile_grid_size <= 0:
        raise ValueError("tile_grid_size must be greater than zero.")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )

    result = clahe.apply(image)

    logger.debug(
        "Enhanced contrast using CLAHE: clip_limit={}, tile_grid_size={}",
        clip_limit,
        tile_grid_size,
    )

    return result


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        image_path: Path to the input image.

    Returns:
        Loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    image_path = image_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    logger.debug(
        "Loaded image: {} | shape={}",
        image_path,
        image.shape,
    )

    return image