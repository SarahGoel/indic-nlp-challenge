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


def threshold_otsu(image: np.ndarray) -> np.ndarray:
    """Apply Otsu's global thresholding to a grayscale image.

    Args:
        image: Input grayscale image as a NumPy array.

    Returns:
        Binary image produced using Otsu's thresholding.

    Raises:
        ValueError: If the input image is not a 2D grayscale image.
    """
    if image.ndim != 2:
        raise ValueError(
            "threshold_otsu expects a grayscale 2D image."
        )

    _, result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    logger.debug(
        "Applied Otsu thresholding: shape={}",
        image.shape,
    )

    return result


def threshold_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    constant: float = 2.0,
    method: str = "gaussian",
) -> np.ndarray:
    """Apply adaptive thresholding to a grayscale image.

    Args:
        image: Input grayscale image as a NumPy array.
        block_size: Size of the local neighbourhood. Must be an odd integer
            greater than one.
        constant: Constant subtracted from the local threshold.
        method: Adaptive thresholding method. Supported values are
            'gaussian' and 'mean'.

    Returns:
        Binary image produced using adaptive thresholding.

    Raises:
        ValueError: If the input image or parameters are invalid.
    """
    if image.ndim != 2:
        raise ValueError(
            "threshold_adaptive expects a grayscale 2D image."
        )

    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            "block_size must be an odd integer greater than one."
        )

    method = method.lower()

    if method == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError(
            "method must be either 'gaussian' or 'mean'."
        )

    result = cv2.adaptiveThreshold(
        image,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )

    logger.debug(
        "Applied adaptive thresholding: method={}, block_size={}, constant={}",
        method,
        block_size,
        constant,
    )

    return result


def deskew(
    image: np.ndarray,
    max_skew_angle: float = 15.0,
    min_line_length_ratio: float = 0.25,
    angle_tolerance: float = 1.0,
) -> np.ndarray:
    """Correct small rotational skew using Hough-line detection.

    Args:
        image: Input grayscale or binary image as a NumPy array.
        max_skew_angle: Maximum absolute skew angle considered reliable.
        min_line_length_ratio: Minimum detected line length as a fraction
            of the image width.
        angle_tolerance: Tolerance in degrees for grouping similar angles.

    Returns:
        Deskewed image. If no reliable skew can be detected, a copy of the
        original image is returned unchanged.

    Raises:
        ValueError: If the input image is not 2D or parameters are invalid.
    """
    if image.ndim != 2:
        raise ValueError(
            "deskew expects a grayscale or binary 2D image."
        )

    if max_skew_angle <= 0:
        raise ValueError(
            "max_skew_angle must be greater than zero."
        )

    if not 0 < min_line_length_ratio <= 1:
        raise ValueError(
            "min_line_length_ratio must be greater than zero and at most one."
        )

    if angle_tolerance <= 0:
        raise ValueError(
            "angle_tolerance must be greater than zero."
        )

    height, width = image.shape

    edges = cv2.Canny(
        image,
        threshold1=50,
        threshold2=150,
        apertureSize=3,
    )

    min_line_length = max(
        1,
        int(width * min_line_length_ratio),
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(50, min_line_length // 2),
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        logger.warning(
            "Deskew skipped: no reliable Hough lines detected."
        )
        return image.copy()

    candidate_angles: list[float] = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))

        while angle <= -90:
            angle += 180

        while angle > 90:
            angle -= 180

        if abs(angle) <= max_skew_angle:
            candidate_angles.append(angle)

    if not candidate_angles:
        logger.warning(
            "Deskew skipped: no reliable near-horizontal lines detected."
        )
        return image.copy()

    angle_array = np.asarray(candidate_angles, dtype=np.float32)

    median_angle = float(np.median(angle_array))

    if abs(median_angle) < angle_tolerance:
        logger.info(
            "Deskew skipped: detected skew {:.3f}° is within tolerance.",
            median_angle,
        )
        return image.copy()

    center = (width / 2.0, height / 2.0)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        median_angle,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.info(
        "Deskew applied: detected angle={:.3f}°, lines={}",
        median_angle,
        len(candidate_angles),
    )

    return rotated


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