
import cv2
import numpy as np

from indicnlp_pipeline.preprocessing.image_prep import deskew


def _make_text_line_image(
    angle_degrees: float = 0.0,
    width: int = 1200,
    height: int = 500,
) -> np.ndarray:
    """Create a synthetic grayscale page with long horizontal text-like lines."""
    image = np.full((height, width), 255, dtype=np.uint8)

    for y in range(100, height - 50, 60):
        cv2.line(
            image,
            (80, y),
            (width - 80, y),
            color=0,
            thickness=3,
        )

    if angle_degrees == 0.0:
        return image

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def test_deskew_rejects_non_grayscale_input() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        deskew(image)
    except ValueError as exc:
        assert "2D" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a 3D image")


def test_deskew_returns_copy_when_no_reliable_lines_exist() -> None:
    image = np.full((300, 400), 255, dtype=np.uint8)

    result = deskew(image)

    assert np.array_equal(result, image)
    assert result is not image


def test_deskew_corrects_synthetic_skew() -> None:
    # The image is rotated counterclockwise by 3 degrees.
    image = _make_text_line_image(angle_degrees=3.0)

    result = deskew(
        image,
        min_line_length_ratio=0.10,
        angle_tolerance=0.5,
    )

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.array_equal(result, image)
def test_deskew_default_detects_skew_in_short_text_lines() -> None:
    """Default deskew settings should detect lines shorter than 25% of width."""
    width, height = 1200, 600
    image = np.full((height, width), 255, dtype=np.uint8)

    for y in range(80, height - 40, 60):
        cv2.line(
            image,
            (100, y),
            (350, y),
            color=0,
            thickness=3,
        )

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        3.0,
        1.0,
    )
    skewed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    result = deskew(skewed)

    assert not np.array_equal(result, skewed)
