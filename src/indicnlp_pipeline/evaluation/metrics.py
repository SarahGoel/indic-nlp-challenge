
"""Evaluation metrics for OCR and machine translation."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein
from jiwer import wer as jiwer_wer
from loguru import logger


def _validate_text_inputs(reference: str, hypothesis: str) -> None:
    """Validate reference and hypothesis inputs.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.

    Raises:
        TypeError: If either input is not a string.
    """
    if not isinstance(reference, str):
        raise TypeError("reference must be a string.")

    if not isinstance(hypothesis, str):
        raise TypeError("hypothesis must be a string.")


def _validate_reference(reference: str, hypothesis: str, metric: str) -> None:
    """Reject a non-empty hypothesis when the reference is empty.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted text.
        metric: Name of the evaluation metric.

    Raises:
        ValueError: If the reference is empty and the hypothesis is not.
    """
    if not reference and hypothesis:
        raise ValueError(
            f"{metric} is undefined for a non-empty hypothesis "
            "with an empty reference."
        )


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate character error rate using Levenshtein distance.

    CER is the character-level edit distance divided by the number
    of characters in the reference.

    Whitespace and Unicode normalization are not modified implicitly.

    Args:
        reference: Human-corrected reference text.
        hypothesis: OCR-predicted text.

    Returns:
        Character error rate as a float.

    Raises:
        TypeError: If either input is not a string.
        ValueError: If the reference is empty and the hypothesis is not.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "CER")

    if not reference:
        return 0.0

    edit_distance = Levenshtein.distance(reference, hypothesis)
    return edit_distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate word error rate using jiwer.

    WER is the number of word substitutions, deletions, and insertions
    divided by the number of words in the reference.

    Words are tokenized by whitespace using jiwer's default
    transformation. Inputs are not implicitly normalized or stripped
    by this function.

    Args:
        reference: Human-corrected reference text.
        hypothesis: OCR-predicted text.

    Returns:
        Word error rate as a float. The value may exceed 1.0.

    Raises:
        TypeError: If either input is not a string.
        ValueError: If the reference is empty and the hypothesis is not.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "WER")

    if not reference:
        return 0.0

    score = float(jiwer_wer(reference, hypothesis))
    logger.debug("Calculated WER: {:.6f}", score)
    return score
