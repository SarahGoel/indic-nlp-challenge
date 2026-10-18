"""Evaluation metrics for OCR and machine translation."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein
from jiwer import wer as jiwer_wer
from loguru import logger
import sacrebleu


def _validate_text_inputs(reference: str, hypothesis: str) -> None:
    """Validate reference and hypothesis inputs.

    Raises:
        TypeError: If either input is not a string.
    """
    if not isinstance(reference, str):
        raise TypeError("reference must be a string.")

    if not isinstance(hypothesis, str):
        raise TypeError("hypothesis must be a string.")


def _validate_reference(
    reference: str,
    hypothesis: str,
    metric: str,
) -> None:
    """Reject a non-empty hypothesis when the reference is empty.

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

    Whitespace and Unicode normalization are not changed implicitly.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "CER")

    if not reference:
        return 0.0

    score = Levenshtein.distance(reference, hypothesis) / len(reference)
    logger.debug("Calculated CER: {:.6f}", score)
    return float(score)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate word error rate using jiwer.

    WER is the number of word substitutions, deletions, and insertions
    divided by the number of words in the reference.

    Words are tokenized using jiwer's default transformation. Inputs
    are not implicitly normalized or stripped by this function.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "WER")

    if not reference:
        return 0.0

    score = float(jiwer_wer(reference, hypothesis))
    logger.debug("Calculated WER: {:.6f}", score)
    return score


def calculate_bleu(
    reference: str,
    hypothesis: str,
    *,
    tokenize: str = "13a",
) -> float:
    """Calculate sentence-level BLEU on a 0–100 scale.

    Args:
        reference: Human-produced reference translation.
        hypothesis: Machine-produced translation.
        tokenize: SacreBLEU tokenizer. The default, ``13a``, is a
            general-purpose tokenizer; select a suitable tokenizer
            explicitly when evaluating other scripts.

    Returns:
        BLEU score between 0.0 and 100.0.

    Raises:
        TypeError: If either text input is not a string.
        ValueError: If the reference is empty and the hypothesis is
            non-empty, or if the tokenizer is invalid.

    Notes:
        BLEU is a corpus-oriented metric. Sentence-level BLEU can be
        unstable for short translations. For final comparisons, use
        corpus BLEU over a set of aligned reference/hypothesis segments.
        A score of zero is returned when both texts are empty.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "BLEU")

    if not reference and not hypothesis:
        return 0.0

    try:
        result = sacrebleu.sentence_bleu(
            hypothesis,
            [reference],
            tokenize=tokenize,
        )
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid BLEU tokenizer {tokenize!r}: {exc}") from exc

    score = float(result.score)
    logger.debug("Calculated sentence BLEU: {:.6f}", score)
    return score
