
"""Evaluation metrics for OCR and machine translation."""

from __future__ import annotations

import numpy as np
import sacrebleu
from jiwer import wer as jiwer_wer
from loguru import logger
from rapidfuzz.distance import Levenshtein


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


def calculate_chrf(
    reference: str,
    hypothesis: str,
) -> float:
    """Calculate sentence-level chrF on a 0–100 scale.

    Uses SacreBLEU's character n-gram F-score with the standard
    chrF configuration: character order 6, word order 0, and beta 2.

    Args:
        reference: Human-produced reference text.
        hypothesis: Machine-produced hypothesis text.

    Returns:
        A chrF score between 0.0 and 100.0.

    Raises:
        TypeError: If either input is not a string.
        ValueError: If the reference is empty and the hypothesis
            is non-empty.

    Notes:
        Both empty strings return 0.0, consistent with the existing
        BLEU function. Inputs are not implicitly normalized.
        Word order 0 means this is chrF, not chrF++.
    """
    _validate_text_inputs(reference, hypothesis)
    _validate_reference(reference, hypothesis, "chrF")

    if not reference and not hypothesis:
        return 0.0

    metric = sacrebleu.metrics.CHRF(
        char_order=6,
        word_order=0,
        beta=2,
        lowercase=False,
        whitespace=False,
    )

    result = metric.sentence_score(
        hypothesis,
        [reference],
    )

    score = float(result.score)
    logger.debug("Calculated sentence chrF: {:.6f}", score)
    return score


_SEMANTIC_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_SEMANTIC_MODEL = None


def _get_semantic_model():
    """Load and cache the multilingual sentence embedding model lazily."""
    global _SEMANTIC_MODEL

    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading multilingual semantic similarity model: {}",
            _SEMANTIC_MODEL_NAME,
        )
        _SEMANTIC_MODEL = SentenceTransformer(
            _SEMANTIC_MODEL_NAME,
            local_files_only=True,
        )

    return _SEMANTIC_MODEL


def calculate_semantic_similarity(
    reference: str,
    hypothesis: str,
) -> float:
    """Calculate cosine similarity between multilingual sentence embeddings.

    Args:
        reference: Reference text, such as a human translation.
        hypothesis: Text being evaluated against the reference.

    Returns:
        Cosine similarity as a float, normally between -1.0 and 1.0.
        Both empty strings return 0.0.

    Raises:
        TypeError: If either input is not a string.
        ValueError: If exactly one input is empty or an embedding is invalid.

    Notes:
        The multilingual model is loaded lazily and reused between calls.
        Model loading uses local files only and does not download weights.
        Inputs are not implicitly normalized or stripped.
    """
    _validate_text_inputs(reference, hypothesis)

    if not reference and not hypothesis:
        return 0.0

    if not reference:
        raise ValueError(
            "Semantic similarity is undefined for a non-empty hypothesis "
            "with an empty reference."
        )

    if not hypothesis:
        raise ValueError(
            "Semantic similarity is undefined for a non-empty reference "
            "with an empty hypothesis."
        )

    model = _get_semantic_model()
    embeddings = np.asarray(
        model.encode(
            [reference, hypothesis],
            convert_to_numpy=True,
        ),
        dtype=np.float64,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != 2:
        raise ValueError(
            "The embedding model must return exactly two embedding vectors."
        )

    reference_embedding, hypothesis_embedding = embeddings

    reference_norm = float(np.linalg.norm(reference_embedding))
    hypothesis_norm = float(np.linalg.norm(hypothesis_embedding))

    if reference_norm == 0.0 or hypothesis_norm == 0.0:
        raise ValueError(
            "Semantic similarity is undefined for a zero-norm embedding."
        )

    score = float(
        np.dot(reference_embedding, hypothesis_embedding)
        / (reference_norm * hypothesis_norm)
    )
    logger.debug("Calculated semantic similarity: {:.6f}", score)
    return score
