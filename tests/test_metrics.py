
"""Unit tests for OCR and translation evaluation metrics."""

import pytest

from indicnlp_pipeline.evaluation.metrics import (
    calculate_cer,
    calculate_wer,
)


# ---------------------------------------------------------------------
# Character Error Rate (CER)
# ---------------------------------------------------------------------


def test_cer_exact_match() -> None:
    assert calculate_cer("hello", "hello") == 0.0


def test_cer_substitution() -> None:
    assert calculate_cer("cat", "cut") == pytest.approx(1 / 3)


def test_cer_insertion() -> None:
    assert calculate_cer("abc", "abxc") == pytest.approx(1 / 3)


def test_cer_deletion() -> None:
    assert calculate_cer("abcd", "acd") == pytest.approx(1 / 4)


def test_cer_multiple_edits() -> None:
    assert calculate_cer("kitten", "sitting") == pytest.approx(3 / 6)


def test_cer_devanagari_text() -> None:
    reference = "\u0930\u093e\u092e"
    hypothesis = "\u0930\u092e"
    assert calculate_cer(reference, hypothesis) == pytest.approx(1 / 3)


def test_cer_devanagari_exact_match() -> None:
    text = (
        "\u0936\u094d\u0930\u0940 "
        "\u0917\u0923\u0947\u0936\u093e\u092f "
        "\u0928\u092e\u0903\u0964"
    )
    assert calculate_cer(text, text) == 0.0


def test_cer_empty_strings() -> None:
    assert calculate_cer("", "") == 0.0


def test_cer_empty_reference_nonempty_hypothesis() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        calculate_cer("", "prediction")


def test_cer_does_not_strip_whitespace() -> None:
    assert calculate_cer("a", "a ") == pytest.approx(1.0)


def test_cer_does_not_normalize_unicode_implicitly() -> None:
    precomposed = "\u00e9"
    decomposed = "e\u0301"
    assert calculate_cer(precomposed, decomposed) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        (None, "text"),
        ("text", None),
        (123, "text"),
        ("text", 123),
    ],
)
def test_cer_rejects_non_string_inputs(
    reference: object,
    hypothesis: object,
) -> None:
    with pytest.raises(TypeError):
        calculate_cer(reference, hypothesis)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Word Error Rate (WER)
# ---------------------------------------------------------------------


def test_wer_exact_match() -> None:
    assert calculate_wer("the cat sat", "the cat sat") == 0.0


def test_wer_single_substitution() -> None:
    assert calculate_wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_single_insertion() -> None:
    assert calculate_wer("the cat sat", "the small cat sat") == pytest.approx(
        1 / 3
    )


def test_wer_single_deletion() -> None:
    assert calculate_wer("the cat sat", "the sat") == pytest.approx(1 / 3)


def test_wer_multiple_edits() -> None:
    assert calculate_wer("a b c d", "a x d") == pytest.approx(2 / 4)


def test_wer_devanagari_text() -> None:
    reference = "\u0930\u093e\u092e \u0918\u0930 \u0917\u092f\u093e"
    hypothesis = "\u0930\u093e\u092e \u0917\u092f\u093e"
    assert calculate_wer(reference, hypothesis) == pytest.approx(1 / 3)


def test_wer_devanagari_exact_match() -> None:
    text = (
        "\u0936\u094d\u0930\u0940 "
        "\u0917\u0923\u0947\u0936\u093e\u092f "
        "\u0928\u092e\u0903\u0964"
    )
    assert calculate_wer(text, text) == 0.0


def test_wer_empty_strings() -> None:
    assert calculate_wer("", "") == 0.0


def test_wer_empty_reference_nonempty_hypothesis() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        calculate_wer("", "prediction")


def test_wer_whitespace_tokenization() -> None:
    # WER uses word tokens, so a trailing space does not add a word.
    assert calculate_wer("a", "a ") == 0.0


def test_wer_does_not_normalize_unicode_implicitly() -> None:
    precomposed = "\u00e9"
    decomposed = "e\u0301"
    assert calculate_wer(precomposed, decomposed) == 1.0


def test_wer_can_exceed_one() -> None:
    assert calculate_wer("a", "a b c") == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [
        (None, "text"),
        ("text", None),
        (123, "text"),
        ("text", 123),
    ],
)
def test_wer_rejects_non_string_inputs(
    reference: object,
    hypothesis: object,
) -> None:
    with pytest.raises(TypeError):
        calculate_wer(reference, hypothesis)  # type: ignore[arg-type]
