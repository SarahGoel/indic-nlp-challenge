"""Unicode-aware tokenization for Hindi and Sanskrit manuscript text."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal

from loguru import logger

from indicnlp_pipeline.segmentation.sentence_segmenter import TextSegment


TokenType = Literal["word", "number", "punctuation", "symbol"]


@dataclass(frozen=True, slots=True)
class Token:
    """A token linked to its source page and segment.

    Character offsets use Python string indices and follow the half-open
    convention: start_char is inclusive and end_char is exclusive.
    """

    page_id: str
    segment_id: str
    token_index: int
    text: str
    token_type: TokenType
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation of this token."""
        return asdict(self)


def _character_kind(character: str) -> TokenType | None:
    """Classify a Unicode character for tokenization."""
    category = unicodedata.category(character)

    if category.startswith("L"):
        return "word"

    if category == "Nd":
        return "number"

    if category.startswith("P"):
        return "punctuation"

    if category.startswith("S"):
        return "symbol"

    return None


def tokenize_text(
    text: str,
    page_id: str,
    segment_id: str,
) -> list[Token]:
    """Tokenize text while preserving Unicode characters and source offsets.

    Letters and their combining marks remain together. Consecutive decimal
    digits form one number token. Punctuation and symbols are emitted as
    separate tokens. Whitespace separates tokens but is not emitted.

    The input text is not normalized or modified.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not isinstance(page_id, str):
        raise TypeError("page_id must be a string")

    if not page_id.strip():
        raise ValueError("page_id must not be empty")

    if not isinstance(segment_id, str):
        raise TypeError("segment_id must be a string")

    if not segment_id.strip():
        raise ValueError("segment_id must not be empty")

    tokens: list[Token] = []
    buffer: list[str] = []
    active_kind: TokenType | None = None
    token_start = 0

    def emit(end_char: int) -> None:
        """Emit the current buffer with its original character offsets."""
        nonlocal active_kind, token_start

        if not buffer or active_kind is None:
            buffer.clear()
            active_kind = None
            return

        value = "".join(buffer)
        tokens.append(
            Token(
                page_id=page_id,
                segment_id=segment_id,
                token_index=len(tokens),
                text=value,
                token_type=active_kind,
                start_char=token_start,
                end_char=end_char,
            )
        )
        buffer.clear()
        active_kind = None

    for index, character in enumerate(text):
        category = unicodedata.category(character)

        # Whitespace separates tokens but is not part of a token.
        if character.isspace():
            emit(index)
            continue

        # Combining marks, including Devanagari matras, nukta, and virama,
        # belong to the preceding word instead of becoming separate tokens.
        if category.startswith("M"):
            if active_kind == "word":
                buffer.append(character)
            else:
                emit(index)
                buffer.append(character)
                active_kind = "symbol"
                token_start = index
                emit(index + 1)
            continue

        # Preserve join controls within a word, including conjunct spellings
        # that explicitly contain a zero-width joiner or non-joiner.
        if character in ("\u200c", "\u200d") and active_kind == "word":
            buffer.append(character)
            continue

        kind = _character_kind(character)

        if kind is None:
            emit(index)
            continue

        # Adjacent letters form words; adjacent decimal digits form numbers.
        if kind in ("word", "number") and kind == active_kind:
            buffer.append(character)
            continue

        emit(index)

        if kind in ("punctuation", "symbol"):
            token_start = index
            buffer.append(character)
            active_kind = kind
            emit(index + 1)
        else:
            token_start = index
            buffer.append(character)
            active_kind = kind

    emit(len(text))

    logger.debug(
        "Tokenized page_id={} segment_id={} into {} tokens",
        page_id,
        segment_id,
        len(tokens),
    )
    return tokens


def tokenize_segment(segment: TextSegment) -> list[Token]:
    """Tokenize an existing segment without losing its source identifiers."""
    if not isinstance(segment, TextSegment):
        raise TypeError("segment must be a TextSegment instance")

    return tokenize_text(
        text=segment.text,
        page_id=segment.page_id,
        segment_id=segment.segment_id,
    )


def tokens_to_dicts(tokens: list[Token]) -> list[dict[str, str | int]]:
    """Convert tokens to dictionaries suitable for JSONL serialization."""
    return [token.to_dict() for token in tokens]
