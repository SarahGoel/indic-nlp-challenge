
"""Manuscript-aware segmentation for Hindi and Sanskrit OCR text."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from loguru import logger


BoundaryType = Literal[
    "single_danda",
    "double_danda",
    "ascii_single_bar",
    "ascii_double_bar",
    "entry_marker",
    "end_of_input",
]

_ENTRY_MARKER = re.compile(r"\(([0-9०-९]+)\)")
_TOKEN = re.compile(r"\([0-9०-९]+\)|॥|।|\|\||\|")

_BOUNDARY_TYPES: dict[str, BoundaryType] = {
    "।": "single_danda",
    "॥": "double_danda",
    "|": "ascii_single_bar",
    "||": "ascii_double_bar",
}


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A source-linked text segment produced by the segmenter."""

    page_id: str
    segment_id: str
    text: str
    boundary_after: BoundaryType
    entry_number: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def segment_text(text: str, page_id: str) -> list[TextSegment]:
    """Segment OCR text using manuscript punctuation and entry markers.

    Newlines alone do not force segmentation. Recognized delimiters remain
    attached to their segments, and incomplete trailing text is preserved.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(page_id, str):
        raise TypeError("page_id must be a string")
    if not page_id.strip():
        raise ValueError("page_id must not be empty")

    if not text.strip():
        logger.debug("No segmentable text for page_id={}", page_id)
        return []

    segments: list[TextSegment] = []
    buffer: list[str] = []
    entry_number: str | None = None

    def emit(boundary: BoundaryType) -> None:
        """Emit buffered text without altering its internal content."""
        nonlocal entry_number

        value = "".join(buffer).strip()
        buffer.clear()

        if not value:
            return

        index = len(segments) + 1
        segments.append(
            TextSegment(
                page_id=page_id,
                segment_id=f"{page_id}::seg-{index:04d}",
                text=value,
                boundary_after=boundary,
                entry_number=entry_number,
            )
        )
        entry_number = None

    cursor = 0

    for match in _TOKEN.finditer(text):
        # Crucially, retain all ordinary text before each recognized token.
        buffer.append(text[cursor:match.start()])
        token = match.group()

        if _ENTRY_MARKER.fullmatch(token):
            if "".join(buffer).strip():
                emit("entry_marker")

            buffer.append(token)
            entry_number = token[1:-1]
        else:
            buffer.append(token)
            emit(_BOUNDARY_TYPES[token])

        cursor = match.end()

    # Retain everything after the final recognized token, too.
    buffer.append(text[cursor:])

    if "".join(buffer).strip():
        emit("end_of_input")

    logger.debug(
        "Segmented page_id={} into {} segments",
        page_id,
        len(segments),
    )
    return segments


def segments_to_dicts(
    segments: list[TextSegment],
) -> list[dict[str, str | None]]:
    """Convert segments to dictionaries suitable for JSONL serialization."""
    return [segment.to_dict() for segment in segments]
