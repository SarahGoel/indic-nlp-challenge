
"""Tests for manuscript-aware text segmentation."""

import pytest

from indicnlp_pipeline.segmentation.sentence_segmenter import (
    segment_text,
    segments_to_dicts,
)


def test_double_danda_is_preserved() -> None:
    segments = segment_text("धर्मः ॥", "BRIGU_V5_P0001")

    assert len(segments) == 1
    assert segments[0].text == "धर्मः ॥"
    assert segments[0].boundary_after == "double_danda"


def test_single_danda_creates_boundary() -> None:
    segments = segment_text("प्रथमम्। द्वितीयम्।", "page-1")

    assert [segment.text for segment in segments] == [
        "प्रथमम्।",
        "द्वितीयम्।",
    ]
    assert segments[0].boundary_after == "single_danda"


def test_numbered_entry_starts_new_segment() -> None:
    segments = segment_text("पहली प्रविष्टि (१४३७) दूसरी प्रविष्टि।", "page-2")

    assert len(segments) == 2
    assert segments[0].text == "पहली प्रविष्टि"
    assert segments[0].boundary_after == "entry_marker"
    assert segments[1].text == "(१४३७) दूसरी प्रविष्टि।"
    assert segments[1].entry_number == "१४३७"


def test_ascii_entry_number_is_supported() -> None:
    segments = segment_text("(1437) पाठ।", "page-3")

    assert len(segments) == 1
    assert segments[0].entry_number == "1437"


def test_newline_alone_does_not_split_segment() -> None:
    segments = segment_text("पहली पंक्ति\nदूसरी पंक्ति", "page-4")

    assert len(segments) == 1
    assert segments[0].text == "पहली पंक्ति\nदूसरी पंक्ति"


def test_trailing_incomplete_text_is_retained() -> None:
    segments = segment_text("पूर्ण वाक्य। अधूरा पाठ", "page-5")

    assert len(segments) == 2
    assert segments[1].text == "अधूरा पाठ"
    assert segments[1].boundary_after == "end_of_input"


def test_empty_input_returns_no_segments() -> None:
    assert segment_text("", "page-6") == []
    assert segment_text(" \n ", "page-6") == []


def test_repeated_delimiters_do_not_create_empty_segments() -> None:
    segments = segment_text("पाठ।।॥", "page-7")

    assert [segment.text for segment in segments] == ["पाठ।", "।", "॥"]
    assert all(segment.text.strip() for segment in segments)


def test_segment_ids_are_deterministic_and_ordered() -> None:
    first = segment_text("एक। दो।", "page-8")
    second = segment_text("एक। दो।", "page-8")

    assert first == second
    assert [segment.segment_id for segment in first] == [
        "page-8::seg-0001",
        "page-8::seg-0002",
    ]


def test_ascii_bar_delimiters_are_supported() -> None:
    segments = segment_text("पहला|दूसरा||", "page-9")

    assert [segment.text for segment in segments] == [
        "पहला|",
        "दूसरा||",
    ]
    assert segments[0].boundary_after == "ascii_single_bar"
    assert segments[1].boundary_after == "ascii_double_bar"


def test_page_id_is_required() -> None:
    with pytest.raises(ValueError):
        segment_text("पाठ।", "  ")


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError):
        segment_text(None, "page-10")  # type: ignore[arg-type]


def test_segments_can_be_serialized_as_dicts() -> None:
    segments = segment_text("पाठ।", "page-11")

    records = segments_to_dicts(segments)

    assert records[0]["page_id"] == "page-11"
    assert records[0]["text"] == "पाठ।"
    assert records[0]["segment_id"] == "page-11::seg-0001"
