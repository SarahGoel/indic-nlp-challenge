
"""Unit tests for the IndicTrans2 inference engine."""

from pathlib import Path

import pytest

from indicnlp_pipeline.translation.indictrans_engine import (
    IndicTransEngine,
)


@pytest.fixture
def engine() -> IndicTransEngine:
    """Create an engine without loading model weights."""
    root = Path(__file__).resolve().parents[1]
    return IndicTransEngine(
        matrix_path=root / "configs" / "translation_matrix.yaml",
        local_files_only=True,
    )


def test_engine_initializes_without_loading_model(
    engine: IndicTransEngine,
) -> None:
    """Initialization must not allocate model weights."""
    assert engine._model is None
    assert engine.checkpoint == "ai4bharat/indictrans2-indic-indic-dist-320M"


def test_hindi_to_bengali_pair(engine: IndicTransEngine) -> None:
    assert engine._validate_pair("hin", "ben") == (
        "hin_Deva",
        "ben_Beng",
    )


def test_sanskrit_to_hindi_pair(engine: IndicTransEngine) -> None:
    assert engine._validate_pair("san", "hin") == (
        "san_Deva",
        "hin_Deva",
    )


def test_excluded_same_language_pair_is_rejected(
    engine: IndicTransEngine,
) -> None:
    with pytest.raises(ValueError, match="not enabled"):
        engine._validate_pair("hin", "hin")


def test_unknown_language_is_rejected(engine: IndicTransEngine) -> None:
    with pytest.raises(ValueError, match="Unknown target language"):
        engine._validate_pair("hin", "xyz")


def test_empty_batch_does_not_load_model(
    engine: IndicTransEngine,
) -> None:
    assert engine.translate_batch([], "hin", "ben") == []
    assert engine._model is None


def test_blank_inputs_preserve_positions(
    engine: IndicTransEngine,
) -> None:
    assert engine.translate_batch(
        ["", "   "],
        "hin",
        "ben",
    ) == ["", "   "]
    assert engine._model is None


def test_string_is_not_accepted_as_batch(
    engine: IndicTransEngine,
) -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        engine.translate_batch("नमस्ते", "hin", "ben")


def test_non_string_batch_item_is_rejected(
    engine: IndicTransEngine,
) -> None:
    with pytest.raises(TypeError, match="Every input segment"):
        engine.translate_batch(
            ["नमस्ते", 123],  # type: ignore[list-item]
            "hin",
            "ben",
        )


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        IndicTransEngine(batch_size=0)