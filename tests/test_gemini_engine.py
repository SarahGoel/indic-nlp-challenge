"""Unit tests for the Gemini translation engine."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indicnlp_pipeline.translation.gemini_engine import (
    GeminiEngineError,
    GeminiTranslationEngine,
)


@pytest.fixture
def mock_client() -> Mock:
    """Return a fake client that never calls the network."""
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="\u092f\u0939 \u0905\u0928\u0941\u0935\u093e\u0926\u093f\u0924 \u092a\u093e\u0920 \u0939\u0948\u0964"
    )
    return client


@pytest.fixture
def engine(mock_client: Mock) -> GeminiTranslationEngine:
    """Build an engine with an injected fake client."""
    return GeminiTranslationEngine(
        model_name="test-model",
        client=mock_client,
    )


def test_translate_returns_trimmed_response(
    engine: GeminiTranslationEngine,
    mock_client: Mock,
) -> None:
    """Return translated text without surrounding whitespace."""
    mock_client.models.generate_content.return_value.text = (
        "  \u09ac\u09be\u0982\u09b2\u09be \u0985\u09a8\u09c1\u09ac\u09be\u09a6  "
    )

    result = engine.translate(
        "\u092f\u0939 \u092e\u0942\u0932 \u092a\u093e\u0920 \u0939\u0948\u0964",
        source_lang="hin",
        target_lang="ben",
    )

    assert result == "\u09ac\u09be\u0982\u09b2\u09be \u0985\u09a8\u09c1\u09ac\u09be\u09a6"
    mock_client.models.generate_content.assert_called_once()

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert "Hindi (hin)" in call_kwargs["contents"]
    assert "Bengali (ben)" in call_kwargs["contents"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_translate_rejects_empty_text(
    engine: GeminiTranslationEngine,
    mock_client: Mock,
    text: str,
) -> None:
    """Reject empty segments without calling the API."""
    with pytest.raises(ValueError, match="non-empty"):
        engine.translate(text, "hin", "ben")

    mock_client.models.generate_content.assert_not_called()


def test_translate_rejects_unsupported_language(
    engine: GeminiTranslationEngine,
    mock_client: Mock,
) -> None:
    """Reject unknown project language codes."""
    with pytest.raises(ValueError, match="Unsupported target_lang"):
        engine.translate("\u0928\u092e\u0942\u0928\u093e \u092a\u093e\u0920", "hin", "xyz")

    mock_client.models.generate_content.assert_not_called()


def test_translate_rejects_non_string_text(
    engine: GeminiTranslationEngine,
) -> None:
    """Reject invalid input types."""
    with pytest.raises(ValueError, match="non-empty"):
        engine.translate(None, "hin", "ben")  # type: ignore[arg-type]


def test_translate_rejects_empty_api_response(
    engine: GeminiTranslationEngine,
    mock_client: Mock,
) -> None:
    """Raise a specific error when Gemini returns no usable text."""
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text="  "
    )

    with pytest.raises(GeminiEngineError, match="empty or unusable"):
        engine.translate("\u0928\u092e\u0942\u0928\u093e \u092a\u093e\u0920", "hin", "ben")


def test_translate_wraps_api_failure(
    engine: GeminiTranslationEngine,
    mock_client: Mock,
) -> None:
    """Convert client exceptions to the engine-specific error."""
    mock_client.models.generate_content.side_effect = RuntimeError(
        "simulated failure"
    )

    with pytest.raises(GeminiEngineError, match="translation failed"):
        engine.translate("\u0928\u092e\u0942\u0928\u093e \u092a\u093e\u0920", "hin", "ben")


def test_initialization_requires_model_name() -> None:
    """Require the model identifier to be configured explicitly."""
    with pytest.raises(ValueError, match="model_name"):
        GeminiTranslationEngine(model_name="  ", client=Mock())


def test_initialization_requires_api_key_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require a key when no client has been injected."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiEngineError, match="API key is missing"):
        GeminiTranslationEngine(model_name="test-model")
