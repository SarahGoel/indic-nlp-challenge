
"""Google Gemini translation engine for the Indic NLP pipeline."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from google import genai
from loguru import logger


_LANGUAGE_NAMES: dict[str, str] = {
    "hin": "Hindi",
    "san": "Sanskrit",
    "ben": "Bengali",
    "guj": "Gujarati",
    "mar": "Marathi",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "pan": "Punjabi",
    "asm": "Assamese",
    "ory": "Odia",
    "urd": "Urdu",
}


def build_zero_shot_prompt(
    source_text: str,
    source_language: str,
    target_language: str,
) -> str:
    """Build a terminology-aware zero-shot translation prompt.

    Args:
        source_text: Verified source segment to translate.
        source_language: Source language name, optionally including its code.
        target_language: Target language name, optionally including its code.

    Returns:
        A complete prompt requesting translation only.

    Raises:
        ValueError: If any required input is empty.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("Source text must not be empty.")
    if not isinstance(source_language, str) or not source_language.strip():
        raise ValueError("Source language must not be empty.")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("Target language must not be empty.")

    source_language = source_language.strip()
    target_language = target_language.strip()
    source_text = source_text.strip()

    return f"""
You are a professional translator specializing in classical
South Asian texts, including Sanskrit and Hindi manuscripts.

TASK
Translate the supplied text from {source_language} into
{target_language}.

TRANSLATION RULES
1. Preserve the original meaning. Do not add explanations,
   interpretations, predictions, or facts absent from the source.
2. Use the conventional writing script of {target_language}.
3. Preserve established astrological, philosophical, religious,
   and cultural terminology. Use a reliable target-language
   equivalent when available; otherwise retain the original term
   or a recognized transliteration.
4. Preserve proper names, dates, numbers, references, and
   parenthesized identifiers such as (१४३७).
5. Preserve meaningful verse boundaries, line order, repetition,
   and punctuation, including danda (।) and double danda (॥),
   wherever appropriate in the target text.
6. Distinguish Sanskrit verses from Hindi prose where relevant.
   Translate the meaning; do not invent a distinction unsupported
   by the supplied text.
7. Do not silently correct uncertain or corrupted source text.
   Do not guess missing words or passages.
8. Do not omit, summarize, expand, or reorder source content.
9. Return only the translated text. Do not include a heading,
   language label, explanation, notes, or quotation marks around
   the complete translation.

SOURCE TEXT
--- BEGIN SOURCE ---
{source_text}
--- END SOURCE ---

Produce the {target_language} translation now.
""".strip()


def build_few_shot_prompt(
    source_text: str,
    source_language: str,
    target_language: str,
    examples: Sequence[tuple[str, str]],
) -> str:
    """Build a translation prompt using reviewed bilingual examples.

    Each example contains (source_text, verified_target_translation).
    Examples must match the requested language direction.
    """
    if not examples:
        raise ValueError("At least one reviewed example is required.")

    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("Source text must not be empty.")

    if (
        not isinstance(source_language, str)
        or not source_language.strip()
        or not isinstance(target_language, str)
        or not target_language.strip()
    ):
        raise ValueError("Source and target languages are required.")

    example_sections: list[str] = []

    for index, (example_source, example_target) in enumerate(
        examples, start=1
    ):
        if (
            not isinstance(example_source, str)
            or not example_source.strip()
            or not isinstance(example_target, str)
            or not example_target.strip()
        ):
            raise ValueError(
                f"Example {index} contains empty source or translation."
            )

        example_sections.append(
            f"Example {index}\n"
            f"Source:\n{example_source.strip()}\n"
            f"Verified {target_language.strip()} translation:\n"
            f"{example_target.strip()}"
        )

    examples_text = "\n\n".join(example_sections)

    base_prompt = build_zero_shot_prompt(
        source_text=source_text,
        source_language=source_language,
        target_language=target_language,
    )

    return (
        "Use the following reviewed bilingual examples to guide "
        "terminology, register, and translation style. They are "
        "examples, not additional source content. Do not copy their "
        "content into the new translation unless it occurs in the "
        "current source.\n\n"
        f"{examples_text}\n\n"
        "--- CURRENT TRANSLATION TASK ---\n"
        f"{base_prompt}"
    )


class GeminiEngineError(RuntimeError):
    """Raised when Gemini translation cannot be completed."""


class GeminiTranslationEngine:
    """Translate text using the Google Gemini API.

    Args:
        model_name: Explicit Gemini model identifier.
        api_key: Optional API key. If omitted, GEMINI_API_KEY is read
            from the process environment.
        client: Optional injected Gemini-compatible client, primarily
            intended for deterministic unit tests.

    Notes:
        This class does not load .env files automatically, write
        translation outputs, or modify source records.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize the engine without making an API request."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name.strip()

        if client is not None:
            self.client = client
            return

        resolved_api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or ""
        ).strip()

        if not resolved_api_key:
            raise GeminiEngineError(
                "Gemini API key is missing. Configure GEMINI_API_KEY "
                "in the process environment before initializing the client."
            )

        try:
            self.client = genai.Client(api_key=resolved_api_key)
        except Exception as exc:
            logger.exception("Gemini client initialization failed.")
            raise GeminiEngineError(
                "Could not initialize the Gemini client."
            ) from exc

    @staticmethod
    def _validate_language(language_code: str, field_name: str) -> str:
        """Validate a supported project language code."""
        if not isinstance(language_code, str):
            raise ValueError(f"{field_name} must be a string.")

        code = language_code.strip().lower()

        if code not in _LANGUAGE_NAMES:
            raise ValueError(
                f"Unsupported {field_name}: {language_code!r}. "
                f"Supported codes: {', '.join(_LANGUAGE_NAMES)}."
            )

        return code

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate one non-empty source segment.

        Args:
            text: Source text from a validated pipeline record.
            source_lang: Source language code, such as 'hin' or 'san'.
            target_lang: Target language code, such as 'ben' or 'tam'.

        Returns:
            The translated text returned by Gemini.

        Raises:
            ValueError: If the input or language codes are invalid.
            GeminiEngineError: If the API call fails or returns no text.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        source_code = self._validate_language(source_lang, "source_lang")
        target_code = self._validate_language(target_lang, "target_lang")

        source_name = _LANGUAGE_NAMES[source_code]
        target_name = _LANGUAGE_NAMES[target_code]

        # Include both human-readable names and project language codes.
        source_label = f"{source_name} ({source_code})"
        target_label = f"{target_name} ({target_code})"

        prompt = build_zero_shot_prompt(
            source_text=text.strip(),
            source_language=source_label,
            target_language=target_label,
        )

        logger.info(
            "Requesting Gemini translation: {} -> {} using model {}.",
            source_code,
            target_code,
            self.model_name,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            translated_text = getattr(response, "text", None)

            if (
                not isinstance(translated_text, str)
                or not translated_text.strip()
            ):
                raise GeminiEngineError(
                    "Gemini returned an empty or unusable text response."
                )

            logger.info(
                "Gemini translation completed: {} -> {}.",
                source_code,
                target_code,
            )

            return translated_text.strip()

        except GeminiEngineError:
            raise
        except Exception as exc:
            logger.exception(
                "Gemini translation request failed: {} -> {}.",
                source_code,
                target_code,
            )
            raise GeminiEngineError(
                f"Gemini translation failed for "
                f"{source_code} -> {target_code}."
            ) from exc
