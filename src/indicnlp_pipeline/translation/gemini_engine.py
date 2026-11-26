
"""Google Gemini translation engine for the Indic NLP pipeline."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import types
from loguru import logger


#: Per-request timeout in milliseconds. Without one, a hung connection stalls
#: the whole run indefinitely (the SDK default is effectively unbounded).
REQUEST_TIMEOUT_MS = 120_000

#: Markers for discovered models that accept ``generateContent`` but cannot do
#: Indic translation (image generation, research agents, embeddings, ...).
#: Trying them on every failed page only wastes time.
NON_TEXT_MODEL_MARKERS: tuple[str, ...] = (
    "image",
    "embedding",
    "aqa",
    "deep-research",
    "nano-banana",
    "omni",
    "gemma",
    "veo",
    "imagen",
    "tts",
    "audio",
    "live",
    "robotics",
    "computer-use",
)


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


# Candidate models tried, in order, when no explicit model is configured.
# Models a key cannot use fail fast and are skipped, so an outdated entry is
# harmless. Override with the GEMINI_MODEL_POOL (or GEMINI_MODEL) env var.
#
# Google retires older models for new accounts: gemini-1.5-*, gemini-2.0-flash,
# gemini-2.5-flash and gemini-2.5-pro now return 404 NOT_FOUND. The pool
# therefore leads with current-generation flash models (verified working:
# gemini-3.6-flash, gemini-3.5-flash).
DEFAULT_MODEL_POOL: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


def _normalise_models(value: str | Sequence[str]) -> list[str]:
    """Flatten a model spec (string or list) into ordered unique names."""
    if isinstance(value, str):
        tokens = value.replace(";", ",").split(",")
    else:
        tokens = []
        for item in value:
            tokens.extend(str(item).replace(";", ",").split(","))

    models: list[str] = []
    for token in tokens:
        name = token.strip()
        if name and name not in models:
            models.append(name)
    return models


def _env_model_pool() -> list[str]:
    """Read an optional model pool override from the environment."""
    raw = os.getenv("GEMINI_MODEL_POOL") or os.getenv("GEMINI_MODEL") or ""
    return _normalise_models(raw)


def _is_permanent_failure(exc: BaseException) -> bool:
    """Return True when the error means the model itself is unusable.

    A 4xx response (except 429 rate limiting) will not improve on retry, so
    that model is disabled for the rest of the session. Transient errors
    (429, 5xx) stay in the pool.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return 400 <= code < 500 and code != 429
    return False


class GeminiEngineError(RuntimeError):
    """Raised when a Gemini request cannot be completed."""


class GeminiTranslationEngine:
    """Generate and translate text through the Google Gemini API.

    The engine holds a *pool* of candidate models and uses the first one
    that answers successfully. When a model is unavailable (for example a
    renamed model returning 404) the next candidate is tried, so the code
    keeps working without changes.

    Args:
        model_name: Preferred model, a comma-separated string, or a list of
            model names, tried in order. When omitted, GEMINI_MODEL_POOL (or
            GEMINI_MODEL) is used, falling back to DEFAULT_MODEL_POOL.
        api_key: Optional API key. If omitted, GEMINI_API_KEY is read
            from the process environment.
        client: Optional injected Gemini-compatible client, primarily
            intended for deterministic unit tests.
        discover: When true (default) and a real client is used, models the
            key can actually access are appended to the pool.

    Notes:
        This class does not load .env files automatically, write
        translation outputs, or modify source records.
    """

    #: Kept for backwards compatibility with earlier callers.
    MODEL_POOL = DEFAULT_MODEL_POOL

    def __init__(
        self,
        model_name: str | Sequence[str] | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        discover: bool = True,
    ) -> None:
        """Initialize the engine without making an API request."""
        if model_name is None:
            self.models = _env_model_pool() or list(DEFAULT_MODEL_POOL)
        else:
            self.models = _normalise_models(model_name)
            if not self.models:
                raise ValueError(
                    "model_name must be a non-empty string or a list of "
                    "model names."
                )

        self._active_model: str | None = None
        self._candidates: list[str] | None = None
        self._disabled: set[str] = set()

        if client is not None:
            self.client = client
            # Injected clients (tests) skip network-based discovery.
            self._discover = False
            return

        self._discover = discover

        resolved_api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or ""
        ).strip()

        if not resolved_api_key:
            raise GeminiEngineError(
                "Gemini API key is missing. Configure GEMINI_API_KEY "
                "in the process environment before initializing the client."
            )

        try:
            self.client = genai.Client(
                api_key=resolved_api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        except Exception as exc:
            # Do not log the traceback: library exception state can include
            # request headers that embed the API key.
            logger.error(
                "Gemini client initialization failed ({}).",
                type(exc).__name__,
            )
            raise GeminiEngineError(
                "Could not initialize the Gemini client."
            ) from exc

    @property
    def model_name(self) -> str:
        """Return the working model, or the first configured candidate."""
        return self._active_model or self.models[0]

    def _discover_models(self) -> list[str]:
        """Best-effort list of models the configured API key can access."""
        try:
            discovered: list[str] = []
            for model in self.client.models.list():
                actions = getattr(model, "supported_actions", None) or getattr(
                    model, "supported_generation_methods", None
                )
                # Keep only models that can actually answer generateContent.
                if actions and "generateContent" not in actions:
                    continue
                raw_name = (
                    getattr(model, "name", None)
                    or getattr(model, "model", None)
                    or ""
                )
                name = str(raw_name).split("/")[-1].strip()
                if any(marker in name.lower() for marker in NON_TEXT_MODEL_MARKERS):
                    continue
                if name and name not in discovered:
                    discovered.append(name)
            return discovered
        except Exception as exc:
            logger.debug(
                "Gemini model discovery unavailable ({}).",
                type(exc).__name__,
            )
            return []

    def _candidate_models(self) -> list[str]:
        """Return the ordered model pool, extended once by live discovery."""
        if self._candidates is None:
            ordered = list(self.models)
            if self._discover:
                for name in self._discover_models():
                    if name not in ordered:
                        ordered.append(name)
            self._candidates = ordered
        return self._candidates

    def _generate_content(
        self,
        contents: Any,
        config: Any | None = None,
    ) -> str:
        """Send one request, trying each candidate model until one works."""
        candidates = [
            name
            for name in self._candidate_models()
            if name not in self._disabled
        ]

        ordered = candidates
        if self._active_model and self._active_model not in self._disabled:
            ordered = [self._active_model] + [
                name for name in candidates if name != self._active_model
            ]

        if not ordered:
            raise GeminiEngineError(
                "No usable Gemini model remains; every candidate was "
                f"rejected. Disabled: {', '.join(sorted(self._disabled))}."
            )

        last_error: BaseException | None = None

        for model_name in ordered:
            logger.info("Calling Gemini model {}.", model_name)
            try:
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "contents": contents,
                }
                if config is not None:
                    kwargs["config"] = config

                response = self.client.models.generate_content(**kwargs)
                text = getattr(response, "text", None)

                if not isinstance(text, str) or not text.strip():
                    raise GeminiEngineError(
                        "Gemini returned an empty or unusable text response."
                    )

                self._active_model = model_name
                logger.info("Gemini model {} succeeded.", model_name)
                return text.strip()

            except Exception as exc:
                last_error = exc
                if _is_permanent_failure(exc):
                    # 404/403/400 etc. will not recover: drop this model.
                    self._disabled.add(model_name)
                logger.warning(
                    "Gemini model {} failed ({}: {}). Trying next model.",
                    model_name,
                    type(exc).__name__,
                    getattr(exc, "code", None),
                )
                continue

        logger.error(
            "All Gemini models failed. Tried: {}.",
            ", ".join(ordered),
        )
        raise GeminiEngineError(
            "Gemini request failed. Tried models: "
            f"{', '.join(ordered)}. Last error: {last_error}"
        ) from last_error

    def generate(self, contents: Any, config: Any | None = None) -> str:
        """Generate text from a prompt string or multimodal content list.

        ``contents`` may be a plain string prompt, or a list combining text
        with google.genai.types.Part payloads (as used for OCR).
        """
        return self._generate_content(contents, config=config)

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
            GeminiEngineError: If every model in the pool fails.
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
            "Requesting Gemini translation: {} -> {}.",
            source_code,
            target_code,
        )

        try:
            return self._generate_content(prompt)
        except GeminiEngineError as exc:
            raise GeminiEngineError(
                f"Gemini translation failed for {source_code} -> "
                f"{target_code}. {exc}"
            ) from exc
