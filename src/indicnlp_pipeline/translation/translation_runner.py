
"""Orchestrate IndicTrans2 and Gemini translation for verified segments.

Outputs are written as independent UTF-8 JSONL files for each engine and
configured source-target language pair. Ground truth is never modified.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class TranslationRunnerError(RuntimeError):
    """Raised when translation orchestration cannot safely continue."""


def _repository_root() -> Path:
    """Locate the repository using its project configuration."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "configs" / "config.yaml").is_file():
            return candidate
    raise TranslationRunnerError("Could not locate the repository root.")


def _load_dotenv(root: Path) -> None:
    """Load GEMINI_API_KEY from the root .env without logging its value.

    Existing process environment variables take precedence. This intentionally
    handles only the project's single required credential.
    """
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return

    env_path = root / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY":
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            if value:
                os.environ["GEMINI_API_KEY"] = value
            return


def _read_matrix(path: Path) -> dict[str, Any]:
    """Load the translation matrix and validate its required sections."""
    if not path.is_file():
        raise FileNotFoundError(f"Translation matrix not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        matrix = yaml.safe_load(file)

    if not isinstance(matrix, dict):
        raise TranslationRunnerError(
            "Translation matrix must be a YAML mapping."
        )

    for section in ("languages", "translation_matrix", "backends"):
        if not isinstance(matrix.get(section), dict):
            raise TranslationRunnerError(
                f"Translation matrix requires a '{section}' mapping."
            )

    languages = matrix["languages"]
    codes = [
        entry.get("model_code")
        for entry in languages.values()
        if isinstance(entry, dict) and entry.get("model_code")
    ]

    validation = matrix.get("validation", {})
    if (
        isinstance(validation, dict)
        and validation.get("require_unique_model_codes", False)
        and len(codes) != len(set(codes))
    ):
        raise TranslationRunnerError(
            "Duplicate model language codes found in translation_matrix.yaml."
        )

    return matrix


def _load_segments(
    path: Path,
    source_languages: list[str],
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """Load verified ground-truth records, preserving stable IDs and text.

    The optional limit applies independently to each source language.
    Duplicate annotation IDs within a source language are rejected.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    selected: dict[str, list[dict[str, Any]]] = {
        language: [] for language in source_languages
    }
    seen_ids: dict[str, set[str]] = {
        language: set() for language in source_languages
    }

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TranslationRunnerError(
                    f"Invalid JSON on ground-truth line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise TranslationRunnerError(
                    f"Ground-truth line {line_number} must be a JSON object."
                )

            language = record.get("language")
            if language not in selected:
                continue

            # Only human-verified source text is eligible for this runner.
            if record.get("annotation_status") != "verified":
                continue

            segment_id = record.get("annotation_id")
            source_text = record.get("text")

            if not isinstance(segment_id, str) or not segment_id.strip():
                raise TranslationRunnerError(
                    f"Missing annotation_id on line {line_number}."
                )
            if not isinstance(source_text, str) or not source_text.strip():
                raise TranslationRunnerError(
                    f"Missing or blank text on line {line_number}."
                )
            if segment_id in seen_ids[language]:
                raise TranslationRunnerError(
                    f"Duplicate annotation_id {segment_id!r} "
                    f"for source language {language}."
                )

            seen_ids[language].add(segment_id)

            if limit is None or len(selected[language]) < limit:
                selected[language].append(record)

    for language, records in selected.items():
        logger.info(
            "Selected {} verified source segments for {}.",
            len(records),
            language,
        )

    return selected


def _validate_pair(
    matrix: dict[str, Any],
    source_language: str,
    target_language: str,
) -> None:
    """Require an explicitly configured, non-identity language pair."""
    languages = matrix["languages"]
    pairs = matrix["translation_matrix"]

    if source_language not in languages:
        raise TranslationRunnerError(
            f"Unknown source language: {source_language}"
        )
    if target_language not in languages:
        raise TranslationRunnerError(
            f"Unknown target language: {target_language}"
        )
    if source_language == target_language:
        raise TranslationRunnerError(
            f"Identity pair {source_language}->{target_language} "
            "is not a translation."
        )

    pair = pairs.get(source_language, {}).get(target_language)
    if not isinstance(pair, dict) or pair.get("status") != "configured":
        reason = (
            pair.get("reason", pair.get("status", "not configured"))
            if isinstance(pair, dict)
            else "not configured"
        )
        raise TranslationRunnerError(
            f"Pair {source_language}->{target_language} is unavailable: "
            f"{reason}"
        )

    if not isinstance(languages[source_language], dict):
        raise TranslationRunnerError(
            f"Invalid language configuration for {source_language}."
        )
    if not isinstance(languages[target_language], dict):
        raise TranslationRunnerError(
            f"Invalid language configuration for {target_language}."
        )
    if not languages[source_language].get("model_code"):
        raise TranslationRunnerError(
            f"Missing model code for {source_language}."
        )
    if not languages[target_language].get("model_code"):
        raise TranslationRunnerError(
            f"Missing model code for {target_language}."
        )


def _output_path(
    root: Path,
    engine_name: str,
    source_language: str,
    target_language: str,
) -> Path:
    """Return the dedicated JSONL path for one engine and language pair."""
    return (
        root
        / "data"
        / "translations"
        / engine_name
        / f"{source_language}_to_{target_language}.jsonl"
    )


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    """Atomically write JSONL using UTF-8 without a BOM."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Use --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())

        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _result_record(
    source_record: dict[str, Any],
    target_language: str,
    engine_name: str,
    model_name: str,
    translated_text: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create an output record retaining source identity and provenance."""
    return {
        "annotation_id": source_record["annotation_id"],
        "page_id": source_record.get("page_id"),
        "source_page": source_record.get("source_page"),
        "source_language": source_record["language"],
        "target_language": target_language,
        "source_text": source_record["text"],
        "translated_text": translated_text,
        "engine": engine_name,
        "model": model_name,
        "translation_status": "success" if error is None else "error",
        "error": error,
    }


def run_translations(
    *,
    root: Path | None = None,
    source_languages: list[str] | None = None,
    target_languages: list[str] | None = None,
    engines: list[str] | None = None,
    limit: int | None = 2,
    gemini_model: str | None = None,
    device: str | None = None,
    local_files_only: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Run selected translations and write separate per-pair JSONL outputs.

    The limit is applied independently to each source language. Model outputs
    and failures are recorded per segment. Existing outputs are protected
    unless overwrite=True.
    """
    root = (root or _repository_root()).resolve()
    source_languages = source_languages or ["hin"]
    target_languages = target_languages or [
        "hin", "ben", "guj", "mar", "tam", "tel",
        "kan", "mal", "pan", "asm", "ory", "urd",
    ]
    engines = engines or ["indictrans2", "gemini"]

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1, or None for all segments.")
    if not source_languages or not target_languages or not engines:
        raise ValueError(
            "Source languages, targets, and engines cannot be empty."
        )
    if len(set(source_languages)) != len(source_languages):
        raise ValueError("Duplicate source languages were requested.")
    if len(set(target_languages)) != len(target_languages):
        raise ValueError("Duplicate target languages were requested.")
    if len(set(engines)) != len(engines):
        raise ValueError("Duplicate engines were requested.")

    unknown_engines = set(engines) - {"indictrans2", "gemini"}
    if unknown_engines:
        raise ValueError(f"Unknown engines: {sorted(unknown_engines)}")

    matrix_path = root / "configs" / "translation_matrix.yaml"
    matrix = _read_matrix(matrix_path)

    # Only load the Gemini credential when Gemini is actually requested.
    if "gemini" in engines:
        _load_dotenv(root)

    selected_segments = _load_segments(
        root / "data" / "ground_truth" / "ground_truth.jsonl",
        source_languages,
        limit,
    )

    # Plan all work and check output collisions before loading models
    # or making paid API requests.
    jobs: list[
        tuple[str, str, str, list[dict[str, Any]], Path]
    ] = []

    for source_language in source_languages:
        for target_language in target_languages:
            for engine_name in engines:
                if source_language == target_language:
                    logger.info(
                        "Skipping identity pair {}->{}.",
                        source_language,
                        target_language,
                    )
                    continue

                _validate_pair(matrix, source_language, target_language)
                records = selected_segments[source_language]

                if not records:
                    logger.warning(
                        "No verified source records available for {}.",
                        source_language,
                    )
                    continue

                output = _output_path(
                    root,
                    engine_name,
                    source_language,
                    target_language,
                )

                if output.exists() and not overwrite:
                    raise FileExistsError(
                        f"Output already exists: {output}. "
                        "Use --overwrite to replace it."
                    )

                jobs.append(
                    (
                        engine_name,
                        source_language,
                        target_language,
                        records,
                        output,
                    )
                )

    if not jobs:
        logger.warning("No translation jobs were eligible to run.")
        return []

    indic_backend = matrix["backends"].get("indictrans2", {})
    gemini_backend = matrix["backends"].get("gemini", {})

    indic_model = str(indic_backend.get("checkpoint") or "").strip()
    resolved_gemini_model = str(
        gemini_model or gemini_backend.get("model") or ""
    ).strip()

    if (
        any(job[0] == "indictrans2" for job in jobs)
        and not indic_model
    ):
        raise TranslationRunnerError(
            "No IndicTrans2 checkpoint is configured in the matrix."
        )

    if (
        any(job[0] == "gemini" for job in jobs)
        and not resolved_gemini_model
    ):
        raise TranslationRunnerError(
            "Gemini model is not configured. Supply --gemini-model explicitly."
        )

    # Lazy imports: each backend is imported only if it has eligible jobs.
    # This lets the WSL IndicTrans2 environment run without google-genai.
    indic_engine = None
    gemini_engine = None

    if any(job[0] == "indictrans2" for job in jobs):
        from indicnlp_pipeline.translation.indictrans_engine import (
            IndicTransEngine,
        )

        indic_engine = IndicTransEngine(
            matrix_path=matrix_path,
            device=device,
            local_files_only=local_files_only,
        )

    if any(job[0] == "gemini" for job in jobs):
        from indicnlp_pipeline.translation.gemini_engine import (
            GeminiTranslationEngine,
        )

        gemini_engine = GeminiTranslationEngine(
            model_name=resolved_gemini_model
        )

    written_paths: list[Path] = []

    for (
        engine_name,
        source_language,
        target_language,
        records,
        output,
    ) in jobs:
        logger.info(
            "Starting {} translation {}->{} for {} segments.",
            engine_name,
            source_language,
            target_language,
            len(records),
        )

        results: list[dict[str, Any]] = []

        if engine_name == "indictrans2":
            assert indic_engine is not None
            model_name = indic_model

            try:
                translations = indic_engine.translate_batch(
                    [record["text"] for record in records],
                    source_lang=source_language,
                    target_lang=target_language,
                )

                if len(translations) != len(records):
                    raise TranslationRunnerError(
                        "IndicTrans2 output count differs from input count."
                    )

                for source_record, translation in zip(
                    records,
                    translations,
                    strict=True,
                ):
                    results.append(
                        _result_record(
                            source_record,
                            target_language,
                            engine_name,
                            model_name,
                            translation,
                        )
                    )

            except Exception as exc:
                logger.exception(
                    "IndicTrans2 batch failed for {}->{}.",
                    source_language,
                    target_language,
                )

                results = [
                    _result_record(
                        record,
                        target_language,
                        engine_name,
                        model_name,
                        None,
                        f"Batch failed: {type(exc).__name__}: {exc}",
                    )
                    for record in records
                ]

        else:
            assert gemini_engine is not None
            model_name = resolved_gemini_model

            for source_record in records:
                try:
                    translation = gemini_engine.translate(
                        source_record["text"],
                        source_lang=source_language,
                        target_lang=target_language,
                    )

                    results.append(
                        _result_record(
                            source_record,
                            target_language,
                            engine_name,
                            model_name,
                            translation,
                        )
                    )

                except Exception as exc:
                    logger.exception(
                        "Gemini segment failed: {} {}->{}.",
                        source_record["annotation_id"],
                        source_language,
                        target_language,
                    )

                    results.append(
                        _result_record(
                            source_record,
                            target_language,
                            engine_name,
                            model_name,
                            None,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )

        _write_jsonl(output, results, overwrite=overwrite)
        written_paths.append(output)

        statuses = Counter(
            record["translation_status"] for record in results
        )
        logger.info(
            "Saved {} records to {} (success={}, error={}).",
            len(results),
            output,
            statuses["success"],
            statuses["error"],
        )

    return written_paths
