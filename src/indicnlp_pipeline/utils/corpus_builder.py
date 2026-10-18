"""Build and validate the canonical JSONL corpus from verified annotations and OCR outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from indicnlp_pipeline.utils.logger import logger


OCR_ENGINES = ("tesseract", "paddleocr")
REQUIRED_GROUND_TRUTH_FIELDS = (
    "annotation_id",
    "page_id",
    "language",
    "text",
    "annotation_status",
)
TRANSLATION_SYSTEMS = ("indictrans2", "gemini", "human")


class CorpusValidationError(ValueError):
    """Raised when corpus inputs or canonical records fail validation."""


def _page_key(value: str) -> str:
    """Normalize a page identifier or image filename to its page ID."""
    return Path(value).stem.upper()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object and provide a useful error on failure."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusValidationError(
            f"Could not read valid JSON from '{path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise CorpusValidationError(
            f"Expected a JSON object in '{path}', got {type(data).__name__}."
        )

    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and report malformed lines with their line numbers."""
    records: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CorpusValidationError(
                        f"Invalid JSON in '{path}', line {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise CorpusValidationError(
                        f"Expected a JSON object in '{path}', "
                        f"line {line_number}."
                    )

                records.append(record)
    except OSError as exc:
        raise CorpusValidationError(
            f"Could not read '{path}': {exc}"
        ) from exc

    return records


def validate_corpus_record(record: dict[str, Any]) -> None:
    """Validate the required structure and types of one canonical record."""
    required = (
        "id",
        "source_language",
        "source_script",
        "page",
        "segment_id",
        "source_page",
        "ocr",
        "ocr_corrected",
        "ground_truth",
        "ground_truth_metadata",
        "human_verified",
        "translations",
    )

    missing = [field for field in required if field not in record]
    if missing:
        raise CorpusValidationError(
            f"Record is missing required fields: {', '.join(missing)}"
        )

    for field in ("id", "source_language", "source_script", "segment_id",
                  "source_page", "ground_truth"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise CorpusValidationError(
                f"Field '{field}' must be a non-empty string."
            )

    if isinstance(record["page"], bool) or not isinstance(record["page"], int):
        raise CorpusValidationError("Field 'page' must be an integer.")

    if not isinstance(record["human_verified"], bool):
        raise CorpusValidationError("Field 'human_verified' must be boolean.")

    if record["ocr_corrected"] is not None and not isinstance(
        record["ocr_corrected"], str
    ):
        raise CorpusValidationError(
            "Field 'ocr_corrected' must be a string or null."
        )

    if not isinstance(record["ocr"], dict):
        raise CorpusValidationError("Field 'ocr' must be an object.")

    for engine in OCR_ENGINES:
        if engine not in record["ocr"] or not isinstance(
            record["ocr"][engine], dict
        ):
            raise CorpusValidationError(
                f"Field 'ocr.{engine}' must be an object keyed by variant."
            )

    if not isinstance(record["ground_truth_metadata"], dict):
        raise CorpusValidationError(
            "Field 'ground_truth_metadata' must be an object."
        )

    translations = record["translations"]
    if not isinstance(translations, dict):
        raise CorpusValidationError("Field 'translations' must be an object.")

    for language, systems in translations.items():
        if not isinstance(language, str) or not isinstance(systems, dict):
            raise CorpusValidationError(
                "Each translation language must map to an object."
            )

        for system in TRANSLATION_SYSTEMS:
            if system not in systems:
                raise CorpusValidationError(
                    f"Missing translations.{language}.{system}."
                )
            if systems[system] is not None and not isinstance(
                systems[system], str
            ):
                raise CorpusValidationError(
                    f"translations.{language}.{system} must be a string or null."
                )


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load page metadata keyed by page ID."""
    if not path.is_file():
        raise CorpusValidationError(f"Page manifest not found: '{path}'.")

    manifest: dict[str, dict[str, str]] = {}

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            required_columns = {"page_id", "pdf_page_number"}
            if not reader.fieldnames or not required_columns.issubset(
                reader.fieldnames
            ):
                raise CorpusValidationError(
                    "Page manifest must contain 'page_id' and "
                    "'pdf_page_number' columns."
                )

            for row in reader:
                page_id = (row.get("page_id") or "").strip()
                if not page_id:
                    raise CorpusValidationError(
                        "Page manifest contains a row with an empty page_id."
                    )

                if page_id in manifest:
                    raise CorpusValidationError(
                        f"Duplicate page ID in manifest: '{page_id}'."
                    )

                manifest[page_id] = row
    except OSError as exc:
        raise CorpusValidationError(
            f"Could not read page manifest '{path}': {exc}"
        ) from exc

    return manifest


def _load_ocr_outputs(
    ocr_root: Path,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Load raw OCR JSON indexed by page ID, engine, and preprocessing variant."""
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    for engine in OCR_ENGINES:
        engine_dir = ocr_root / engine
        if not engine_dir.is_dir():
            raise CorpusValidationError(
                f"OCR engine directory not found: '{engine_dir}'."
            )

        for path in sorted(engine_dir.rglob("*.json")):
            payload = _read_json(path)

            for field in ("source_page", "preprocessing_variant", "ocr_engine"):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise CorpusValidationError(
                        f"Missing or invalid '{field}' in '{path}'."
                    )

            declared_engine = payload["ocr_engine"].lower()
            if declared_engine != engine:
                raise CorpusValidationError(
                    f"OCR engine mismatch in '{path}': directory says "
                    f"'{engine}', JSON says '{declared_engine}'."
                )

            page_id = _page_key(payload["source_page"])
            variant = payload["preprocessing_variant"]

            page_outputs = results.setdefault(page_id, {})
            engine_outputs = page_outputs.setdefault(engine, {})

            if variant in engine_outputs:
                raise CorpusValidationError(
                    f"Duplicate OCR output for page '{page_id}', engine "
                    f"'{engine}', variant '{variant}'."
                )

            # Keep the original payload, including arrays and metadata.
            engine_outputs[variant] = payload

    return results


def build_corpus_records(repo_root: str | Path) -> list[dict[str, Any]]:
    """Build canonical records in memory without writing any files.

    One record is produced for each ground-truth annotation. In the current
    pilot dataset, these are page-level annotations, not verse-level segments.
    """
    root = Path(repo_root).resolve()
    config_path = root / "configs" / "config.yaml"
    ground_truth_path = root / "data" / "ground_truth" / "ground_truth.jsonl"
    manifest_path = root / "data" / "raw_scans" / "page_manifest.csv"
    ocr_root = root / "data" / "ocr_raw"

    if not config_path.is_file():
        raise CorpusValidationError(f"Config not found: '{config_path}'.")

    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise CorpusValidationError(
            f"Could not read project configuration '{config_path}': {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise CorpusValidationError("Project configuration must be a YAML object.")

    project_config = config.get("project", {})
    target_languages = project_config.get("target_languages")

    if not isinstance(target_languages, list) or not all(
        isinstance(language, str) and language for language in target_languages
    ):
        raise CorpusValidationError(
            "configs/config.yaml must define a list of target_languages."
        )

    annotations = _read_jsonl(ground_truth_path)
    if not annotations:
        raise CorpusValidationError("Ground-truth JSONL contains no records.")

    manifest = _load_manifest(manifest_path)
    ocr_outputs = _load_ocr_outputs(ocr_root)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for line_number, annotation in enumerate(annotations, start=1):
        missing = [
            field for field in REQUIRED_GROUND_TRUTH_FIELDS
            if field not in annotation
        ]
        if missing:
            raise CorpusValidationError(
                f"Ground-truth record {line_number} is missing: "
                f"{', '.join(missing)}."
            )

        annotation_id = annotation["annotation_id"]
        page_id = annotation["page_id"]
        language = annotation["language"]
        text = annotation["text"]
        status = annotation["annotation_status"]

        for field, value in (
            ("annotation_id", annotation_id),
            ("page_id", page_id),
            ("language", language),
            ("text", text),
            ("annotation_status", status),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CorpusValidationError(
                    f"Ground-truth record {line_number}: '{field}' "
                    "must be a non-empty string."
                )

        record_id = annotation_id.lower()
        if record_id in seen_ids:
            raise CorpusValidationError(
                f"Duplicate annotation/corpus ID: '{annotation_id}'."
            )
        seen_ids.add(record_id)

        manifest_row = manifest.get(page_id)
        if manifest_row is None:
            raise CorpusValidationError(
                f"Page '{page_id}' from ground truth is absent from the manifest."
            )

        try:
            pdf_page_number = int(manifest_row["pdf_page_number"])
        except (TypeError, ValueError) as exc:
            raise CorpusValidationError(
                f"Invalid pdf_page_number for page '{page_id}'."
            ) from exc

        page_ocr = ocr_outputs.get(_page_key(page_id), {})
        engine_data: dict[str, dict[str, dict[str, Any]]] = {}

        for engine in OCR_ENGINES:
            engine_data[engine] = page_ocr.get(engine, {})

            if not engine_data[engine]:
                logger.warning(
                    "No {} OCR outputs found for page {}.",
                    engine,
                    page_id,
                )

        record = {
            "id": record_id,
            "source_language": language,
            "source_script": "Devanagari",
            "page": pdf_page_number,
            "segment_id": annotation_id,
            "source_page": annotation.get("source_page", f"{page_id}.png"),
            "ocr": engine_data,
            "ocr_corrected": None,
            "ground_truth": text,
            "ground_truth_metadata": {
                "annotation_id": annotation_id,
                "page_id": page_id,
                "annotation_status": status,
                "confidence": annotation.get("confidence"),
                "annotator": annotation.get("annotator"),
                "notes": annotation.get("notes"),
            },
            "human_verified": status.lower() == "verified",
            "translations": {
                language_code: {
                    system: None for system in TRANSLATION_SYSTEMS
                }
                for language_code in target_languages
            },
        }

        validate_corpus_record(record)
        records.append(record)

    return records


def write_corpus_jsonl(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write JSONL as UTF-8 without a BOM.

    Writes to a temporary sibling file first to avoid leaving a partial output
    if serialization fails.
    """
    destination = Path(output_path).resolve()

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing corpus: '{destination}'. "
            "Pass overwrite=True only when replacement is intentional."
        )

    seen_ids: set[str] = set()
    for record in records:
        validate_corpus_record(record)
        record_id = record["id"]
        if record_id in seen_ids:
            raise CorpusValidationError(
                f"Duplicate corpus ID before writing: '{record_id}'."
            )
        seen_ids.add(record_id)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Wrote {} validated corpus records to {}.",
        len(records),
        destination,
    )
    return destination


def build_corpus(
    repo_root: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Build, validate, and write the corpus; return the records."""
    root = Path(repo_root).resolve()
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "data" / "corpus" / "corpus.jsonl"
    )

    records = build_corpus_records(root)
    write_corpus_jsonl(records, destination, overwrite=overwrite)
    return records
