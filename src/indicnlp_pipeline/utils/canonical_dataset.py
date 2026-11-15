"""Build and validate the canonical full-volume segment dataset (Phase H).

The canonical dataset is a UTF-8 (no BOM) JSONL file in which each record is
one NLP segment produced by Phase G. Machine-produced text is never marked as
ground truth: every record carries an explicit ``text_provenance`` block and
``human_verified`` is always ``False`` unless a human verification workflow
later replaces the record.

This module is separate from :mod:`indicnlp_pipeline.utils.corpus_builder`,
which builds the page-level pilot corpus. Keeping the two builders separate
avoids changing the meaning of existing pilot artifacts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from indicnlp_pipeline.utils.logger import logger


TRANSLATION_SYSTEMS = ("indictrans2", "gemini", "human")
TEXT_PROVENANCE_STAGE = "ocr_normalized+nlp_segmentation"
_REQUIRED_SEGMENT_FIELDS = (
    "segment_id",
    "page_id",
    "page_number",
    "segment_index",
    "language",
    "language_confidence",
    "boundary_after",
    "character_count",
    "text",
    "ocr_engine",
    "preprocessing_variant",
)

# Fields every canonical record must contain, with their expected Python type.
_REQUIRED_RECORD_TYPES: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "segment_id": str,
    "page_id": str,
    "page": int,
    "segment_index": int,
    "source_page": str,
    "source_script": str,
    "source_language": str,
    "language_confidence": (int, float),
    "boundary_after": str,
    "character_count": int,
    "text": str,
    "text_provenance": dict,
    "ocr": dict,
    "human_verified": bool,
    "translations": dict,
}


class CanonicalDatasetError(ValueError):
    """Raised when inputs or canonical records fail validation."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, reporting malformed lines with their line numbers."""
    if not path.is_file():
        raise CanonicalDatasetError(f"JSONL input not found: '{path}'.")

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CanonicalDatasetError(
                        f"Invalid JSON in '{path}', line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CanonicalDatasetError(
                        f"Expected a JSON object in '{path}', line {line_number}."
                    )
                records.append(record)
    except OSError as exc:
        raise CanonicalDatasetError(f"Could not read '{path}': {exc}") from exc
    return records


def _read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalDatasetError(
            f"Could not read valid JSON from '{path}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CanonicalDatasetError(
            f"Expected a JSON object in '{path}', got {type(data).__name__}."
        )
    return data


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_record_id(page_id: str, segment_index: int) -> str:
    """Return a deterministic, lowercase canonical record ID."""
    return f"{page_id.lower()}_seg{segment_index:04d}"


def _load_target_languages(config_path: Path) -> list[str]:
    """Read the configured target languages from the project configuration."""
    if not config_path.is_file():
        raise CanonicalDatasetError(f"Config not found: '{config_path}'.")
    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise CanonicalDatasetError(
            f"Could not read project configuration '{config_path}': {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise CanonicalDatasetError("Project configuration must be a YAML object.")
    target_languages = config.get("project", {}).get("target_languages")
    if not isinstance(target_languages, list) or not all(
        isinstance(language, str) and language for language in target_languages
    ):
        raise CanonicalDatasetError(
            "configs/config.yaml must define a list of target_languages."
        )
    return list(target_languages)


def _load_page_ocr(
    engine_dir: Path, engine: str, base: Path
) -> dict[str, dict[str, Any]]:
    """Load compact OCR provenance for one engine, keyed by page ID then variant.

    Full OCR payloads stay in the raw OCR directories. The canonical dataset
    only records where the output lives and how to verify it, which avoids
    duplicating every page's OCR text once per segment.
    """
    outputs: dict[str, dict[str, Any]] = {}
    if not engine_dir.is_dir():
        logger.warning("OCR engine directory missing, skipping: {}", engine_dir)
        return outputs

    for path in sorted(engine_dir.glob("*.json")):
        payload = _read_json(path)
        page_id = str(payload.get("page_id") or "").strip()
        variant = str(payload.get("preprocessing_variant") or "").strip()
        declared = str(payload.get("ocr_engine") or "").strip().lower()
        if not page_id or not variant:
            logger.warning("Skipping OCR payload without page_id/variant: {}", path)
            continue
        if declared and declared != engine:
            raise CanonicalDatasetError(
                f"OCR engine mismatch in '{path}': directory says '{engine}', "
                f"JSON says '{declared}'."
            )

        text = payload.get("text")
        if isinstance(text, str):
            char_count = len(text)
            text_field = "string"
        elif isinstance(text, list):
            char_count = sum(len(str(item)) for item in text)
            text_field = "list"
        else:
            char_count = 0
            text_field = type(text).__name__

        engine_outputs = outputs.setdefault(page_id.upper(), {})
        if variant in engine_outputs:
            raise CanonicalDatasetError(
                f"Duplicate OCR payload for page '{page_id}', engine "
                f"'{engine}', variant '{variant}'."
            )
        engine_outputs[variant] = {
            "path": str(path.resolve().relative_to(base)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "engine_version": payload.get("engine_version"),
            "engine_params": payload.get("engine_params"),
            "char_count": char_count,
            "text_field": text_field,
        }
    return outputs


def _load_references(path: Path) -> dict[str, dict[str, Any]]:
    """Load page-level human references keyed by page ID."""
    references: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        logger.warning("Ground-truth file not found: {}", path)
        return references
    for record in _read_jsonl(path):
        page_id = str(record.get("page_id") or "").strip().upper()
        if not page_id:
            continue
        references[page_id] = record
    return references


def _page_reference(page_id: str, references: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return a compact reference descriptor for a page, or ``None``."""
    annotation = references.get(page_id.upper())
    if annotation is None:
        return None
    return {
        "annotation_id": annotation.get("annotation_id"),
        "annotation_status": annotation.get("annotation_status"),
        "confidence": annotation.get("confidence"),
        "annotator": annotation.get("annotator"),
        "scope": "page",
    }


def build_records(
    segments: list[dict[str, Any]],
    normalized_index: dict[str, dict[str, Any]],
    ocr_by_engine: dict[str, dict[str, dict[str, Any]]],
    references: dict[str, dict[str, Any]],
    target_languages: list[str],
) -> list[dict[str, Any]]:
    """Build canonical records from NLP segments without writing any files.

    Records are ordered deterministically by ``(page_number, segment_index)``.
    """
    records: list[dict[str, Any]] = []
    ordered = sorted(
        segments,
        key=lambda seg: (int(seg["page_number"]), int(seg["segment_index"])),
    )

    for segment in ordered:
        for field in _REQUIRED_SEGMENT_FIELDS:
            if field not in segment:
                raise CanonicalDatasetError(
                    f"Segment is missing required field '{field}': "
                    f"{segment.get('segment_id')!r}."
                )

        page_id = str(segment["page_id"])
        page_key = page_id.upper()
        segment_index = int(segment["segment_index"])
        engine = str(segment["ocr_engine"]).lower()
        variant = str(segment["preprocessing_variant"])

        normalized_entry = normalized_index.get(page_key)
        if normalized_entry is None:
            raise CanonicalDatasetError(
                f"No normalized OCR provenance found for page '{page_id}'."
            )

        record_id = make_record_id(page_id, segment_index)

        engine_payloads: dict[str, dict[str, Any]] = {}
        for engine_name in ("tesseract", "paddleocr"):
            engine_payloads[engine_name] = ocr_by_engine.get(engine_name, {}).get(
                page_key, {}
            )

        record: dict[str, Any] = {
            "id": record_id,
            "segment_id": segment["segment_id"],
            "page_id": page_id,
            "page": int(segment["page_number"]),
            "segment_index": segment_index,
            "source_page": f"{page_id}.png",
            "source_script": "Devanagari",
            "source_language": segment["language"],
            "language_confidence": segment["language_confidence"],
            "boundary_after": segment["boundary_after"],
            "entry_number": segment.get("entry_number"),
            "character_count": int(segment["character_count"]),
            "text": segment["text"],
            "text_provenance": {
                "stage": TEXT_PROVENANCE_STAGE,
                "engine": engine,
                "variant": variant,
                "normalization": "NFC",
                "source_ocr_engine": engine,
                "normalized_ocr_path": normalized_entry["path"],
                "normalized_ocr_sha256": normalized_entry["sha256"],
            },
            "ocr": engine_payloads,
            "ocr_corrected": None,
            "ground_truth": None,
            "ground_truth_metadata": None,
            "page_reference": _page_reference(page_id, references),
            "human_verified": False,
            "translations": {
                language: {system: None for system in TRANSLATION_SYSTEMS}
                for language in target_languages
            },
        }

        validate_record(record)
        records.append(record)

    return records


def validate_record(record: dict[str, Any]) -> None:
    """Validate the required structure and types of one canonical record."""
    missing = [field for field in _REQUIRED_RECORD_TYPES if field not in record]
    if missing:
        raise CanonicalDatasetError(
            f"Record '{record.get('id')}' is missing fields: {', '.join(missing)}."
        )

    for field, expected in _REQUIRED_RECORD_TYPES.items():
        value = record[field]
        if expected is int and isinstance(value, bool):
            raise CanonicalDatasetError(f"Field '{field}' must be an integer.")
        if not isinstance(value, expected):
            raise CanonicalDatasetError(
                f"Field '{field}' has type {type(value).__name__}, "
                f"expected {expected}."
            )

    for field in ("id", "segment_id", "page_id", "source_language", "text"):
        if not record[field].strip():
            raise CanonicalDatasetError(f"Field '{field}' must be non-empty.")

    if record["human_verified"] is not False:
        raise CanonicalDatasetError(
            "Machine-produced records must not be marked human_verified=True."
        )

    provenance = record["text_provenance"]
    for field in (
        "stage",
        "engine",
        "variant",
        "normalization",
        "normalized_ocr_path",
        "normalized_ocr_sha256",
    ):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise CanonicalDatasetError(
                f"text_provenance.{field} must be a non-empty string."
            )

    ocr = record["ocr"]
    for engine in ("tesseract", "paddleocr"):
        if engine not in ocr or not isinstance(ocr[engine], dict):
            raise CanonicalDatasetError(
                f"Field 'ocr.{engine}' must be an object keyed by variant."
            )

    translations = record["translations"]
    if not isinstance(translations, dict) or not translations:
        raise CanonicalDatasetError("Field 'translations' must be a non-empty object.")
    for language, systems in translations.items():
        if not isinstance(systems, dict):
            raise CanonicalDatasetError(
                f"translations.{language} must map to an object."
            )
        for system in TRANSLATION_SYSTEMS:
            if system not in systems:
                raise CanonicalDatasetError(
                    f"Missing translations.{language}.{system}."
                )
            if systems[system] is not None and not isinstance(systems[system], str):
                raise CanonicalDatasetError(
                    f"translations.{language}.{system} must be a string or null."
                )


def write_dataset(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write the dataset as UTF-8 JSONL without a BOM.

    Writes to a temporary sibling file first so a serialization failure cannot
    leave a partially written dataset.
    """
    destination = Path(output_path).resolve()

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: '{destination}'. "
            "Pass overwrite=True only when replacement is intentional."
        )

    seen_ids: set[str] = set()
    for record in records:
        validate_record(record)
        record_id = record["id"]
        if record_id in seen_ids:
            raise CanonicalDatasetError(
                f"Duplicate canonical record ID before writing: '{record_id}'."
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
        "Wrote {} validated canonical records to {}.",
        len(records),
        destination,
    )
    return destination


def _scan_dataset_file(path: Path, expected_ids: list[str]) -> dict[str, Any]:
    """Re-read the written dataset and verify line-level integrity and encoding."""
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")

    ids: list[str] = []
    bad_lines: list[int] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            bad_lines.append(line_number)
            continue
        ids.append(payload.get("id"))

    duplicates = sorted(
        rid for rid, count in Counter(ids).items() if count > 1
    )
    return {
        "has_bom": has_bom,
        "line_count": len(ids),
        "bad_lines": bad_lines,
        "duplicate_ids": duplicates,
        "ids_match_build": ids == expected_ids,
    }


def validate_dataset(
    records: list[dict[str, Any]],
    *,
    dataset_path: Path,
    expected_pages: list[str],
    expected_segment_counts: dict[str, int],
    target_languages: list[str],
    reference_pages: list[str],
) -> dict[str, Any]:
    """Validate the built dataset and return a machine-readable report."""
    from indicnlp_pipeline.ocr.production_runner import utc_now_iso

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "dataset_path": str(dataset_path),
        "expected_pages": len(expected_pages),
        "target_languages": target_languages,
    }

    ids = [record["id"] for record in records]
    id_counter = Counter(ids)
    report["record_count"] = len(records)
    report["unique_id_count"] = len(id_counter)
    report["duplicate_ids"] = sorted(
        rid for rid, count in id_counter.items() if count > 1
    )

    counts_by_page: dict[str, int] = {}
    indices_by_page: dict[str, list[int]] = {}
    for record in records:
        page_id = record["page_id"].upper()
        counts_by_page[page_id] = counts_by_page.get(page_id, 0) + 1
        indices_by_page.setdefault(page_id, []).append(int(record["segment_index"]))

    page_keys = {page.upper() for page in expected_pages}
    report["page_count"] = len(counts_by_page)
    report["missing_pages"] = sorted(page_keys - set(counts_by_page))
    report["extra_pages"] = sorted(set(counts_by_page) - page_keys)

    non_contiguous = [
        page_id
        for page_id, indices in indices_by_page.items()
        if sorted(indices) != list(range(1, len(indices) + 1))
    ]
    report["non_contiguous_pages"] = sorted(non_contiguous)

    mismatched = {
        page_id: {
            "expected": expected_segment_counts.get(page_id.upper()),
            "actual": counts_by_page.get(page_id),
        }
        for page_id in page_keys
        if counts_by_page.get(page_id) != expected_segment_counts.get(page_id.upper())
    }
    report["segment_count_mismatches"] = mismatched

    provenance_missing = [
        record["id"]
        for record in records
        if not record["text_provenance"]["normalized_ocr_sha256"]
    ]
    report["records_missing_provenance"] = provenance_missing

    machine_labeled_ground_truth = [
        record["id"]
        for record in records
        if record["human_verified"]
        or (record["ground_truth"] is not None and record["page_reference"] is None)
    ]
    report["machine_labeled_as_ground_truth"] = machine_labeled_ground_truth

    # Count pages, not segments, for engine coverage.
    pages_with_tess = {r["page_id"] for r in records if "otsu" in r["ocr"]["tesseract"]}
    pages_with_paddle = {r["page_id"] for r in records if "otsu" in r["ocr"]["paddleocr"]}
    report["ocr_page_coverage"] = {
        "tesseract": len(pages_with_tess),
        "paddleocr": len(pages_with_paddle),
    }

    report["reference_pages"] = sorted({page.upper() for page in reference_pages})
    report["records_with_page_reference"] = sum(
        1 for record in records if record["page_reference"] is not None
    )

    file_scan = _scan_dataset_file(dataset_path, ids)
    report["file_scan"] = file_scan

    checks = {
        "unique_ids": not report["duplicate_ids"],
        "page_coverage_complete": not report["missing_pages"],
        "no_extra_pages": not report["extra_pages"],
        "segment_order_contiguous": not report["non_contiguous_pages"],
        "segment_counts_reconciled": not report["segment_count_mismatches"],
        "provenance_complete": not provenance_missing,
        "machine_not_ground_truth": not machine_labeled_ground_truth,
        "encoding_utf8_no_bom": not file_scan["has_bom"],
        "line_integrity": not file_scan["bad_lines"],
        "file_ids_match_build": file_scan["ids_match_build"],
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def build_canonical_dataset(
    repo_root: str | Path,
    *,
    segments_path: str | Path | None = None,
    pages_path: str | Path | None = None,
    tesseract_root: str | Path | None = None,
    paddleocr_root: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build, validate, and write the canonical dataset and its report."""
    from indicnlp_pipeline.ocr.production_runner import atomic_write_json

    root = Path(repo_root).resolve()
    segments_path = Path(segments_path) if segments_path else root / "data" / "nlp" / "segments.jsonl"
    pages_path = Path(pages_path) if pages_path else root / "data" / "nlp" / "pages.jsonl"
    tesseract_root = (
        Path(tesseract_root) if tesseract_root else root / "data" / "ocr_full" / "tesseract"
    )
    paddleocr_root = (
        Path(paddleocr_root)
        if paddleocr_root
        else root / "data" / "ocr_paddle_full" / "paddleocr"
    )
    ground_truth_path = (
        Path(ground_truth_path)
        if ground_truth_path
        else root / "data" / "ground_truth" / "ground_truth.jsonl"
    )
    config_path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    output_path = (
        Path(output_path) if output_path else root / "data" / "corpus" / "corpus_full.jsonl"
    )
    report_path = (
        Path(report_path)
        if report_path
        else root / "evaluation" / "reports" / "canonical_dataset_validation.json"
    )

    target_languages = _load_target_languages(config_path)
    segments = _read_jsonl(segments_path)
    pages = _read_jsonl(pages_path)
    if not segments:
        raise CanonicalDatasetError("NLP segment dataset contains no records.")
    if not pages:
        raise CanonicalDatasetError("NLP page dataset contains no records.")

    normalized_index: dict[str, dict[str, Any]] = {}
    for page in pages:
        page_key = str(page["page_id"]).upper()
        rel_path = page.get("normalized_ocr_path")
        if not rel_path:
            raise CanonicalDatasetError(
                f"Page '{page_key}' is missing normalized_ocr_path."
            )
        abs_path = root / rel_path
        normalized_index[page_key] = {
            "path": str(rel_path).replace("\\", "/"),
            "sha256": _sha256_file(abs_path) if abs_path.is_file() else "",
        }

    ocr_by_engine = {
        "tesseract": _load_page_ocr(tesseract_root, "tesseract", root),
        "paddleocr": _load_page_ocr(paddleocr_root, "paddleocr", root),
    }
    references = _load_references(ground_truth_path)

    records = build_records(
        segments, normalized_index, ocr_by_engine, references, target_languages
    )
    write_dataset(records, output_path, overwrite=overwrite)

    expected_pages = [str(page["page_id"]) for page in pages]
    expected_segment_counts = {
        str(page["page_id"]).upper(): int(page["segment_count"]) for page in pages
    }
    report = validate_dataset(
        records,
        dataset_path=Path(output_path).resolve(),
        expected_pages=expected_pages,
        expected_segment_counts=expected_segment_counts,
        target_languages=target_languages,
        reference_pages=[str(ref.get("page_id")) for ref in references.values()],
    )
    atomic_write_json(Path(report_path), report)
    logger.info(
        "Canonical dataset validation: passed={} records={}",
        report["passed"],
        report["record_count"],
    )
    return records, report
