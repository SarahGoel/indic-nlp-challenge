"""Generate reproducible OCR and translation error logs.

The canonical CSV schema is defined by ERROR_TAXONOMY.md.
Raw source files are read-only inputs. Use --write to create output files;
without it, the script performs a dry run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from indicnlp_pipeline.error_analysis.classifier import (  # noqa: E402
    ERROR_RECORD_FIELDS,
    classify_error,
    make_error_id,
)

ERROR_DIR = ROOT / "evaluation" / "error_logs"
COMPARISON_DIR = ERROR_DIR / "ocr_comparisons"
CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.jsonl"
TRANSLATION_DIRS = {
    "indictrans2": ROOT / "data" / "translations" / "indictrans2",
    "gemini": ROOT / "data" / "translations" / "gemini",
}
OUTPUTS = {
    "ocr": ERROR_DIR / "ocr_errors.csv",
    "translation": ERROR_DIR / "translation_errors.csv",
}
PROVENANCE_PATH = ERROR_DIR / "error_log_provenance.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL without modifying the input file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )
            records.append(record)
    return records


def normalized_for_comparison(text: str) -> str:
    """Normalize whitespace only for comparison; preserve original output."""
    return " ".join(text.split())


def severity_for(reference: str, output: str) -> str:
    """Assign a reproducible heuristic severity from character similarity."""
    reference_cmp = normalized_for_comparison(reference)
    output_cmp = normalized_for_comparison(output)

    if not reference_cmp:
        return "high"

    similarity = SequenceMatcher(
        None, reference_cmp, output_cmp, autojunk=True
    ).ratio()

    if similarity >= 0.90:
        return "low"
    if similarity >= 0.70:
        return "medium"
    if similarity >= 0.40:
        return "high"
    return "critical"


def page_key(value: Any) -> str | None:
    """Convert known page identifiers or filenames to a page ID."""
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip()).stem


def load_verified_references() -> dict[str, dict[str, Any]]:
    """Index verified ground-truth corpus records by source page."""
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    references: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(CORPUS_PATH):
        metadata = record.get("ground_truth_metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("annotation_status") != "verified":
            continue
        if record.get("human_verified") is not True:
            continue

        reference = record.get("ground_truth")
        if not isinstance(reference, str) or not reference.strip():
            continue

        keys = {
            page_key(metadata.get("page_id")),
            page_key(record.get("source_page")),
        }
        for key in keys - {None}:
            if key in references:
                previous = references[key]
                if previous.get("ground_truth") != reference:
                    raise ValueError(
                        "Multiple different verified references map to "
                        f"page {key!r}; refusing ambiguous OCR comparison."
                    )
            references[key] = record

    return references


def make_record(
    *,
    segment_id: str,
    stage: str,
    category: str,
    severity: str,
    system_output: str,
    reference_text: str,
) -> dict[str, str]:
    """Build one row in the canonical CSV schema."""
    error_id = make_error_id(
        segment_id,
        stage,
        category,
        system_output,
        reference_text,
    )
    return {
        "error_id": error_id,
        "segment_id": segment_id,
        "stage": stage,
        "category": category,
        "system_output": system_output,
        "reference_text": reference_text,
        "severity": severity,
    }


def build_ocr_errors(
    references: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    """Compare OCR output against verified reference text."""
    rows: list[dict[str, str]] = []
    provenance: list[dict[str, Any]] = []
    stats = {
        "comparison_files": 0,
        "matched_reference_files": 0,
        "unmatched_reference_files": 0,
        "unaligned_reference_files": 0,
        "invalid_or_missing_engine_texts": 0,
        "matching_outputs": 0,
        "mismatching_engine_outputs": 0,
    }

    if not COMPARISON_DIR.is_dir():
        return rows, provenance, stats

    seen_ids: set[str] = set()

    for path in sorted(COMPARISON_DIR.glob("*.json")):
        stats["comparison_files"] += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read OCR comparison {path}: {exc}") from exc

        if not isinstance(data, dict):
            continue

        key = page_key(data.get("source_page"))
        reference_record = references.get(key) if key else None
        if reference_record is None:
            stats["unmatched_reference_files"] += 1
            continue

        stats["matched_reference_files"] += 1
        comparison = data.get("comparison")
        if not isinstance(comparison, dict):
            continue

        reference_segment_id = str(
            reference_record.get("segment_id")
            or reference_record.get("id")
            or ""
        )
        comparison_segment_id = data.get("segment_id")
        if comparison_segment_id is None:
            comparison_segment_id = comparison.get("segment_id")

        if (
            not isinstance(comparison_segment_id, str)
            or comparison_segment_id != reference_segment_id
        ):
            stats["unaligned_reference_files"] += 1
            continue

        reference = reference_record["ground_truth"]
        segment_id = str(
            reference_record.get("segment_id")
            or reference_record.get("id")
            or key
        )
        variant = data.get("preprocessing_variant", "unknown")

        for engine, field in (
            ("tesseract", "tesseract_text"),
            ("paddleocr", "paddleocr_text"),
        ):
            output = comparison.get(field)
            if not isinstance(output, str):
                # Do not flatten array-based OCR payloads implicitly.
                stats["invalid_or_missing_engine_texts"] += 1
                continue

            if (
                normalized_for_comparison(output)
                == normalized_for_comparison(reference)
            ):
                stats["matching_outputs"] += 1
                continue

            stats["mismatching_engine_outputs"] += 1
            severity = severity_for(reference, output)
            classification = classify_error(
                stage="ocr",
                evidence={
                    "rationale": (
                        "OCR output differs from verified corpus ground truth."
                    )
                },
                category="ocr",
                severity=severity,
            )

            # Include engine and preprocessing variant in the deterministic
            # ID input while preserving the canonical CSV columns.
            id_output = (
                f"[engine={engine}; variant={variant}]\n{output}"
            )
            error_id = make_error_id(
                segment_id, "ocr", "ocr", id_output, reference
            )
            if error_id in seen_ids:
                continue
            seen_ids.add(error_id)

            rows.append({
                "error_id": error_id,
                "segment_id": segment_id,
                "stage": classification.stage.value,
                "category": classification.category.value,
                "system_output": output,
                "reference_text": reference,
                "severity": classification.severity.value,
            })
            reference_cmp = normalized_for_comparison(reference)
            output_cmp = normalized_for_comparison(output)
            similarity = SequenceMatcher(
                None, reference_cmp, output_cmp, autojunk=True
            ).ratio()

            provenance.append({
                "error_id": error_id,
                "source_page": data.get("source_page"),
                "engine": engine,
                "preprocessing_variant": variant,
                "source_comparison_file": path.relative_to(ROOT).as_posix(),
                "reference_characters": len(reference_cmp),
                "output_characters": len(output_cmp),
                "character_similarity": round(similarity, 6),
                "classification_basis": (
                    "OCR output differs from verified corpus ground truth."
                ),
                "severity_method": (
                    "Heuristic based on character similarity; "
                    "not a human-validated severity assessment."
                ),
                "requires_manual_review": True,
            })

    return rows, provenance, stats


def build_translation_errors() -> tuple[
    list[dict[str, str]], list[dict[str, Any]], dict[str, int]
]:
    """Compare available system translations to actual human references."""
    rows: list[dict[str, str]] = []
    provenance: list[dict[str, Any]] = []
    stats = {
        "translation_files": 0,
        "translation_records": 0,
        "matched_human_references": 0,
        "missing_human_references": 0,
        "mismatching_translations": 0,
    }

    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")

    corpus_records = read_jsonl(CORPUS_PATH)
    by_annotation: dict[str, dict[str, Any]] = {}
    for record in corpus_records:
        metadata = record.get("ground_truth_metadata")
        annotation_id = (
            metadata.get("annotation_id")
            if isinstance(metadata, dict)
            else None
        )
        annotation_id = annotation_id or record.get("segment_id")
        if isinstance(annotation_id, str):
            by_annotation[annotation_id] = record

    seen_ids: set[str] = set()

    for engine, directory in TRANSLATION_DIRS.items():
        if not directory.is_dir():
            continue

        for path in sorted(directory.rglob("*.jsonl")):
            stats["translation_files"] += 1
            for translation in read_jsonl(path):
                stats["translation_records"] += 1
                annotation_id = translation.get("annotation_id")
                target = translation.get("target_language")
                output = translation.get("translated_text")

                if (
                    not isinstance(annotation_id, str)
                    or not isinstance(target, str)
                    or not isinstance(output, str)
                    or not output.strip()
                ):
                    continue

                corpus_record = by_annotation.get(annotation_id)
                if corpus_record is None:
                    stats["missing_human_references"] += 1
                    continue

                translations = corpus_record.get("translations")
                language_slots = (
                    translations.get(target)
                    if isinstance(translations, dict)
                    else None
                )
                human_reference = (
                    language_slots.get("human")
                    if isinstance(language_slots, dict)
                    else None
                )

                if (
                    not isinstance(human_reference, str)
                    or not human_reference.strip()
                ):
                    stats["missing_human_references"] += 1
                    continue

                stats["matched_human_references"] += 1
                if normalized_for_comparison(
                    output
                ) == normalized_for_comparison(human_reference):
                    continue

                stats["mismatching_translations"] += 1
                segment_id = str(
                    corpus_record.get("segment_id") or annotation_id
                )
                severity = severity_for(human_reference, output)
                classification = classify_error(
                    stage="translation",
                    evidence={
                        "rationale": (
                            "System translation differs from the supplied "
                            "human reference."
                        )
                    },
                    category="translation",
                    severity=severity,
                )
                error_id = make_error_id(
                    segment_id,
                    "translation",
                    "translation",
                    output,
                    human_reference,
                )

                if error_id in seen_ids:
                    continue
                seen_ids.add(error_id)

                rows.append({
                    "error_id": error_id,
                    "segment_id": segment_id,
                    "stage": classification.stage.value,
                    "category": classification.category.value,
                    "system_output": output,
                    "reference_text": human_reference,
                    "severity": classification.severity.value,
                })
                provenance.append({
                    "error_id": error_id,
                    "source_annotation_id": annotation_id,
                    "target_language": target,
                    "engine": engine,
                    "source_translation_file": path.relative_to(ROOT).as_posix(),
                    "classification_basis": (
                        "System translation differs from supplied human reference."
                    ),
                    "severity_method": (
                        "Character similarity heuristic; manually review."
                    ),
                })

    return rows, provenance, stats


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write canonical UTF-8 CSV without a BOM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(ERROR_RECORD_FIELDS),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write CSV logs and provenance manifest; default is dry run.",
    )
    args = parser.parse_args()

    references = load_verified_references()
    ocr_rows, ocr_provenance, ocr_stats = build_ocr_errors(references)
    translation_rows, translation_provenance, translation_stats = (
        build_translation_errors()
    )

    print("Error log generator — dry run" if not args.write else "Error log generator — write mode")
    print(f"Verified reference pages indexed: {len(references)}")
    print("\nOCR:")
    for key, value in ocr_stats.items():
        print(f"  {key}: {value}")
    print(f"  candidate CSV rows: {len(ocr_rows)}")

    print("\nTranslation:")
    for key, value in translation_stats.items():
        print(f"  {key}: {value}")
    print(f"  candidate CSV rows: {len(translation_rows)}")

    if not args.write:
        print("\nDry run only. No output files were changed.")
        print("Review the counts before rerunning with --write.")
        return 0

    write_csv(OUTPUTS["ocr"], ocr_rows)
    write_csv(OUTPUTS["translation"], translation_rows)

    all_provenance = ocr_provenance + translation_provenance
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROVENANCE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_provenance:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    print("\nFiles written:")
    for path in (*OUTPUTS.values(), PROVENANCE_PATH):
        print(f"  {path.relative_to(ROOT).as_posix()}")
    print("Note: severity is a heuristic; review classifications before reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
