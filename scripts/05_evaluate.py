"""Run OCR, translation, and semantic evaluation for the project."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth" / "ground_truth.jsonl"
OCR_ROOT = ROOT / "data" / "ocr_raw"
TRANSLATIONS_ROOT = ROOT / "data" / "translations"
METRICS_DIR = ROOT / "evaluation" / "metrics"

from indicnlp_pipeline.evaluation.metrics import (  # noqa: E402
    calculate_bleu,
    calculate_chrf,
    calculate_semantic_similarity,
)
from indicnlp_pipeline.evaluation.ocr_evaluator import (  # noqa: E402
    evaluate_ocr_outputs,
)

TRANSLATION_COLUMNS = [
    "annotation_id",
    "page_id",
    "source_page",
    "source_language",
    "target_language",
    "engine",
    "model",
    "translation_status",
    "input_file",
    "reference_type",
    "reference_language",
    "bleu",
    "chrf",
    "evaluation_status",
    "evaluation_note",
]

SEMANTIC_COLUMNS = [
    "annotation_id",
    "page_id",
    "source_page",
    "source_language",
    "target_language",
    "engine",
    "model",
    "translation_status",
    "input_file",
    "comparison",
    "metric",
    "metric_model",
    "semantic_similarity",
    "evaluation_status",
    "evaluation_note",
]

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and report malformed input with its location."""
    records = []

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
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


def load_ground_truth() -> dict[str, dict[str, Any]]:
    """Load verified annotations indexed by stable annotation ID."""
    if not GROUND_TRUTH_PATH.is_file():
        raise FileNotFoundError(
            f"Ground-truth file not found: {GROUND_TRUTH_PATH}"
        )

    result = {}

    for record in load_jsonl(GROUND_TRUTH_PATH):
        annotation_id = record.get("annotation_id")

        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("A ground-truth record has no valid annotation_id.")

        if annotation_id in result:
            raise ValueError(
                f"Duplicate ground-truth annotation_id: {annotation_id}"
            )

        if record.get("annotation_status") == "verified":
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Verified annotation {annotation_id} has invalid text."
                )
            result[annotation_id] = record

    if not result:
        raise ValueError("No verified ground-truth annotations were found.")

    logger.info("Loaded {} verified ground-truth annotations.", len(result))
    return result


def load_human_references() -> dict[tuple[str, str], dict[str, Any]]:
    """Index available human translations by annotation ID and target language."""
    references = {}

    human_root = TRANSLATIONS_ROOT / "human"
    if not human_root.is_dir():
        logger.info("No human translation directory exists; BLEU/chrF will be unavailable.")
        return references

    for path in sorted(human_root.rglob("*.jsonl")):
        for record in load_jsonl(path):
            annotation_id = record.get("annotation_id")
            target_language = record.get("target_language")
            translated_text = record.get("translated_text")

            # Support human-reference files that use "text" for the translation.
            if translated_text is None:
                translated_text = record.get("text")

            if not isinstance(annotation_id, str) or not annotation_id:
                raise ValueError(
                    f"Missing annotation_id in human reference file {path}."
                )
            if not isinstance(target_language, str) or not target_language:
                raise ValueError(
                    f"Missing target_language in human reference file {path}."
                )
            if not isinstance(translated_text, str):
                raise ValueError(
                    f"Missing human reference text in {path} for {annotation_id}."
                )

            key = (annotation_id, target_language)
            if key in references:
                raise ValueError(
                    "Duplicate human reference for annotation/target pair: "
                    f"{annotation_id}/{target_language}"
                )

            references[key] = {
                "text": translated_text,
                "language": target_language,
                "path": path,
            }

    logger.info("Loaded {} aligned human translation references.", len(references))
    return references


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write CSV data as UTF-8 without overwriting unrelated outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote {} rows to {}", len(rows), path.relative_to(ROOT))


def evaluate_translations(
    ground_truth: dict[str, dict[str, Any]],
    references: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate reference-based and source/translation semantic metrics."""
    if not TRANSLATIONS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Translation directory not found: {TRANSLATIONS_ROOT}"
        )

    translation_files = [
        path
        for path in sorted(TRANSLATIONS_ROOT.rglob("*.jsonl"))
        if "human" not in path.relative_to(TRANSLATIONS_ROOT).parts
    ]

    if not translation_files:
        raise ValueError(
            f"No machine translation JSONL files found under {TRANSLATIONS_ROOT}."
        )

    translation_rows = []
    semantic_rows = []
    semantic_model_error = None
    records_seen = set()

    for path in translation_files:
        for record in load_jsonl(path):
            annotation_id = record.get("annotation_id")
            if not isinstance(annotation_id, str) or not annotation_id:
                raise ValueError(f"Missing annotation_id in {path}.")

            key_for_record = (str(path), annotation_id)
            if key_for_record in records_seen:
                raise ValueError(
                    f"Duplicate annotation record in {path}: {annotation_id}"
                )
            records_seen.add(key_for_record)

            source = ground_truth.get(annotation_id)
            if source is None:
                logger.warning(
                    "Skipping translation without verified ground truth: {} ({})",
                    annotation_id,
                    path.relative_to(ROOT),
                )
                continue

            target_language = record.get("target_language")
            source_language = record.get("source_language")
            translated_text = record.get("translated_text")
            source_text = record.get("source_text")

            if not isinstance(target_language, str) or not target_language:
                raise ValueError(
                    f"Missing target_language for {annotation_id} in {path}."
                )
            if not isinstance(translated_text, str):
                raise ValueError(
                    f"Invalid translated_text for {annotation_id} in {path}."
                )

            base = {
                "annotation_id": annotation_id,
                "page_id": record.get("page_id", source.get("page_id", "")),
                "source_page": record.get(
                    "source_page", source.get("source_page", source.get("page_id", ""))
                ),
                "source_language": source_language or source.get("language", ""),
                "target_language": target_language,
                "engine": record.get("engine", ""),
                "model": record.get("model", ""),
                "translation_status": record.get("translation_status", ""),
                "input_file": path.relative_to(ROOT).as_posix(),
            }

            alignment_ok = (
                record.get("page_id") == source.get("page_id")
                and source_language == source.get("language")
                and source_text == source.get("text")
            )

            reference = references.get((annotation_id, target_language))
            translation_status = str(record.get("translation_status", ""))

            translation_row = {
                **base,
                "reference_type": "human" if reference else "",
                "reference_language": reference["language"] if reference else "",
                "bleu": "",
                "chrf": "",
                "evaluation_status": "",
                "evaluation_note": "",
            }

            if not alignment_ok:
                translation_row["evaluation_status"] = "source_alignment_mismatch"
                translation_row["evaluation_note"] = (
                    "Source page, language, or text does not exactly match "
                    "the verified ground-truth annotation."
                )
            elif not translated_text.strip():
                translation_row["evaluation_status"] = "translation_empty"
                translation_row["evaluation_note"] = (
                    "No non-empty translated text is available for evaluation."
                )
            elif reference is None:
                translation_row["evaluation_status"] = "reference_unavailable"
                translation_row["evaluation_note"] = (
                    "BLEU and chrF require an aligned human reference in the "
                    "target language; no matching reference was found."
                )
            elif not reference["text"].strip():
                translation_row["evaluation_status"] = "reference_empty"
                translation_row["evaluation_note"] = (
                    "The aligned human reference is empty; scores were not calculated."
                )
            else:
                translation_row["bleu"] = calculate_bleu(
                    reference["text"], translated_text
                )
                translation_row["chrf"] = calculate_chrf(
                    reference["text"], translated_text
                )
                translation_row["evaluation_status"] = "evaluated"
                translation_row["evaluation_note"] = (
                    "Scores are calculated against the aligned human target-language reference."
                )

            translation_rows.append(translation_row)

            semantic_row = {
                **base,
                "comparison": "source_text_vs_translated_text",
                "metric": "cosine_similarity",
                "metric_model": SEMANTIC_MODEL_NAME,
                "semantic_similarity": "",
                "evaluation_status": "",
                "evaluation_note": "",
            }

            if not alignment_ok:
                semantic_row["evaluation_status"] = "source_alignment_mismatch"
                semantic_row["evaluation_note"] = (
                    "Source alignment failed; semantic similarity was not calculated."
                )
            elif not translated_text.strip():
                semantic_row["evaluation_status"] = "translation_empty"
                semantic_row["evaluation_note"] = (
                    "No non-empty translated text is available for evaluation."
                )
            elif semantic_model_error is not None:
                semantic_row["evaluation_status"] = "semantic_model_unavailable"
                semantic_row["evaluation_note"] = semantic_model_error
            else:
                try:
                    semantic_row["semantic_similarity"] = (
                        calculate_semantic_similarity(source["text"], translated_text)
                    )
                    semantic_row["evaluation_status"] = (
                        "evaluated_source_translation_similarity"
                    )
                    semantic_row["evaluation_note"] = (
                        "Cross-lingual diagnostic only; not a direct measure "
                        "of translation correctness."
                    )
                except Exception as exc:
                    semantic_model_error = (
                        f"Local semantic model unavailable or evaluation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.warning("{}", semantic_model_error)
                    semantic_row["evaluation_status"] = "semantic_model_unavailable"
                    semantic_row["evaluation_note"] = semantic_model_error

            semantic_rows.append(semantic_row)

    if not translation_rows:
        raise ValueError(
            "No machine translation records matched verified ground truth."
        )

    return translation_rows, semantic_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR outputs and machine translations."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the three dedicated Step 48 output CSVs.",
    )
    args = parser.parse_args()

    outputs = {
        "ocr": METRICS_DIR / "ocr_metrics_step48.csv",
        "translation": METRICS_DIR / "translation_metrics_step48.csv",
        "semantic": METRICS_DIR / "semantic_metrics_step48.csv",
    }

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        logger.error(
            "Output file(s) already exist; refusing to overwrite: {}. "
            "Inspect them or rerun with --overwrite.",
            ", ".join(str(path.relative_to(ROOT)) for path in existing),
        )
        return 2

    try:
        ground_truth = load_ground_truth()
        references = load_human_references()

        ocr_count = evaluate_ocr_outputs(
            OCR_ROOT,
            GROUND_TRUTH_PATH,
            outputs["ocr"],
        )

        translation_rows, semantic_rows = evaluate_translations(
            ground_truth,
            references,
        )
        write_csv(
            outputs["translation"],
            TRANSLATION_COLUMNS,
            translation_rows,
        )
        write_csv(
            outputs["semantic"],
            SEMANTIC_COLUMNS,
            semantic_rows,
        )

        logger.success(
            "Evaluation complete: {} OCR rows, {} translation rows, "
            "{} semantic rows.",
            ocr_count,
            len(translation_rows),
            len(semantic_rows),
        )
        return 0

    except Exception:
        logger.exception("Evaluation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
