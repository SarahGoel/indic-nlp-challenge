"""Run normalization, segmentation, language identification, and tokenization.

Processes the canonical corpus on the configured experimental page subset.
Each run writes a separate, traceable set of artifacts under outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from indicnlp_pipeline.normalization.language_identifier import identify_language
from indicnlp_pipeline.normalization.unicode_normalizer import normalize_text
from indicnlp_pipeline.segmentation.sentence_segmenter import segment_text
from indicnlp_pipeline.segmentation.tokenizer import tokenize_segment
from indicnlp_pipeline.utils.config_loader import (
    find_repository_root,
    load_config,
    resolve_config_paths,
)
from indicnlp_pipeline.utils.logger import configure_logger


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file without modifying its source."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank JSONL line at {path}:{line_number}"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )

            records.append(record)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL without a BOM."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write readable UTF-8 JSON without a BOM."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    """Calculate a file hash for reproducibility."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and segment the experimental Indic NLP corpus."
        )
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="First inclusive PDF page; defaults to config.yaml.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Last inclusive PDF page; defaults to config.yaml.",
    )
    parser.add_argument(
        "--input-corpus",
        type=Path,
        default=None,
        help="Input corpus JSONL; defaults to data/corpus/corpus.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base output directory; defaults to outputs/normalization_runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = find_repository_root()
    config = load_config()
    resolved_config = resolve_config_paths(config, repository_root)

    paths = resolved_config["paths"]
    configure_logger(paths["logs"])

    preprocessing = config.get("preprocessing", {})
    start_page = (
        args.start_page
        if args.start_page is not None
        else int(preprocessing["experimental_start_page"])
    )
    end_page = (
        args.end_page
        if args.end_page is not None
        else int(preprocessing["experimental_end_page"])
    )

    if start_page < 1 or end_page < start_page:
        logger.error(
            "Invalid page range: start_page={}, end_page={}",
            start_page,
            end_page,
        )
        return 2

    input_path = args.input_corpus
    if input_path is None:
        input_path = paths["corpus"] / "corpus.jsonl"
    elif not input_path.is_absolute():
        input_path = repository_root / input_path
    input_path = input_path.resolve()

    if not input_path.is_file():
        logger.error("Input corpus not found: {}", input_path)
        return 2

    output_root = args.output_root
    if output_root is None:
        output_root = paths["outputs"] / "normalization_runs"
    elif not output_root.is_absolute():
        output_root = repository_root / output_root
    output_root = output_root.resolve()

    # Timestamped run directories preserve all earlier artifacts.
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = output_root / run_id

    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.error("Run directory already exists: {}", run_directory)
        return 2

    logger.info("Step 30 pipeline started")
    logger.info("Repository root: {}", repository_root)
    logger.info("Input corpus: {}", input_path)
    logger.info("Experimental page range: {}–{}", start_page, end_page)
    logger.info("Run output directory: {}", run_directory)

    try:
        input_records = read_jsonl(input_path)
    except (OSError, ValueError) as exc:
        logger.exception("Unable to read the input corpus: {}", exc)
        return 1

    selected_records: list[dict[str, Any]] = []

    for record in input_records:
        page = record.get("page")
        if isinstance(page, int) and not isinstance(page, bool):
            if start_page <= page <= end_page:
                selected_records.append(record)

    if not selected_records:
        logger.error(
            "No corpus records found for PDF pages {}–{}.",
            start_page,
            end_page,
        )
        return 1

    ids = [record.get("id") for record in selected_records]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        logger.error("Selected records contain missing or invalid IDs.")
        return 1

    if len(set(ids)) != len(ids):
        logger.error("Selected records contain duplicate IDs.")
        return 1

    normalized_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    language_records: list[dict[str, Any]] = []
    token_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    empty_text_pages: list[str] = []

    for record in selected_records:
        corpus_id = record["id"]
        page = record["page"]
        source_page = record.get("source_page")
        annotation_id = record.get("segment_id", corpus_id)
        page_id = record.get(
            "ground_truth_metadata", {}
        ).get("page_id")

        if not page_id:
            page_id = Path(str(source_page or corpus_id)).stem

        try:
            original_text = record.get("ground_truth")
            if not isinstance(original_text, str):
                raise TypeError("ground_truth must be a string")

            if not original_text.strip():
                empty_text_pages.append(str(page_id))

            normalized_text = normalize_text(original_text)

            normalized_records.append(
                {
                    "corpus_id": corpus_id,
                    "annotation_id": annotation_id,
                    "page_id": page_id,
                    "page": page,
                    "source_page": source_page,
                    "source_language": record.get("source_language"),
                    "text_original": original_text,
                    "text_normalized": normalized_text,
                    "normalization": {
                        "unicode_form": "NFC",
                        "normalize_whitespace": True,
                        "remove_nul": True,
                    },
                }
            )

            segments = segment_text(normalized_text, page_id=page_id)

            for index, segment in enumerate(segments, start=1):
                # Make IDs unique to the source annotation, even if a page
                # eventually contains more than one annotation.
                segment_id = f"{annotation_id}::seg-{index:04d}"

                segment_record = {
                    "corpus_id": corpus_id,
                    "annotation_id": annotation_id,
                    "page_id": page_id,
                    "page": page,
                    "source_page": source_page,
                    "source_language": record.get("source_language"),
                    "segment_id": segment_id,
                    "segment_index": index,
                    "entry_number": segment.entry_number,
                    "boundary_after": segment.boundary_after,
                    "text": segment.text,
                }
                segment_records.append(segment_record)

                identification = identify_language(segment.text)
                language_records.append(
                    {
                        "corpus_id": corpus_id,
                        "annotation_id": annotation_id,
                        "page_id": page_id,
                        "page": page,
                        "source_page": source_page,
                        "segment_id": segment_id,
                        "language": identification.language,
                        "confidence": identification.confidence,
                        "hindi_markers": list(identification.hindi_markers),
                        "sanskrit_markers": list(
                            identification.sanskrit_markers
                        ),
                        "reason": identification.reason,
                    }
                )

                # The tokenizer requires the segment's source identifiers.
                from dataclasses import replace

                linked_segment = replace(
                    segment,
                    segment_id=segment_id,
                )

                tokens = tokenize_segment(linked_segment)

                for token in tokens:
                    token_records.append(
                        {
                            "corpus_id": corpus_id,
                            "annotation_id": annotation_id,
                            "page_id": page_id,
                            "page": page,
                            "source_page": source_page,
                            **token.to_dict(),
                        }
                    )

        except Exception as exc:
            logger.exception(
                "Processing failed for corpus_id={} page={}",
                corpus_id,
                page,
            )
            errors.append(
                {
                    "corpus_id": corpus_id,
                    "annotation_id": annotation_id,
                    "page_id": page_id,
                    "page": page,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    output_files = {
        "normalized": "normalized.jsonl",
        "segments": "segments.jsonl",
        "language_identification": "language_identification.jsonl",
        "tokens": "tokens.jsonl",
        "errors": "errors.jsonl",
    }

    try:
        write_jsonl(run_directory / output_files["normalized"], normalized_records)
        write_jsonl(run_directory / output_files["segments"], segment_records)
        write_jsonl(
            run_directory / output_files["language_identification"],
            language_records,
        )
        write_jsonl(run_directory / output_files["tokens"], token_records)
        write_jsonl(run_directory / output_files["errors"], errors)
    except OSError as exc:
        logger.exception("Unable to write pipeline artifacts: {}", exc)
        return 1

    boundary_counts = Counter(
        record["boundary_after"] for record in segment_records
    )
    language_counts = Counter(
        record["language"] for record in language_records
    )
    token_type_counts = Counter(
        record["token_type"] for record in token_records
    )

    expected_pages = sorted(
        {record["page"] for record in selected_records}
    )
    processed_ids = {record["corpus_id"] for record in normalized_records}
    failed_ids = {record["corpus_id"] for record in errors}

    validation = {
        "input_records_selected": len(selected_records),
        "normalized_records": len(normalized_records),
        "unique_input_ids": len(set(ids)),
        "records_processed": len(processed_ids),
        "records_failed": len(errors),
        "segments": len(segment_records),
        "language_identification_records": len(language_records),
        "tokens": len(token_records),
        "pages": expected_pages,
        "all_selected_records_accounted_for": (
            processed_ids | failed_ids == set(ids)
            and processed_ids.isdisjoint(failed_ids)
        ),
        "all_segments_nonempty": all(
            isinstance(record["text"], str) and record["text"].strip()
            for record in segment_records
        ),
        "all_tokens_have_valid_offsets": all(
            isinstance(record["start_char"], int)
            and isinstance(record["end_char"], int)
            and 0 <= record["start_char"] < record["end_char"]
            and record["end_char"] <= len(
                next(
                    segment["text"]
                    for segment in segment_records
                    if segment["segment_id"] == record["segment_id"]
                )
            )
            for record in token_records
        ),
        "segment_boundary_counts": dict(boundary_counts),
        "language_label_counts": dict(language_counts),
        "token_type_counts": dict(token_type_counts),
        "empty_text_pages": empty_text_pages,
        "passed": (
            len(errors) == 0
            and len(normalized_records) == len(selected_records)
            and len(set(ids)) == len(selected_records)
            and (processed_ids | failed_ids) == set(ids)
            and processed_ids.isdisjoint(failed_ids)
            and all(
                isinstance(record["text"], str) and record["text"].strip()
                for record in segment_records
            )
            and all(
                isinstance(record["start_char"], int)
                and isinstance(record["end_char"], int)
                and 0 <= record["start_char"] < record["end_char"]
                and record["end_char"] <= len(
                    next(
                        segment["text"]
                        for segment in segment_records
                        if segment["segment_id"] == record["segment_id"]
                    )
                )
                for record in token_records
            ),
        ),
    }

    metadata = {
        "step": 30,
        "script": "scripts/03_normalize_and_segment.py",
        "run_id": run_id,
        "started_at_utc": run_timestamp.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "repository_root": str(repository_root),
        "input_corpus": str(input_path),
        "input_corpus_sha256": sha256_file(input_path),
        "page_range_inclusive": {
            "start": start_page,
            "end": end_page,
        },
        "source_text_field": "ground_truth",
        "source_text_is_modified_in_place": False,
        "normalization": "Unicode NFC; horizontal whitespace and line endings normalized",
        "language_identifier": "lexical heuristic; confidence is not calibrated",
        "output_files": output_files,
        "validation": validation,
    }

    try:
        write_json(run_directory / "validation_report.json", validation)
        write_json(run_directory / "run_metadata.json", metadata)
    except OSError as exc:
        logger.exception("Unable to write run metadata: {}", exc)
        return 1

    logger.info(
        "Pipeline summary: records={}, segments={}, language labels={}, tokens={}, errors={}",
        len(normalized_records),
        len(segment_records),
        dict(language_counts),
        len(token_records),
        len(errors),
    )
    logger.info("Validation passed: {}", validation["passed"])
    logger.info("Artifacts: {}", run_directory)

    if validation["passed"]:
        logger.success("Step 30 pilot processing completed successfully.")
        return 0

    logger.error(
        "Step 30 validation failed. Inspect {} and {}.",
        run_directory / "errors.jsonl",
        run_directory / "validation_report.json",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
