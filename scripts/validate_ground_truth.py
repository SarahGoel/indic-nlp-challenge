import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth" / "ground_truth.jsonl"
MANIFEST = ROOT / "data" / "raw_scans" / "page_manifest.csv"

REQUIRED = {
    "annotation_id", "page_id", "source_page", "language",
    "text", "annotation_status", "confidence", "annotator", "notes"
}
ALLOWED_LANGUAGES = {"hin", "san"}
ALLOWED_STATUSES = {"verified", "uncertain", "unreadable"}


def main() -> int:
    if not GROUND_TRUTH.is_file():
        print(f"ERROR: Missing {GROUND_TRUTH}")
        return 1

    if not MANIFEST.is_file():
        print(f"ERROR: Missing {MANIFEST}")
        return 1

    import csv

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))

    valid_page_ids = {
        row["page_id"].strip()
        for row in manifest_rows
        if row.get("page_id", "").strip()
    }

    errors = []
    records = []

    raw = GROUND_TRUTH.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("File contains a UTF-8 BOM.")

    content = raw.decode("utf-8")
    if not content.strip():
        print("ERROR: Ground-truth file is empty; no records validated.")
        return 1

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            errors.append(f"Line {line_number}: blank lines are not allowed.")
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_number}: invalid JSON: {exc}")
            continue

        if not isinstance(record, dict):
            errors.append(f"Line {line_number}: record must be a JSON object.")
            continue

        records.append((line_number, record))
        missing = REQUIRED - record.keys()
        if missing:
            errors.append(
                f"Line {line_number}: missing fields {sorted(missing)}"
            )
            continue

        for key in ("annotation_id", "page_id", "source_page",
                    "language", "text", "annotation_status", "annotator"):
            if not isinstance(record[key], str) or not record[key].strip():
                errors.append(f"Line {line_number}: {key} must be non-empty text.")

        if record["page_id"] not in valid_page_ids:
            errors.append(
                f"Line {line_number}: unknown page_id {record['page_id']!r}"
            )

        if record["language"] not in ALLOWED_LANGUAGES:
            errors.append(
                f"Line {line_number}: invalid language {record['language']!r}"
            )

        if record["annotation_status"] not in ALLOWED_STATUSES:
            errors.append(
                f"Line {line_number}: invalid annotation_status "
                f"{record['annotation_status']!r}"
            )

        confidence = record["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            errors.append(
                f"Line {line_number}: confidence must be a number from 0 to 1."
            )

        if record["annotation_status"] == "verified" and (
            not isinstance(record["text"], str)
            or "REPLACE_WITH_HUMAN_VERIFIED_TEXT" in record["text"]
        ):
            errors.append(
                f"Line {line_number}: verified record has placeholder text."
            )

    ids = [
        r.get("annotation_id") for _, r in records
        if isinstance(r.get("annotation_id"), str)
    ]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate annotation IDs: {duplicates}")

    segment_keys = [
        (r.get("page_id"), r.get("annotation_id"))
        for _, r in records
        if isinstance(r.get("page_id"), str)
        and isinstance(r.get("annotation_id"), str)
    ]
    if len(segment_keys) != len(set(segment_keys)):
        errors.append("Duplicate page/annotation ID pairs detected.")

    page_ids = {r.get("page_id") for _, r in records}
    languages = Counter(r.get("language") for _, r in records)
    statuses = Counter(r.get("annotation_status") for _, r in records)

    print(f"Records parsed: {len(records)}")
    print(f"Unique pages represented: {len(page_ids)}")
    print(f"Language counts: {dict(languages)}")
    print(f"Annotation status counts: {dict(statuses)}")
    print(f"Manifest pages: {len(valid_page_ids)}")

    print("\nRepresentative records:")
    for language in ("san", "hin"):
        match = next(
            (r for _, r in records if r.get("language") == language), None
        )
        if match:
            print(f"{language}: {json.dumps(match, ensure_ascii=False)}")
        else:
            print(f"{language}: no record available")

    numeric_records = [
        r for _, r in records
        if any(char.isdigit() for char in r.get("text", ""))
    ]
    print(f"Records containing digits: {len(numeric_records)}")

    if errors:
        print("\nVALIDATION FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nJSONL STRUCTURE VALIDATION PASSED.")
    print("Check source-page correspondence and transcription accuracy manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
