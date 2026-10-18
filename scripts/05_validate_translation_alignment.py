import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth" / "ground_truth.jsonl"
TRANSLATIONS = ROOT / "data" / "translations"
REPORT = ROOT / "evaluation" / "reports" / "translation_alignment_report.json"


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}: blank line {line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
    return records


def main():
    source_records = load_jsonl(GROUND_TRUTH)
    source_by_id = {}
    duplicate_source_ids = []

    for record in source_records:
        annotation_id = record.get("annotation_id")
        if not annotation_id:
            duplicate_source_ids.append("<missing annotation_id>")
        elif annotation_id in source_by_id:
            duplicate_source_ids.append(annotation_id)
        else:
            source_by_id[annotation_id] = record

    files_checked = 0
    records_checked = 0
    issues = []

    for path in sorted(TRANSLATIONS.rglob("*.jsonl")):
        if "human" in path.relative_to(TRANSLATIONS).parts:
            continue

        files_checked += 1
        records = load_jsonl(path)
        seen_ids = set()

        for index, record in enumerate(records, 1):
            records_checked += 1
            annotation_id = record.get("annotation_id")
            location = f"{path.relative_to(ROOT)}:line {index}"

            if not annotation_id:
                issues.append(f"{location}: missing annotation_id")
                continue

            if annotation_id in seen_ids:
                issues.append(f"{location}: duplicate annotation_id {annotation_id}")
            seen_ids.add(annotation_id)

            source = source_by_id.get(annotation_id)
            if source is None:
                issues.append(f"{location}: ID not found in ground truth: {annotation_id}")
                continue

            if record.get("page_id") != source.get("page_id"):
                issues.append(f"{location}: page_id mismatch for {annotation_id}")

            if record.get("source_language") != source.get("language"):
                issues.append(f"{location}: source language mismatch for {annotation_id}")

            if record.get("source_text") != source.get("text"):
                issues.append(f"{location}: source text mismatch for {annotation_id}")

            if not record.get("target_language"):
                issues.append(f"{location}: missing target_language for {annotation_id}")

            if not str(record.get("translated_text", "")).strip():
                issues.append(f"{location}: empty translated_text for {annotation_id}")

    report = {
        "ground_truth_records": len(source_records),
        "unique_ground_truth_ids": len(source_by_id),
        "duplicate_or_missing_source_ids": duplicate_source_ids,
        "translation_files_checked": files_checked,
        "translation_records_checked": records_checked,
        "issue_count": len(issues) + len(duplicate_source_ids),
        "issues": issues,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport written to: {REPORT}")

    if report["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
