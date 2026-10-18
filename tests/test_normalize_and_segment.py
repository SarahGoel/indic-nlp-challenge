import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "03_normalize_and_segment.py"
)

SPEC = importlib.util.spec_from_file_location(
    "normalize_and_segment_script",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None

normalize_and_segment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalize_and_segment)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_pipeline_processes_temporary_corpus(monkeypatch, tmp_path):
    input_corpus = tmp_path / "input.jsonl"
    output_root = tmp_path / "runs"

    records = [
        {
            "id": "test-page-5",
            "page": 5,
            "source_page": "BRIGU_V5_P0005",
            "ground_truth": "यह एक परीक्षण है।",
        },
        {
            "id": "test-page-6",
            "page": 6,
            "source_page": "BRIGU_V5_P0006",
            "ground_truth": "संस्कृत परीक्षण ॥",
        },
    ]
    write_jsonl(input_corpus, records)

    monkeypatch.setattr(
        normalize_and_segment.sys,
        "argv",
        [
            "03_normalize_and_segment.py",
            "--start-page", "5",
            "--end-page", "6",
            "--input-corpus", str(input_corpus),
            "--output-root", str(output_root),
        ],
    )

    assert normalize_and_segment.main() == 0

    run_directories = list(output_root.iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]

    expected_files = [
        "normalized.jsonl",
        "segments.jsonl",
        "language_identification.jsonl",
        "tokens.jsonl",
        "errors.jsonl",
        "validation_report.json",
        "run_metadata.json",
    ]
    for filename in expected_files:
        assert (run_directory / filename).is_file(), filename

    validation = json.loads(
        (run_directory / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["input_records_selected"] == 2
    assert validation["normalized_records"] == 2
    assert validation["records_failed"] == 0
    assert validation["all_selected_records_accounted_for"] is True
    assert validation["all_segments_nonempty"] is True
    assert validation["all_tokens_have_valid_offsets"] is True
    assert validation["passed"] == [True]

    errors = (run_directory / "errors.jsonl").read_text(encoding="utf-8")
    assert errors == ""


def test_pipeline_rejects_invalid_page_range(monkeypatch):
    monkeypatch.setattr(
        normalize_and_segment.sys,
        "argv",
        [
            "03_normalize_and_segment.py",
            "--start-page", "0",
            "--end-page", "6",
        ],
    )

    assert normalize_and_segment.main() == 2
