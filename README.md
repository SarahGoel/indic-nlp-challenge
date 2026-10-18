# Indic NLP & Multilingual AI Challenge

A research-oriented pipeline for processing scanned Hindi and Sanskrit
manuscript pages into structured text and preparing them for multilingual
NLP and translation experiments.

## Project source

- Source: Shri Bhrigu Samhita Kundali Rahasyam – Volume 5
- Primary language: Hindi (`hin`)
- Secondary language: Sanskrit (`san`)
- Script: Devanagari
- Experimental development subset: PDF pages 5–14
- Evaluation subset: 100 pages; selection and evaluation are tracked separately

Raw scans and raw OCR outputs are source artifacts. Processing stages should
write separate outputs rather than silently overwriting upstream data.

## Environment

The current development environment uses Python 3.11 in a project-specific
virtual environment. Activate the environment before running commands.

From the repository root, install project dependencies according to the
repository's dependency instructions, then run tests with:

```powershell
python -m pytest -q
```

## Step 30 — Normalization and segmentation

### Purpose

`scripts/03_normalize_and_segment.py` reads the existing canonical corpus,
normalizes its ground-truth text, segments it, identifies likely language
labels, tokenizes segments, and writes run-specific outputs.

The script does not modify the input corpus in place. Each run writes to a
new timestamped directory under `outputs/normalization_runs/` by default.

### Run the experimental subset

From the repository root, with the project virtual environment activated:

```powershell
python scripts/03_normalize_and_segment.py
```

The default page range is read from `configs/config.yaml`. In the current
configuration, the experimental range is pages 5–14 inclusive.

To override the page range:

```powershell
python scripts/03_normalize_and_segment.py --start-page 5 --end-page 14
```

To specify an alternate input corpus and output directory:

```powershell
python scripts/03_normalize_and_segment.py `
    --start-page 5 `
    --end-page 14 `
    --input-corpus data/corpus/corpus.jsonl `
    --output-root outputs/normalization_runs
```

The input JSONL records must contain a unique string `id`, an integer `page`,
and a `ground_truth` text field.

### Run artifacts

Every run produces the following files:

| File | Purpose |
| --- | --- |
| `normalized.jsonl` | Normalized text records |
| `segments.jsonl` | Text segments with traceable identifiers and boundaries |
| `language_identification.jsonl` | Language label and identification details per segment |
| `tokens.jsonl` | Tokens with segment references and character offsets |
| `errors.jsonl` | Per-record processing errors, when any occur |
| `validation_report.json` | Counts and validation checks for the run |
| `run_metadata.json` | Input hash, run times, environment, page range, and output names |

Run-specific artifacts should be inspected before a run is accepted. Check
that selected records are accounted for, segment text is nonempty, token
offsets are valid, and `errors.jsonl` contains no unexpected failures.

### Language-identification limitation

The current language identifier is a lexical heuristic. Its labels and
confidence values are not equivalent to predictions from a trained,
calibrated language-identification model. Mixed and unknown labels should be
preserved rather than automatically forced into Hindi or Sanskrit.

### Verified pilot result

The pilot run processed PDF pages 5–14 inclusive:

- 10 selected input records and 10 normalized records
- 218 text segments
- 218 language-identification records
- 3,491 tokens
- 0 recorded processing errors
- All selected records accounted for
- Nonempty segments and valid token offsets reported by validation
- Input corpus SHA-256 recorded in `run_metadata.json`

The full test suite passed with 86 tests during this development session.
The two focused tests for the Step 30 executable also passed after the test
fixture was aligned with the script's input schema.

These results establish a successful pilot execution; they do not establish
that segmentation or language labels are linguistically correct for every
manuscript passage.

## Data integrity and reproducibility

- Keep `data/raw_scans/` immutable.
- Preserve raw OCR outputs separately from corrected text.
- Do not treat normalized text as a replacement for ground truth.
- Keep each pipeline run's artifacts and metadata together.
- Record validation failures and investigate them before using outputs for
  downstream translation or evaluation.
- Treat pilot pages as development data, not as a representative evaluation
  sample without further assessment.

## Project documentation

- `data/raw_scans/SOURCE_METADATA.md` — source metadata
- `data/raw_scans/PAGE_ID_CONVENTION.md` — page identifier convention
- `data/raw_scans/SUBSET_PLAN.md` — development and evaluation subset plan
- `data/ground_truth/ANNOTATION_GUIDELINES.md` — ground-truth annotation guidance
- `data/ocr_corrected/CORRECTION_PROTOCOL.md` — OCR correction protocol
- `data/corpus/SCHEMA.md` — canonical corpus schema
