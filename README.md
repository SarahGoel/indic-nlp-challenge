# Indic NLP Pipeline

[![Live on Render](https://img.shields.io/badge/Render-Live-46E3B7?logo=render&logoColor=white)](https://indic-nlp-challenge.onrender.com/)

A pipeline for processing scanned manuscripts into structured text and preparing them for multilingual translation.

## Live Deployment

The FastAPI web app in this repository is deployed and publicly reachable at:

**https://indic-nlp-challenge.onrender.com/**

| Setting | Value |
| --- | --- |
| Platform | [Render](https://render.com) Web Service |
| Runtime | Python 3 (pinned by `.python-version`) |
| Branch | `main` |
| Build Command | `pip install fastapi "uvicorn[standard]" python-multipart google-genai python-dotenv loguru` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | *(unset — the app has no `/healthz` route)* |
| Required environment variable | `GEMINI_API_KEY` |

> The service runs on Render's free instance tier: it spins down after roughly 15 minutes
> of inactivity and takes a few seconds to cold-start on the next request.

The build command deliberately installs only the web app's runtime dependencies. The full
`requirements.txt` describes the offline OCR/translation pipeline and pulls in heavy
packages (transformers, sentence-transformers, scipy, opencv, jupyterlab) that are not
needed to serve the web app.

## Web Application

`app.py` serves the FastAPI backend together with the static UI in `frontend/`:

- `GET /` — upload UI (`frontend/index.html`)
- `POST /api/process` — multipart form with `file`, `source_language`, and `target_language`;
  runs Gemini vision OCR on the uploaded image, then translates the extracted text with
  `GeminiTranslationEngine`. Returns `{ "transcription": ..., "translation": ... }`.

Run it locally:

```bash
pip install fastapi "uvicorn[standard]" python-multipart google-genai python-dotenv loguru
uvicorn app:app --host 0.0.0.0 --port 8000
```

`GEMINI_API_KEY` must be available in the process environment or in a `.env` file at the
repository root (see `.env.example`).

## Features

- PDF to page image extraction
- OCR text extraction (Hindi/Sanskrit/Devanagari)
- Text normalization and cleaning
- Translation to 11 target languages
- DOCX export for OCR and translated text

## Supported Languages

**Source:**
- Hindi (hin)
- Sanskrit (san)

**Target Languages:**
- Bengali (ben)
- Gujarati (guj)
- Marathi (mar)
- Tamil (tam)
- Telugu (tel)
- Kannada (kan)
- Malayalam (mal)
- Punjabi (pan)
- Assamese (asm)
- Odia (ory)
- Urdu (urd)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline on any PDF:

```bash
python scripts/run_pipeline.py --input manuscript.pdf --target-language ben
```

This will:
1. Extract pages from PDF
2. Run OCR on all pages
3. Normalize text
4. Translate to target language
5. Export translated text to DOCX

To also export the original OCR text:

```bash
python scripts/run_pipeline.py --input manuscript.pdf --target-language tam --export-ocr
```

## Individual Steps

### OCR Export
Export original OCR text to DOCX without translation:

```bash
python scripts/17_export_ocr_docx.py
```

### Translation
Translate OCR text to a target language and export to DOCX:

```bash
python scripts/18_translate_and_export.py --target-language ben
```

## Pipeline Output

All outputs are written to the specified output directory (default: `pipeline_output/`):

- `pages/` - Extracted page images
- `ocr/` - OCR results (JSON)
- `normalized/` - Normalized text
- `translated/` - Translated text (JSON)
- `translated_<language>.docx` - Final DOCX output

## Dependencies

- pypdfium2 - PDF processing
- pytesseract - OCR engine
- Pillow - Image processing
- transformers - Translation model
- IndicTransToolkit - Indic language translation
- loguru - Logging

## Translation Requirements

IndicTransToolkit is required for translation. Installation requires compilation tools.

**On Windows:**
Install Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**On Linux:**
```bash
pip install IndicTransToolkit
```

## License

See LICENSE file for details.

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

| File                            | Purpose                                                          |
| ------------------------------- | ---------------------------------------------------------------- |
| `normalized.jsonl`              | Normalized text records                                          |
| `segments.jsonl`                | Text segments with traceable identifiers and boundaries          |
| `language_identification.jsonl` | Language label and identification details per segment            |
| `tokens.jsonl`                  | Tokens with segment references and character offsets             |
| `errors.jsonl`                  | Per-record processing errors, when any occur                     |
| `validation_report.json`        | Counts and validation checks for the run                         |
| `run_metadata.json`             | Input hash, run times, environment, page range, and output names |

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

The full test suite passed with 222 tests (`python -m pytest -q`), with 0 failures in the latest verification run.
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
