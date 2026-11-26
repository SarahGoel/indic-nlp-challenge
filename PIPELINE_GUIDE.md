# Pipeline Setup Guide

## Quick Start

Process any PDF manuscript through the complete pipeline:

```bash
# Set Gemini API key first
export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
# or
set GEMINI_API_KEY=your_api_key_here  # Windows

# Run pipeline
python scripts/run_pipeline.py --input manuscript.pdf --target-language ben
```

## Pipeline Steps

The pipeline automatically executes these steps:

1. **Page Extraction** - Converts PDF to PNG images
2. **OCR Processing** - Extracts text using Tesseract OCR
3. **Normalization** - Cleans and normalizes text
4. **Translation** - Translates to target language using Gemini API
5. **DOCX Export** - Generates Word document

## Target Languages

Choose from: ben, guj, mar, tam, tel, kan, mal, pan, asm, ory, urd

## Installation

```bash
pip install -r requirements.txt
pip install google-generativeai
```

## OCR-Only Export

To export OCR text without translation:

```bash
python scripts/17_export_ocr_docx.py
```

## Translation-Only Export

To translate existing OCR output:

```bash
python scripts/18_translate_and_export.py --target-language ben
```

## Output Structure

```
pipeline_output/
├── pages/              # Extracted page images
├── ocr/                # OCR results (JSON)
├── normalized/         # Normalized text
├── translated/         # Translated text (JSON)
└── translated_<lang>.docx  # Final DOCX output
```

## Requirements

- Python 3.11+
- pypdfium2 (PDF processing)
- pytesseract (OCR)
- Pillow (images)
- google-generativeai (Gemini API)
- Gemini API key (set as GEMINI_API_KEY environment variable)
