# Pipeline Setup Guide

## Quick Start

Process any PDF manuscript through the complete pipeline:

```bash
python scripts/run_pipeline.py --input manuscript.pdf --target-language ben
```

## Pipeline Steps

The pipeline automatically executes these steps:

1. **Page Extraction** - Converts PDF to PNG images
2. **OCR Processing** - Extracts text using Tesseract OCR
3. **Normalization** - Cleans and normalizes text
4. **Translation** - Translates to target language
5. **DOCX Export** - Generates Word document

## Target Languages

Choose from: ben, guj, mar, taM, tel, kan, mal, pan, asm, ory, urd

## Installation

```bash
pip install -r requirements.txt
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
- transformers (translation)
- IndicTransToolkit (Indic translation)

## Windows Users

IndicTransToolkit requires Microsoft C++ Build Tools for compilation:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Install tools, then:
```bash
pip install IndicTransToolkit
```
