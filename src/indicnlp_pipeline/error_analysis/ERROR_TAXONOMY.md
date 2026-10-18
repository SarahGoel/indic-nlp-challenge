# Four-Category Error Analysis Taxonomy

## Purpose
Provide a consistent, evidence-based classification framework for the Indic NLP pipeline.

## Canonical categories

- `ocr`: Character confusion, matra loss, conjunct breakdown, diacritic loss, spacing errors, and danda corruption.
- `translation`: Omission, unsupported addition, mistranslation, and grammatical distortion.
- `terminology`: Incorrect interpretation of classical astrological, philosophical, or proper-name terminology.
- `segmentation`: Broken verse lines, merged shlokas, incorrect sentence boundaries, and entry-boundary errors.

## Severity levels

- `low`: Minor defect with negligible impact on interpretation.
- `medium`: Localized defect affecting readability or a limited detail.
- `high`: Substantial defect affecting meaning, important terminology, or text structure.
- `critical`: Severe defect making a segment unusable or fundamentally changing its intended meaning.

Severity reflects impact, not category alone.

## Pipeline stages

- `ocr`
- `normalization`
- `segmentation`
- `translation`

A stage describes where an error was observed. A category describes its nature. For example, a terminology error may occur during translation.

## Required error record fields

`error_id`, `segment_id`, `stage`, `category`, `system_output`, `reference_text`, `severity`

## Classification and integrity rules

1. Use explicit annotations or supporting evidence.
2. A difference between system output and reference does not establish the cause by itself.
3. Flag ambiguous classifications for human review.
4. Keep stage and category separate.
5. Avoid duplicate records for the same underlying defect.
6. Preserve source segment identifiers and original text.
7. Keep classifications deterministic for unchanged inputs and evidence.
8. Do not modify immutable raw scans or overwrite raw pipeline artifacts.
9. Use UTF-8 encoding without a BOM for text and CSV outputs.
10. Empty error datasets are valid when exported with the required headers.