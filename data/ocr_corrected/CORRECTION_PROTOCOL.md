# Human OCR Correction Protocol

## 1. Purpose

This protocol defines how OCR text from Shri Bhrigu Samhita Kundali Rahasyam - Volume 5 is reviewed and corrected. It applies to the experimental subset processed during Phase 3 of the Indic NLP & Multilingual AI Challenge.

The objectives are to improve transcription accuracy, preserve source fidelity, maintain reproducible corrections, and retain complete data provenance.

## 2. Source of Evidence

The original scanned page is the primary evidence for every correction.

The following artifacts may support review:

- Original page image in `data/raw_scans/`.
- Preprocessed image used for OCR.
- Tesseract OCR output.
- PaddleOCR output, where available.
- OCR comparison JSON artifacts.

OCR agreement does not prove that a reading is correct. When OCR engines disagree, consult the source image. If the image does not resolve the disagreement, mark the reading as uncertain rather than guessing.

## 3. Immutable Source Artifacts

The following locations must not be modified by the correction process:

- `data/raw_scans/`
- `data/ocr_raw/tesseract/`
- `data/ocr_raw/paddleocr/`

Corrections must be written separately under `data/ocr_corrected/`.

Original OCR text must remain recoverable. A correction must never silently replace or erase its source OCR text.

## 4. General Correction Principles

1. Transcribe what is visible in the source, not what is expected to appear.
2. Preserve the original language, wording, spelling, grammar, and terminology.
3. Do not translate, paraphrase, modernize, or editorially rewrite the text.
4. Use linguistic context to identify passages needing review, not as a substitute for visual evidence.
5. Make only corrections supported by the source image or clearly documented evidence.
6. Preserve meaningful punctuation, numerals, word boundaries, and verse boundaries.
7. Record unresolved ambiguities explicitly.
8. Distinguish corrected OCR from independently verified ground truth.
9. Retain source page IDs and OCR provenance for every correction.
10. Apply correction rules consistently across the experimental subset.

## 5. Character-Level Correction Rules

### 5.1 Devanagari conjuncts

Correct conjuncts that OCR has split, omitted, or misrecognized only when the source image supports the corrected reading. Do not insert a conjunct merely because it would make a word linguistically plausible.

### 5.2 Matras

Restore missing, duplicated, or incorrect matras when the source image supports the correction. If the relevant character is obscured or ambiguous, record the uncertainty.

### 5.3 Halant and virama

Preserve legitimate halant/virama usage. Correct missing or spurious marks only when justified by the scan. Do not normalize away distinctions that affect the written text.

### 5.4 Character substitutions

Review visually confusable characters individually. Accept a substitution only when the source image supports it. Do not apply global replacements that may corrupt legitimate occurrences elsewhere.

### 5.5 Danda punctuation

Preserve the distinction between single danda `।` and double danda `॥`. Correct damaged, missing, or duplicated punctuation only when the source supports the change. Do not automatically replace one form with the other.

## 6. Word Boundaries and Whitespace

### 6.1 Broken words

Rejoin words split by OCR when the scan confirms that the split is artificial. Preserve genuine word boundaries and meaningful manuscript line breaks when required by the annotation policy.

### 6.2 Whitespace

Remove accidental repeated spaces and correct OCR-generated spacing errors without changing the text's meaning or segmentation. Do not merge distinct words based solely on linguistic expectation.

### 6.3 Line breaks

A line break in the scan does not necessarily represent a sentence or verse boundary. Preserve sufficient source-layout information to support later manuscript-aware segmentation.

## 7. OCR Noise and Page Furniture

### 7.1 OCR-inserted noise

Remove characters or strings that are demonstrably OCR artifacts and are not present in the source. If their origin is uncertain, flag them for review.

### 7.2 Headers, footers, and page numbers

Determine whether each element belongs to the transcription scope established by the annotation guidelines. Exclude out-of-scope material from the transcription, but retain its source-page relationship and document the exclusion when relevant.

Do not remove text simply because it appears near a page margin.

## 8. Hindi, Sanskrit, and Numerals

### 8.1 Mixed Hindi and Sanskrit

Preserve language mixing as printed. Do not translate Sanskrit passages into Hindi or rewrite Hindi passages into Sanskrit. Language labels are metadata and must not alter the transcription.

### 8.2 Numerals and entry markers

Verify Devanagari numerals, other printed numeral forms, and numbered entry markers against the source. Do not automatically convert numerals between scripts or alter entry numbering.

A marker such as `(१४३७)` must be preserved when supported by the scan. The canonical representation of ambiguous or alternative numeral forms must follow the ground-truth annotation guidelines.

### 8.3 Classical terminology

Preserve names, astrological terminology, philosophical terms, and other domain-specific expressions as printed. Do not substitute a more familiar term based on subject-matter expectations.

## 9. Unreadable and Uncertain Text

Do not invent characters or words to fill gaps.

Use the following correction-status vocabulary:

- `pending_review`: the passage has not yet been fully reviewed.
- `corrected`: an OCR error was corrected using source evidence.
- `verified`: the transcription was checked against the scan and accepted.
- `uncertain`: the source permits multiple plausible readings or does not support a definitive reading.
- `unreadable`: the relevant text cannot be reliably recovered.

These statuses describe the review state; they are not interchangeable.

When only part of a passage is unreadable, retain the readable portions. Record the affected location and explanation in review notes. Use a documented uncertainty notation in the transcription only after its representation has been established in the ground-truth annotation guidelines. Never allow a review placeholder to be mistaken for literal manuscript text.

## 10. Review and Correction Procedure

For each candidate text:

1. Identify the source page using its stable page ID.
2. Locate the corresponding OCR output and preprocessing variant.
3. Inspect the OCR comparison artifact, when available.
4. Compare the candidate reading with the source image.
5. Correct only discrepancies supported by the evidence.
6. Record the corrected text and review status.
7. Record uncertainty, exclusions, and substantive corrections in review notes.
8. Preserve links to the original OCR artifacts.
9. Verify that the record can be traced back to its source page.

When multiple OCR engines or preprocessing variants contribute to a decision, record all relevant source artifacts or explicitly identify the comparison-based review. Do not falsely attribute a combined review to one engine.

## 11. Provenance Requirements

Each corrected record must retain, directly or through a documented reference:

- A unique correction-record identifier.
- The stable source `page_id`.
- The source-image path.
- The OCR engine or engines used.
- The original OCR artifact path or paths.
- The preprocessing variant, where applicable.
- The comparison artifact path, where applicable.
- The original OCR text.
- The corrected text.
- The correction status.
- Review notes for corrections, exclusions, or uncertainty.

The exact serialization schema will be established before corrected records are produced. These requirements do not prescribe an unverified schema for existing Phase 3 artifacts.

Use repository-relative paths. All text artifacts must use UTF-8 without a byte-order mark.

## 12. Corrected OCR Versus Ground Truth

Corrected OCR is the reviewed output of an OCR-based transcription workflow. Ground truth is the verified reference transcription created according to `data/ground_truth/ANNOTATION_GUIDELINES.md`.

The two artifacts must remain distinct. A corrected OCR record must not automatically be considered ground truth merely because it has been reviewed.

## 13. Validation Requirements

Before accepting corrected records, verify that:

- Source page IDs are valid and recoverable.
- Original OCR artifacts remain unchanged.
- Source-image and OCR provenance are retained.
- Corrected text is stored separately.
- Correction statuses use the defined vocabulary.
- Uncertain readings are explicitly documented.
- Text serialization is valid UTF-8 without a BOM.
- No unsupported linguistic guesses have been introduced.
- Records can be traced to their original source artifacts.

## 14. Scope and Change Control

This protocol governs human OCR correction during Phase 4. Ground-truth annotation, Unicode normalization, language identification, segmentation, tokenization, transliteration, and translation are separate pipeline stages.

Changes to this protocol must be documented and version-controlled. Existing raw OCR artifacts must never be altered to conform to a revised policy.
'@
