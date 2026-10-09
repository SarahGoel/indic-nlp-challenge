# Ground Truth Annotation Guidelines

## 1. Purpose

These guidelines define a consistent, auditable human annotation process for building a high-quality Hindi-first, Sanskrit-aware Devanagari corpus from Shri Bhrigu Samhita Kundali Rahasyam – Volume 5.

Ground truth is the human-verified transcription of the source image. It is used to evaluate OCR, support text processing, and provide a reliable source for downstream translation and multilingual NLP.

## 2. Scope

These guidelines cover:
- Human transcription and verification of printed source text.
- Review of Tesseract and PaddleOCR output against the original page image.
- Devanagari character, word-boundary, punctuation, numeral, and line-break handling.
- Explicit recording of uncertainty and unreadable text.
- Annotation provenance and quality control.

The source language configuration is Hindi (`hin`) as primary and Sanskrit (`san`) as secondary. Do not assume every page or passage contains only one of these languages.

## 3. Source of Evidence

Use this evidence hierarchy:

1. The corresponding original page image under `data/raw_scans/` is the authoritative evidence.
2. Preprocessed page images may help reveal characters but must not override the original scan.
3. Tesseract and PaddleOCR output are hypotheses to review, not ground truth.
4. OCR comparison files may help identify disagreements but cannot establish the correct reading by themselves.
5. Context may help identify a likely reading, but must not justify inventing characters that the image does not support.

Never modify files in `data/raw_scans/` during annotation.

## 4. Transcription Principles

- Transcribe the printed source faithfully.
- Preserve the source's wording, spelling, grammatical forms, and historical or traditional terminology.
- Do not translate, paraphrase, summarize, modernize, or silently correct the source text.
- Do not replace an unfamiliar word with a more familiar word merely because it seems more plausible.
- Preserve meaningful verse structure and source punctuation.
- Do not infer missing text from context or another edition without recording the issue and its evidence.
- Keep transcription separate from interpretation, language classification, and translation.

## 5. Devanagari Character Handling

Inspect character-level details carefully, especially:

- Conjunct consonants and their components.
- Independent vowels and dependent vowel signs (matras).
- Halant/virama and conjunct formation.
- Anusvara, chandrabindu, visarga, and nukta where present.
- Similar-looking characters that OCR engines may confuse.
- Character sequences affected by blur, ink, page curvature, or scan defects.

Do not infer a character solely from the OCR engine's confidence or output agreement. Verify it against the scan.

## 6. Word Boundaries, Lines, and Verses

- Preserve meaningful word boundaries supported by the source.
- Do not join separate words or split a word merely to improve readability.
- Treat line wrapping as layout unless it changes the textual structure.
- Preserve verse boundaries and meaningful line order.
- Retain danda (`।`) and double danda (`॥`) when present.
- Do not insert punctuation solely to make the text read more naturally.
- If a boundary cannot be determined reliably, record the uncertainty in the annotation notes.

## 7. Numerals, Headings, and Page Furniture

- Transcribe source numerals and verse or entry markers faithfully.
- Do not convert numeral systems unless a separate, documented transformation requires it.
- Distinguish textual headings from running headers, footers, page numbers, stamps, and scanning artifacts.
- Do not silently discard potentially meaningful content.
- Record a consistent decision about non-textual page furniture in the annotation notes or applicable metadata.

## 8. Uncertain and Unreadable Text

Use explicit uncertainty handling rather than guessing.

- If a character or passage is ambiguous, preserve the best-supported reading only when the evidence justifies it, and record the uncertainty.
- If text is genuinely unreadable, mark it using the documented project representation for unreadable content.
- Do not use an empty string to imply unreadability.
- Do not confuse missing source content, cropped content, illegible printing, and OCR failure.
- Record the affected location and explain what makes the reading uncertain.
- If review cannot resolve the issue, leave it pending or uncertain rather than falsely marking it verified.

The exact machine-readable representation of uncertainty must be defined consistently in the ground-truth schema before annotations are serialized.

## 9. Hindi and Sanskrit

- Treat Hindi (`hin`) as the primary configured language and Sanskrit (`san`) as secondary.
- A passage may contain either language, both languages, names, quotations, or traditional terms.
- Do not force Sanskrit forms into Hindi spelling or Hindi forms into Sanskrit spelling.
- Do not infer a language label from script alone because both languages commonly use Devanagari.
- Record language uncertainty for later review rather than silently choosing a label.
- Language labels describe the text; they do not authorize changing its transcription.

## 10. Unicode and Text Normalization

- Preserve the observed text during transcription.
- Use valid Unicode text and UTF-8 for stored annotation data.
- Do not manually replace visually similar characters with unrelated Unicode characters.
- Keep source-faithful transcription distinct from any later programmatic normalization.
- Unicode normalization, whitespace normalization, tokenization, transliteration, and translation must be separate, documented pipeline operations.
- Any normalization must preserve traceability to the original transcription.

## 11. Annotation Statuses

Use the following conceptual statuses consistently:

- `pending_review`: not yet reviewed by a human.
- `corrected`: transcription has been edited but has not completed verification.
- `verified`: a reviewer has checked the transcription against the source image.
- `uncertain`: some reading remains ambiguous after review.
- `unreadable`: the relevant source text cannot be read reliably.

Do not mark a record `verified` merely because two OCR engines agree. A verified status requires human inspection of the source image.

The final serialization schema must define which statuses are valid for each record type and how unresolved text is represented.

## 12. Provenance and Traceability

Each annotation must be traceable to its source. Record or link the following information through the project schema:

- Stable page identifier and PDF page number.
- Repository-relative original image path.
- Relevant preprocessing variant, if used during review.
- OCR engine and source output path, where applicable.
- Original OCR transcription and corrected transcription, where applicable.
- Annotation or correction status.
- Reviewer identifier or approved reviewer label, if used.
- Review timestamp in a documented format.
- Notes describing uncertain readings and consequential decisions.

Use repository-relative paths. Do not embed machine-specific absolute Windows paths in dataset records.

Keep raw OCR, corrected OCR, and verified ground truth as distinct artifacts. Never overwrite OCR output to make it appear human-verified.

## 13. Review and Quality Control

For every annotation:

1. Open the correct original page image.
2. Confirm the page identifier and source location.
3. Review the OCR text against the image, character by character where necessary.
4. Check word boundaries, matras, conjuncts, punctuation, numerals, and verse structure.
5. Mark uncertain or unreadable regions explicitly.
6. Record consequential decisions and provenance.
7. Perform an independent second review for a defined quality-control subset or for high-uncertainty records.
8. Mark the record verified only when the applicable review requirements are satisfied.

If a later review changes a verified transcription, preserve an auditable record of the change rather than concealing the earlier decision.

## 14. Separation from Downstream Processing

Ground truth represents the source transcription, not a cleaned translation-ready rewrite.

Keep these operations separate:
- OCR correction.
- Ground-truth verification.
- Unicode normalization.
- Language identification.
- Verse and sentence segmentation.
- Tokenization.
- Transliteration.
- Translation.
- Evaluation and error analysis.

Derived text must remain traceable to its source annotation. Do not replace the source transcription with translated, transliterated, or normalized output.

## 15. Acceptance Criteria

An annotation is ready for verified ground truth only when:

- Its source page is identifiable.
- The transcription has been checked against the original scan.
- No unsupported text has been invented.
- Uncertainty and unreadable regions are explicitly handled.
- Source-faithful wording and meaningful textual structure are preserved.
- Provenance is sufficient to reproduce the review.
- The record conforms to the project's finalized serialization schema.
- The reviewer has assigned an appropriate status.

## 16. Change Control

Apply these guidelines consistently across the dataset. Document any policy change, explain its impact on existing annotations, and review affected records when necessary.

These guidelines define annotation policy. The machine-readable field names, required fields, uncertainty encoding, and validation rules are finalized separately in the ground-truth schema and serialization steps.