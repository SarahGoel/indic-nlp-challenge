"""Translate OCR text to target languages and export to Word documents.

This script reads OCR text from the full volume, translates it to a specified
target language using Google Gemini API, and exports the translated text to a DOCX
file with proper formatting.

Supported target languages:
- ben (Bengali)
- guj (Gujarati)
- mar (Marathi)
- tam (Tamil)
- tel (Telugu)
- kan (Kannada)
- mal (Malayalam)
- pan (Punjabi)
- asm (Assamese)
- ory (Odia)
- urd (Urdu)

The source language is Hindi (hin) with Sanskrit (san) elements.

Dependencies:
- Google Generative AI library: pip install google-generativeai
- Gemini API key: Set GEMINI_API_KEY environment variable

Example usage:
    python scripts/18_translate_and_export.py --target-language ben
    python scripts/18_translate_and_export.py --target-language tam --start-page 1 --end-page 10
"""

from __future__ import annotations

import argparse
import html
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OCR_ROOT = ROOT / "data" / "ocr_full"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "translations"
DEFAULT_MANIFEST = ROOT / "data" / "raw_scans" / "page_manifest.csv"

WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

LANGUAGE_NAMES = {
    "hin": "Hindi",
    "san": "Sanskrit",
    "ben": "Bengali",
    "guj": "Gujarati",
    "mar": "Marathi",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "pan": "Punjabi",
    "asm": "Assamese",
    "ory": "Odia",
    "urd": "Urdu",
}

TRANSLATION_NOTE = (
    "The text below is machine-translated from the original Devanagari OCR. "
    "Translation quality varies; errors may occur. The translation has not "
    "been proofread or verified. For critical applications, consult the "
    "original source or a qualified translator."
)


def page_id_for(page_number: int) -> str:
    """Return the canonical page ID for a page number (1-based)."""
    return f"BRIGU_V5_P{page_number:04d}"


def load_ocr_record(ocr_root: Path, page_id: str) -> dict[str, Any] | None:
    """Load one page's OCR JSON, or return None when it does not exist."""
    path = ocr_root / "tesseract" / f"{page_id}__otsu.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error(
            "Could not read OCR file for {} ({}): {}",
            page_id,
            path.name,
            type(exc).__name__,
        )
        return None


def build_engine(model_names: str | list[str] | None) -> Any:
    """Build the Gemini engine once per run, or return None if unavailable."""
    try:
        from indicnlp_pipeline.translation.gemini_engine import GeminiTranslationEngine

        return GeminiTranslationEngine(model_name=model_names)
    except Exception as exc:
        logger.error("Could not initialise the Gemini engine: {} - {}", type(exc).__name__, exc)
        logger.error("Gemini API key may be missing or invalid; set GEMINI_API_KEY.")
        return None


def checkpoint_path(output_root: Path, target_language: str) -> Path:
    """Return the JSONL checkpoint path for a target language."""
    return output_root / f"brigu_v5_{target_language}.pages.jsonl"


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load previously translated pages from the checkpoint, keyed by page ID.

    Only records flagged ``ok`` are returned, so pages that were previously
    exported untranslated (for example during a network outage) are retried.
    """
    done: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return done
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.error("Could not read checkpoint {} ({}).", path.name, type(exc).__name__)
        return done
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("ok") and str(record.get("translated_text", "")).strip():
            done[str(record["page_id"])] = record
    return done


def translate_text(text: str, source_lang: str, target_lang: str, engine: Any) -> tuple[str, bool]:
    """Translate text using the shared Gemini engine (model pool).

    The engine is built once per run so its model blacklist and last-known-good
    model persist across pages instead of being reset on every call.

    Returns:
        A ``(translated_text, ok)`` pair. ``ok`` is False when the text was
        empty or translation failed (the original text is returned unchanged),
        so callers never checkpoint untranslated output.
    """
    # Handle empty text
    if not text or not text.strip():
        return text, False

    if engine is None:
        return text, False

    try:
        return engine.translate(text, source_lang, target_lang), True
    except Exception as exc:
        logger.error("Translation failed: {} - {}", type(exc).__name__, str(exc))
        return text, False  # Return original if translation fails


def _valid_xml_chars(text: str) -> str:
    """Drop characters that are not legal in an XML 1.0 document."""
    out = []
    for ch in text:
        code = ord(ch)
        if (
            code in (0x09, 0x0A, 0x0D)
            or 0x20 <= code <= 0xD7FF
            or 0xE000 <= code <= 0xFFFD
            or 0x10000 <= code <= 0x10FFFF
        ):
            out.append(ch)
    return "".join(out)


def _esc(text: str) -> str:
    return html.escape(_valid_xml_chars(text), quote=False)


def _runs(text: str) -> str:
    """Return run XML for text, preserving line breaks as <w:br/>."""
    parts: list[str] = []
    for index, line in enumerate(text.split("\n")):
        if index:
            parts.append("<w:br/>")
        parts.append(f'<w:t xml:space="preserve">{_esc(line)}</w:t>')
    return "".join(parts)


def _para(
    text: str = "",
    *,
    size: int | None = None,
    bold: bool = False,
    italic: bool = False,
    align: str | None = None,
    space_after: int = 120,
    keep_with_next: bool = False,
) -> str:
    """Build one paragraph. An empty ``text`` still yields a valid paragraph."""
    props = f'<w:spacing w:after="{space_after}"/>'
    if align:
        props += f'<w:jc w:val="{align}"/>'
    if keep_with_next:
        props += "<w:keepNext/>"
    ppr = f"<w:pPr>{props}</w:pPr>"

    run_props = ""
    if bold or italic or size:
        run_props = "<w:rPr>"
        if bold:
            run_props += "<w:b/>"
        if italic:
            run_props += "<w:i/>"
        if size:
            run_props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        run_props += "</w:rPr>"

    return f"<w:p>{ppr}<w:r>{run_props}{_runs(text)}</w:r></w:p>"


def _page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def _styles(font: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:styles xmlns:w="{WNS}"><w:docDefaults><w:rPrDefault><w:rPr>'
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
        '<w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault>'
        "</w:docDefaults></w:styles>"
    )


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    "</Types>"
)

_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)

_DOC_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)


def build_document_xml(body: str) -> str:
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document xmlns:w="{WNS}"><w:body>{body}{sect}</w:body></w:document>'
    )


def write_docx(path: Path, document_xml: str, font: str) -> None:
    """Write a minimal DOCX package atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _ROOT_RELS)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", _styles(font))
        archive.writestr("word/_rels/document.xml.rels", _DOC_RELS)
    tmp.replace(path)


def build_body(
    pages: list[dict[str, Any]],
    *,
    source_lang: str,
    target_lang: str,
    page_break: bool,
) -> str:
    blocks: list[str] = []

    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    blocks.append(
        _para(
            "Shri Bhrigu Samhita Kundali Rahasyam — Volume 5",
            size=36,
            bold=True,
            align="center",
            space_after=80,
        )
    )
    blocks.append(
        _para(
            f"{target_name} Translation — Full Volume",
            size=26,
            align="center",
            space_after=200,
        )
    )
    blocks.append(_para("Document details", size=26, bold=True, keep_with_next=True, space_after=80))
    meta_lines = [
        f"Source language: {source_name}",
        f"Target language: {target_name}",
        f"Translation engine: Gemini API",
        f"Pages in this document: {len(pages)}",
        f"Generated (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
    ]
    for line in meta_lines:
        blocks.append(_para(line, space_after=40))
    blocks.append(
        _para(
            TRANSLATION_NOTE,
            italic=True,
            space_after=200,
        )
    )

    for index, page in enumerate(pages):
        page_number = page["page_number"]
        if page_break and index > 0:
            blocks.append(_page_break())
        blocks.append(
            _para(
                f"Page {page_number} — {page['page_id']}",
                size=26,
                bold=True,
                keep_with_next=True,
                space_after=60,
            )
        )
        blocks.append(
            _para(
                f"Source image: {page['source_image']}",
                italic=True,
                space_after=120,
                keep_with_next=True,
            )
        )
        text = page["translated_text"]
        if text.strip():
            blocks.append(_para(text))
        elif page["found"]:
            blocks.append(_para("(No text was recognized for this page.)", italic=True))
        else:
            blocks.append(_para("(OCR output for this page was not found.)", italic=True))

    return "".join(blocks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate OCR text to target language and export to Word document."
    )
    parser.add_argument(
        "--target-language",
        required=True,
        choices=["ben", "guj", "mar", "tam", "tel", "kan", "mal", "pan", "asm", "ory", "urd"],
        help="Target language code.",
    )
    parser.add_argument(
        "--source-language",
        default="hin",
        choices=["hin", "san"],
        help="Source language code (default: hin).",
    )
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help=f"Root of the OCR output tree (default: {DEFAULT_OCR_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for translation outputs (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First page number (default: 1).")
    parser.add_argument("--end-page", type=int, default=888, help="Last page number (default: 888).")
    parser.add_argument(
        "--font",
        default="Nirmala UI",
        help="Document font (default: Nirmala UI).",
    )
    parser.add_argument(
        "--no-page-break",
        action="store_true",
        help="Do not start each source page on a new Word page.",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation and export original OCR text (for testing).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the checkpoint and re-translate every page from scratch.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Gemini model, or comma-separated model pool, tried in order "
            "(default: GEMINI_MODEL_POOL env var, else built-in pool)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.start_page < 1 or args.end_page < args.start_page:
        logger.error("Invalid page range {}–{}.", args.start_page, args.end_page)
        return 2

    target_name = LANGUAGE_NAMES.get(args.target_language, args.target_language)
    output_path = args.output_root / f"brigu_v5_{args.target_language}.docx"

    numbers = list(range(args.start_page, args.end_page + 1))
    logger.info(
        "Translating {} pages {}–{} from {} to {}.",
        len(numbers),
        args.start_page,
        args.end_page,
        args.source_language,
        args.target_language,
    )

    pages: list[dict[str, Any]] = []
    missing: list[int] = []
    empty: list[int] = []

    ckpt_path = checkpoint_path(args.output_root, args.target_language)
    cached = {} if args.no_resume else load_checkpoint(ckpt_path)
    if cached:
        logger.info("Resuming: found {} already-translated pages in {}.", len(cached), ckpt_path.name)

    engine = None if args.skip_translation else build_engine(args.model_name)

    args.output_root.mkdir(parents=True, exist_ok=True)
    resumed = 0

    with ckpt_path.open("a", encoding="utf-8") as ckpt:
        for number in numbers:
            page_id = page_id_for(number)
            record = load_ocr_record(args.ocr_root, page_id)
            found = record is not None
            text = str(record.get("text", "")) if found else ""

            if not found:
                missing.append(number)
            elif not text.strip():
                empty.append(number)

            source_image = (
                str(record.get("source_image"))
                if found and record.get("source_image")
                else f"{page_id}.png"
            )

            previous = cached.get(page_id)
            if previous is not None:
                translated_text = str(previous["translated_text"])
                resumed += 1
            elif args.skip_translation:
                translated_text, ok = text, True
            else:
                translated_text, ok = translate_text(
                    text, args.source_language, args.target_language, engine
                )

            # Persist the translation as soon as it succeeds, so a stop, crash, or
            # network outage never discards pages that are already translated.
            if previous is None and ok:
                ckpt.write(
                    json.dumps(
                        {
                            "page_id": page_id,
                            "page_number": number,
                            "ok": True,
                            "translated_text": translated_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ckpt.flush()

            pages.append(
                {
                    "page_number": int(record.get("page_number", number)) if found else number,
                    "page_id": page_id,
                    "source_image": source_image,
                    "translated_text": translated_text,
                    "found": found,
                }
            )

            if (number % 10) == 0:
                logger.info("Processed {} pages...", number)

    body = build_body(
        pages,
        source_lang=args.source_language,
        target_lang=args.target_language,
        page_break=not args.no_page_break,
    )

    try:
        write_docx(output_path, build_document_xml(body), args.font)
    except OSError as exc:
        logger.error("Could not write the Word document ({}).", type(exc).__name__)
        return 1

    logger.info("Wrote {} ({} pages, {} resumed from checkpoint).", output_path, len(pages), resumed)
    logger.info(
        "Summary: {} pages written, {} with OCR output, {} empty, {} missing.",
        len(pages),
        len(pages) - len(missing),
        len(empty),
        len(missing),
    )
    if missing:
        logger.warning("Missing OCR output for pages: {}", ", ".join(map(str, missing[:20])))
    if empty:
        logger.warning("Empty OCR output for pages: {}", ", ".join(map(str, empty[:20])))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
