"""Export the original Devanagari OCR text of every scanned page to a Word document.

The script reads the per-page raw OCR JSON produced by the project's OCR
pipeline and writes a single ``.docx`` file in which every source page has its
own heading with the page details (page number, page ID, source image, page
type) followed by that page's original OCR text.

The OCR language configuration for the full-volume run is ``san+hin``, so the
exported text covers both the Hindi and the Sanskrit (Devanagari) portions of
the manuscript exactly as the OCR engine produced them. The text is the raw
engine output: it is not normalized, corrected, translated, or edited.

The Word file is written with the Python standard library only (no
``python-docx`` or other third-party dependency). The document is a minimal,
valid Office Open XML package with the font set to a Devanagari-capable face
(``Nirmala UI`` by default, which ships with Windows).

Examples
--------
Export the full volume (all pages) from the default Tesseract ``otsu`` run::

    python scripts/17_export_ocr_docx.py

Smoke test on a few pages::

    python scripts/17_export_ocr_docx.py --start-page 1 --end-page 5 \
        --output evaluation/reports/ocr_preview.docx

Export the PaddleOCR subset instead (only the pages that were processed)::

    python scripts/17_export_ocr_docx.py --engine paddleocr \
        --ocr-root data/ocr_paddle_full \
        --output evaluation/reports/ocr_paddleocr.docx

Export the machine transcription produced by the vision model, which is far
more accurate than the classical engines on this manuscript::

    python scripts/17_export_ocr_docx.py --engine gemini --variant vision \
        --ocr-root data/ocr_gemini_full
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OCR_ROOT = ROOT / "data" / "ocr_full"
DEFAULT_MANIFEST = ROOT / "data" / "raw_scans" / "page_manifest.csv"
DEFAULT_OUTPUT = ROOT / "evaluation" / "reports" / "ocr_full_devanagari.docx"

WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

ENGINE_DISPLAY = {
    "tesseract": "Tesseract",
    "paddleocr": "PaddleOCR",
    "gemini": "Gemini (multimodal vision model)",
}

RAW_OCR_NOTE = (
    "The text below is the raw OCR engine output for each source page. "
    "It has not been normalized, corrected, translated, or edited. "
    "Recognition errors, garbled characters, and empty pages reflect the "
    "original OCR result and are reproduced here unchanged."
)

VISION_NOTE = (
    "The text below is a machine transcription produced by a multimodal "
    "vision model reading the original page scans. It is UNVERIFIED: it has "
    "not been proofread against the source, and it has not been normalized, "
    "corrected, translated, or edited. On the ten pages with a human-verified "
    "reference the measured character accuracy is 91% (range 82–96%). "
    "Recognition errors and unreadable passages marked [अपठनीय] reflect the "
    "model output and are reproduced here unchanged."
)

NOTE_BY_ENGINE = {"gemini": VISION_NOTE}


# --------------------------------------------------------------------------
# Page / OCR loading
# --------------------------------------------------------------------------
def page_id_for(page_number: int) -> str:
    """Return the canonical page ID for a page number (1-based)."""
    return f"BRIGU_V5_P{page_number:04d}"


def paginate_id(page_id: str) -> int | None:
    """Return the numeric page number encoded in a page ID, if present."""
    digits = page_id.rsplit("P", 1)[-1]
    return int(digits) if digits.isdigit() else None


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Load the page manifest into a mapping keyed by page ID.

    A missing manifest is not fatal: page-type metadata simply becomes
    unavailable and the export continues.
    """
    if not path.is_file():
        logger.warning("Page manifest not found at {}; continuing without it.", path)
        return {}

    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            page_id = (row.get("page_id") or "").strip()
            if page_id:
                records[page_id] = row
    return records


def load_ocr_record(ocr_root: Path, engine: str, variant: str, page_id: str) -> dict[str, Any] | None:
    """Load one page's raw OCR JSON, or return None when it does not exist."""
    path = ocr_root / engine / f"{page_id}__{variant}.json"
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


# --------------------------------------------------------------------------
# WordprocessingML helpers (standard library only)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------
def _page_heading(page_number: int, page_id: str, manifest_row: dict[str, str] | None) -> str:
    details = [f"PDF page {page_number}", page_id]
    if manifest_row:
        page_type = (manifest_row.get("page_type") or "").strip()
        status = (manifest_row.get("content_status") or "").strip()
        printed = (manifest_row.get("printed_page_number") or "").strip()
        if printed:
            details.append(f"printed page {printed}")
        if page_type:
            details.append(f"type: {page_type}")
        if status:
            details.append(f"status: {status}")
    return "  |  ".join(details)


def build_body(
    pages: list[dict[str, Any]],
    *,
    engine: str,
    variant: str,
    source_image_of: dict[int, str],
    page_break: bool,
    note: str | None = None,
) -> str:
    engine_label = ENGINE_DISPLAY.get(engine, engine)
    engines_seen = sorted({p["engine_version"] for p in pages if p["engine_version"]})
    params_seen = sorted({json.dumps(p["engine_params"], sort_keys=True) for p in pages if p["engine_params"]})
    params_display = params_seen[0] if len(params_seen) == 1 else "varies per page"

    blocks: list[str] = []
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
            "Devanagari Text (Hindi / Sanskrit) — Full Volume",
            size=26,
            align="center",
            space_after=200,
        )
    )
    blocks.append(_para("Document details", size=26, bold=True, keep_with_next=True, space_after=80))
    meta_lines = [
        f"OCR engine: {engine_label} {', '.join(engines_seen) if engines_seen else ''}".rstrip(),
        f"Preprocessing variant: {variant}",
        f"Engine parameters: {params_display}",
    ]
    if any(
        isinstance(p["engine_params"], dict) and "language" in p["engine_params"]
        for p in pages
    ):
        meta_lines.append("OCR language packs: san+hin (Devanagari — Hindi and Sanskrit)")
    meta_lines.extend(
        [
            f"Pages in this document: {len(pages)}",
            f"Generated (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
        ]
    )
    for line in meta_lines:
        blocks.append(_para(line, space_after=40))
    blocks.append(
        _para(
            note if note is not None else NOTE_BY_ENGINE.get(engine, RAW_OCR_NOTE),
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
                _page_heading(page_number, page["page_id"], page["manifest_row"]),
                size=26,
                bold=True,
                keep_with_next=True,
                space_after=60,
            )
        )
        source_image = source_image_of.get(page_number, page["source_image"])
        blocks.append(
            _para(
                f"Source image: {source_image}",
                italic=True,
                space_after=120,
                keep_with_next=True,
            )
        )
        text = page["text"]
        if text.strip():
            blocks.append(_para(text))
        elif page["found"]:
            blocks.append(_para("(No text was recognized for this page.)", italic=True))
        else:
            blocks.append(_para("(OCR output for this page was not found.)", italic=True))

    return "".join(blocks)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the original Devanagari OCR text of every page to a Word document."
        )
    )
    parser.add_argument("--engine", default="tesseract", help="OCR engine directory (default: tesseract).")
    parser.add_argument("--variant", default="otsu", help="Preprocessing variant (default: otsu).")
    parser.add_argument(
        "--ocr-root",
        type=Path,
        default=DEFAULT_OCR_ROOT,
        help=f"Root of the OCR output tree (default: {DEFAULT_OCR_ROOT}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Page manifest CSV used for page-type details.",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First page number (default: 1).")
    parser.add_argument("--end-page", type=int, default=888, help="Last page number (default: 888).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .docx path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--font",
        default="Nirmala UI",
        help="Document font with Devanagari coverage (default: Nirmala UI).",
    )
    parser.add_argument(
        "--no-page-break",
        action="store_true",
        help="Do not start each source page on a new Word page.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Override the provenance note printed in the document front matter.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.start_page < 1 or args.end_page < args.start_page:
        logger.error("Invalid page range {}–{}.", args.start_page, args.end_page)
        return 2

    numbers = list(range(args.start_page, args.end_page + 1))
    logger.info(
        "Exporting {} pages {}–{} from engine '{}', variant '{}'.",
        len(numbers),
        args.start_page,
        args.end_page,
        args.engine,
        args.variant,
    )

    try:
        manifest = load_manifest(args.manifest)
    except OSError as exc:
        logger.error("Could not read the page manifest ({}).", type(exc).__name__)
        return 1

    pages: list[dict[str, Any]] = []
    source_image_of: dict[int, str] = {}
    missing: list[int] = []
    empty: list[int] = []

    for number in numbers:
        page_id = page_id_for(number)
        record = load_ocr_record(args.ocr_root, args.engine, args.variant, page_id)
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
        source_image_of[number] = source_image

        pages.append(
            {
                "page_number": int(record.get("page_number", number)) if found else number,
                "page_id": page_id,
                "source_image": source_image,
                "text": text,
                "found": found,
                "engine_version": str(record.get("engine_version", "")) if found else "",
                "engine_params": record.get("engine_params") if found else None,
                "manifest_row": manifest.get(page_id),
            }
        )

    body = build_body(
        pages,
        engine=args.engine,
        variant=args.variant,
        source_image_of=source_image_of,
        page_break=not args.no_page_break,
        note=args.note,
    )

    output = args.output if args.output.is_absolute() else (ROOT / args.output)
    try:
        write_docx(output, build_document_xml(body), args.font)
    except OSError as exc:
        logger.error("Could not write the Word document ({}).", type(exc).__name__)
        return 1

    logger.info("Wrote {} ({} pages).", output, len(pages))
    logger.info(
        "Summary: {} pages written, {} with OCR output, {} empty, {} missing.",
        len(pages),
        len(pages) - len(missing),
        len(empty),
        len(missing),
    )
    if missing:
        logger.warning("Missing OCR output for pages: {}", _shorten(missing))
    if empty:
        logger.warning("Empty OCR output for pages: {}", _shorten(empty))
    return 0


def _shorten(numbers: list[int], limit: int = 25) -> str:
    shown = ", ".join(str(n) for n in numbers[:limit])
    return shown if len(numbers) <= limit else f"{shown}, … (+{len(numbers) - limit} more)"


if __name__ == "__main__":
    raise SystemExit(main())
