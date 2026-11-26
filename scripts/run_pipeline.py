"""Complete pipeline runner for PDF to translation processing.

This script orchestrates the entire pipeline:
1. PDF to page images
2. OCR processing
3. Text normalization
4. Translation to target languages (using Gemini API)
5. DOCX export

Usage:
    python scripts/run_pipeline.py --input manuscript.pdf --target-language ben
    python scripts/run_pipeline.py --input manuscript.pdf --target-language tam --export-ocr
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# The numbered helper scripts cannot be imported with a normal ``from ... import``
# statement (module names may not start with a digit), so they are loaded by name.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def step_1_extract_pages(pdf_path: Path, output_dir: Path) -> None:
    """Extract pages from PDF to images."""
    logger.info("Step 1: Extracting pages from PDF...")
    
    try:
        from pypdfium2 import PdfDocument
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf = PdfDocument(str(pdf_path))
        page_count = len(pdf)
        
        for page_num in range(page_count):
            page = pdf[page_num]
            pil_image = page.render(scale=2).to_pil()
            output_path = output_dir / f"page_{page_num + 1:04d}.png"
            pil_image.save(output_path)
            
            if (page_num + 1) % 10 == 0:
                logger.info("Extracted {} / {} pages", page_num + 1, page_count)
        
        pdf.close()
        logger.info("Step 1 complete: {} pages extracted", page_count)
        
    except ImportError:
        logger.error("pypdfium2 not installed. Install with: pip install pypdfium2")
        raise
    except Exception as exc:
        logger.error("Failed to extract pages: {}", type(exc).__name__)
        raise


def step_2_run_ocr(images_dir: Path, output_dir: Path) -> None:
    """Run OCR on all page images."""
    logger.info("Step 2: Running OCR...")
    
    try:
        from indicnlp_pipeline.ocr.production_runner import ProductionRunner
        from indicnlp_pipeline.ocr.tesseract_engine import run_tesseract_ocr
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = sorted(images_dir.glob("*.png"))
        total = len(images)
        
        for i, image_path in enumerate(images):
            try:
                result = run_tesseract_ocr(image_path, psm=6, oem=3, language="hin+san")
                
                output_path = output_dir / f"{image_path.stem}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                if (i + 1) % 10 == 0:
                    logger.info("OCR processed {} / {} pages", i + 1, total)
                    
            except Exception as exc:
                logger.error("OCR failed for {}: {}", image_path.name, type(exc).__name__)
        
        logger.info("Step 2 complete: OCR processed {} pages", total)
        
    except ImportError:
        logger.error("Required modules not available")
        raise
    except Exception as exc:
        logger.error("OCR failed: {}", type(exc).__name__)
        raise


def step_3_normalize_text(ocr_dir: Path, output_dir: Path) -> None:
    """Normalize OCR text."""
    logger.info("Step 3: Normalizing text...")
    
    try:
        from indicnlp_pipeline.normalization.unicode_normalizer import normalize_text
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ocr_files = sorted(ocr_dir.glob("*.json"))
        total = len(ocr_files)
        
        for i, ocr_file in enumerate(ocr_files):
            try:
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                
                text = data.get('text', '')
                normalized = normalize_text(text)
                
                output_path = output_dir / ocr_file.name
                data['text'] = normalized
                data['normalized'] = True
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                if (i + 1) % 10 == 0:
                    logger.info("Normalized {} / {} pages", i + 1, total)
                    
            except Exception as exc:
                logger.error("Normalization failed for {}: {}", ocr_file.name, type(exc).__name__)
        
        logger.info("Step 3 complete: Normalized {} pages", total)
        
    except Exception as exc:
        logger.error("Normalization failed: {}", type(exc).__name__)
        raise


def step_4_translate(normalized_dir: Path, target_lang: str, output_dir: Path, model_name: str | None = None) -> None:
    """Translate text to target language using Gemini API."""
    logger.info("Step 4: Translating to {}...", target_lang)

    try:
        from indicnlp_pipeline.translation.gemini_engine import GeminiTranslationEngine

        output_dir.mkdir(parents=True, exist_ok=True)

        norm_files = sorted(normalized_dir.glob("*.json"))
        total = len(norm_files)

        engine = GeminiTranslationEngine(model_name=model_name)

        for i, norm_file in enumerate(norm_files):
            try:
                with open(norm_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)

                text = data.get('text', '')
                if not text.strip():
                    logger.warning("Skipping empty text for {}", norm_file.name)
                    continue

                translated_text = engine.translate(text, 'hin', target_lang)

                output_path = output_dir / norm_file.name
                data['text'] = translated_text
                data['target_language'] = target_lang
                data['translated'] = True

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                if (i + 1) % 10 == 0:
                    logger.info("Translated {} / {} pages", i + 1, total)

            except Exception as exc:
                logger.error("Translation failed for {}: {}", norm_file.name, type(exc).__name__)

        logger.info("Step 4 complete: Translated {} pages", total)

    except Exception as exc:
        logger.error("Translation failed: {}", type(exc).__name__)
        logger.error("Gemini API key may be missing or invalid")
        raise


def step_5_export_docx(translated_dir: Path, target_lang: str, output_path: Path) -> None:
    """Export translated text to DOCX."""
    logger.info("Step 5: Exporting to DOCX...")
    
    try:
        export_mod = importlib.import_module("18_translate_and_export")
        build_body = export_mod.build_body
        build_document_xml = export_mod.build_document_xml
        write_docx = export_mod.write_docx
        LANGUAGE_NAMES = export_mod.LANGUAGE_NAMES

        
        translated_files = sorted(translated_dir.glob("*.json"))
        pages = []
        
        for i, trans_file in enumerate(translated_files):
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                
                page_num = i + 1
                pages.append({
                    'page_number': page_num,
                    'page_id': f"page_{page_num:04d}",
                    'source_image': f"page_{page_num:04d}.png",
                    'translated_text': data.get('text', ''),
                    'found': True
                })
                
            except Exception as exc:
                logger.error("Failed to read {}: {}", trans_file.name, type(exc).__name__)
        
        body = build_body(pages, source_lang='hin', target_lang=target_lang, page_break=True)
        doc_xml = build_document_xml(body)
        write_docx(output_path, doc_xml, font='Nirmala UI')
        
        logger.info("Step 5 complete: Exported {} pages to {}", len(pages), output_path)
        
    except Exception as exc:
        logger.error("DOCX export failed: {}", type(exc).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description="Run complete PDF to translation pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input PDF file")
    parser.add_argument("--target-language", required=True, help="Target language code (ben, guj, mar, tam, tel, kan, mal, pan, asm, ory, urd)")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline_output"), help="Output directory")
    parser.add_argument("--export-ocr", action="store_true", help="Also export original OCR to DOCX")
    parser.add_argument("--model-name", default=None, help="Gemini model or comma-separated model pool (default: GEMINI_MODEL_POOL env var, else built-in pool)")

    args = parser.parse_args()

    # Set up output directories
    pages_dir = args.output_dir / "pages"
    ocr_dir = args.output_dir / "ocr"
    normalized_dir = args.output_dir / "normalized"
    translated_dir = args.output_dir / "translated"

    logger.info("Starting pipeline for: {}", args.input)
    logger.info("Target language: {}", args.target_language)
    logger.info("Gemini model: {}", args.model_name)

    # Run pipeline steps
    step_1_extract_pages(args.input, pages_dir)
    step_2_run_ocr(pages_dir, ocr_dir)
    step_3_normalize_text(ocr_dir, normalized_dir)
    step_4_translate(normalized_dir, args.target_language, translated_dir, args.model_name)
    step_5_export_docx(translated_dir, args.target_language, args.output_dir / f"translated_{args.target_language}.docx")

    if args.export_ocr:
        logger.info("Exporting original OCR...")
        export_ocr_main = importlib.import_module("17_export_ocr_docx").main
        import sys
        sys.argv = ['export_ocr', '--ocr-root', str(ocr_dir), '--output', str(args.output_dir / "ocr_original.docx")]
        export_ocr_main()

    logger.info("Pipeline complete!")
    logger.info("Output directory: {}", args.output_dir)


if __name__ == "__main__":
    main()
