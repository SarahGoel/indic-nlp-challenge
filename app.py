import sys
import os
import shutil
import uuid
from pathlib import Path

# 1. Add 'src' folder to Python path so 'indicnlp_pipeline' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 2. Load GEMINI_API_KEY from your .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types

# 3. Import your Gemini Translation engine
from indicnlp_pipeline.translation.gemini_engine import (
    GeminiTranslationEngine,
    GeminiEngineError,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini engine once at startup.
# No model is passed: the engine uses its model pool (GEMINI_MODEL_POOL
# env var, or the built-in default) and picks the first available model.
try:
    gemini_translator = GeminiTranslationEngine()
except Exception as init_err:
    print(f"Warning: GeminiTranslationEngine failed to initialize: {init_err}")
    gemini_translator = None

# Map frontend dropdown values to the pipeline's 3-letter codes.
# Only the languages the pipeline actually supports are listed here: the
# engine rejects any other code, so a value missing from this map is a
# client error rather than something to silently coerce.
LANG_MAP = {
    "hindi": "hin",
    "sanskrit": "san",
    "assamese": "asm",
    "bengali": "ben",
    "gujarati": "guj",
    "kannada": "kan",
    "malayalam": "mal",
    "marathi": "mar",
    "odia": "ory",
    "punjabi": "pan",
    "tamil": "tam",
    "telugu": "tel",
    "urdu": "urd",
}

SOURCE_LANGS = {"hin", "san"}
TARGET_LANGS = set(LANG_MAP.values()) - SOURCE_LANGS


def resolve_language(value: str, allowed: set[str], field_name: str) -> str:
    """Resolve a dropdown value to a pipeline code, or raise 400."""
    code = LANG_MAP.get((value or "").strip().lower())
    if code not in allowed:
        supported = ", ".join(sorted(name for name, c in LANG_MAP.items() if c in allowed))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported {field_name}: {value!r}. Supported: {supported}.",
        )
    return code


@app.post("/api/process")
def process_document(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    if gemini_translator is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The translation service is not configured: GEMINI_API_KEY is "
                "missing from the server environment."
            ),
        )

    file_ext = os.path.splitext(file.filename)[1]
    temp_path = PROJECT_ROOT / f"temp_{uuid.uuid4().hex}{file_ext}"

    src_code = resolve_language(source_language, SOURCE_LANGS, "source_language")
    tgt_code = resolve_language(target_language, TARGET_LANGS, "target_language")

    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step A: Extract text (OCR) using Gemini Vision
        try:
            ocr_prompt = (
                f"Extract all text from this handwritten {source_language} manuscript image accurately. "
                "Preserve the exact script, lines, and punctuation. Return only the extracted text."
            )
            # generate() walks the model pool, so OCR uses whatever
            # vision-capable model is actually available for this API key.
            transcription_text = gemini_translator.generate(
                [
                    ocr_prompt,
                    types.Part.from_bytes(
                        data=temp_path.read_bytes(),
                        mime_type=file.content_type or "image/jpeg",
                    ),
                ]
            )
        except Exception as ocr_exc:
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(ocr_exc)}")

        # Step B: Translate extracted text using your GeminiTranslationEngine
        if not transcription_text:
            translation_text = "No text was detected in the uploaded document."
        elif src_code == tgt_code:
            translation_text = transcription_text
        else:
            try:
                translation_text = gemini_translator.translate(
                    text=transcription_text,
                    source_lang=src_code,
                    target_lang=tgt_code,
                )
            except (GeminiEngineError, ValueError) as trans_exc:
                translation_text = f"Translation error: {str(trans_exc)}"

        return {
            "transcription": transcription_text,
            "translation": translation_text
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)

# Serve frontend at root URL
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")