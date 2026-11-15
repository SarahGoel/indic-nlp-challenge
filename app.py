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

# Initialize Gemini engine once at startup
try:
    gemini_translator = GeminiTranslationEngine(model_name="gemini-2.5-flash")
except Exception as init_err:
    print(f"Warning: GeminiTranslationEngine failed to initialize: {init_err}")
    gemini_translator = None

# Map frontend dropdown values to your pipeline's 3-letter codes
LANG_MAP = {
    "hindi": "hin",
    "sanskrit": "san",
    "bengali": "ben",
    "gujarati": "guj",
    "marathi": "mar",
    "tamil": "tam",
    "telugu": "tel",
    "kannada": "kan",
    "malayalam": "mal",
    "punjabi": "pan",
    "assamese": "asm",
    "odia": "ory",
    "urdu": "urd",
    "english": "eng",
}

@app.post("/api/process")
def process_document(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    if gemini_translator is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing or invalid. Please check your .env file."
        )

    file_ext = os.path.splitext(file.filename)[1]
    temp_path = PROJECT_ROOT / f"temp_{uuid.uuid4().hex}{file_ext}"
    
    src_code = LANG_MAP.get(source_language.lower(), "hin")
    tgt_code = LANG_MAP.get(target_language.lower(), "eng")

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
            response = gemini_translator.client.models.generate_content(
                model=gemini_translator.model_name,
                contents=[
                    ocr_prompt,
                    types.Part.from_bytes(
                        data=temp_path.read_bytes(),
                        mime_type=file.content_type or "image/jpeg",
                    ),
                ],
            )
            transcription_text = (response.text or "").strip()
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