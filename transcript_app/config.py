"""Transcript App configuration loaded from transcript_app/.env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
load_dotenv(APP_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("TRANSCRIPT_APP_NAME", "Transcript OCR Application")
    ollama_url: str = os.getenv("TRANSCRIPT_OLLAMA_URL", "http://127.0.0.1:11434")
    ocr_model: str = os.getenv(
        "TRANSCRIPT_OCR_MODEL", "scb10x/typhoon-ocr1.5-3b"
    )
    text_model: str = os.getenv("TRANSCRIPT_TEXT_MODEL", "qwen3:4b")
    max_upload_mb: int = int(os.getenv("TRANSCRIPT_MAX_UPLOAD_MB", "20"))
    max_image_side: int = int(os.getenv("TRANSCRIPT_MAX_IMAGE_SIDE", "1800"))


settings = Settings()

