import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent

load_dotenv(APP_DIR / ".env")
WEB_DIR = APP_DIR / "static"

ALLOWED_EXTENSIONS = {".pdf"}

MAX_FILE_SIZE_MB = int(
    os.getenv("MAX_FILE_SIZE_MB", "20")
)

LAB7_DPI = int(
    os.getenv("LAB7_DPI", "150")
)

LAB7_OCR_MODEL = os.getenv(
    "LAB7_OCR_MODEL",
    "scb10x/typhoon-ocr1.5-3b:latest"
)

LAB7_TEXT_MODEL = os.getenv(
    "LAB7_TEXT_MODEL",
    "qwen3:4b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434"
)