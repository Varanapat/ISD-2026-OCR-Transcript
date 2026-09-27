from pathlib import Path

from .config import ALLOWED_EXTENSIONS
from ocr_system.vlm.lab7a_transcript_phet import (
    load_pages,
    pipeline_vlm,
)


def process_transcript(file_path: str | Path):
    """
    Process a transcript PDF with the existing OCR + VLM pipeline.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("รองรับเฉพาะไฟล์ PDF")

    pages = load_pages(path)

    if not pages:
        raise ValueError("ไม่สามารถอ่านหน้า PDF ได้")

    return pipeline_vlm(pages)