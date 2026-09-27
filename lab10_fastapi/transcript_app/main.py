"""Application 2: upload a transcript and extract structured fields."""

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import PROJECT_ROOT, settings

# เชื่อม Lab 10 -> Lab 8A โดยตรง และใช้ Lab 7A ซึ่งเป็น OCR engine ภายใน Lab 8A
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ["OLLAMA_HOST"] = settings.ollama_url
os.environ["LAB7_MODEL_OCR"] = settings.ocr_model
os.environ["LAB7_MODEL_TEXT"] = settings.text_model
os.environ["LAB7_MAX_IMAGE_SIDE"] = str(settings.max_image_side)
from ocr_system import lab8a_denoise as lab8a  # noqa: E402
from ocr_system import lab7a_transcript as lab7a  # noqa: E402

from .pipeline_service import (  # noqa: E402
    ALLOWED_SUFFIXES, TranscriptPipeline,
)
from .schemas import TranscriptResponse


STATIC_DIR = Path(__file__).resolve().parent / "static"
pipeline = TranscriptPipeline(lab7a, lab8a)
app = FastAPI(
    title=f"{settings.app_name} — Transcript",
    description="Upload -> preprocessing -> Typhoon OCR -> Qwen JSON -> postprocessing",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ocr_model": settings.ocr_model,
        "text_model": settings.text_model,
        "max_upload_mb": settings.max_upload_mb,
        "lab8a_module": str(Path(lab8a.__file__).resolve()),
    }


@app.post("/api/transcript/extract", response_model=TranscriptResponse)
async def extract_transcript(
    file: UploadFile = File(...),
    preprocessing: str = Form(default="none"),
    include_markdown: bool = Form(default=False),
) -> dict:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="รองรับเฉพาะ PDF/PNG/JPG/TIFF")

    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"ไฟล์ใหญ่เกิน {settings.max_upload_mb} MB",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="lab10_transcript_") as temp_dir:
            path = Path(temp_dir) / f"upload{suffix}"
            path.write_bytes(content)
            result = await run_in_threadpool(
                pipeline.extract, path, preprocessing, include_markdown
            )
            result["filename"] = file.filename or path.name
            return result
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
