"""Application 2: upload a transcript and extract structured fields."""

import os
import shutil
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
os.environ["LAB7_DPI"] = str(settings.dpi)
from ocr_system.vlm import lab8a_denoise as lab8a  # noqa: E402
from ocr_system.vlm import lab7a_transcript as lab7a  # noqa: E402

from .pipeline_service import (  # noqa: E402
    ALLOWED_SUFFIXES, METHODS, OcrTranscriptPipeline, TranscriptPipeline,
)
from .schemas import TranscriptResponse


STATIC_DIR = Path(__file__).resolve().parent / "static"


class NoCacheStaticFiles(StaticFiles):
    """ไฟล์ static ต้องถูกโหลดใหม่ทุกครั้ง ไม่งั้นเบราว์เซอร์อาจใช้ app.js เก่า (เช่น ไม่ส่ง method) ทั้งที่แก้โค้ดแล้ว"""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

pipelines = {
    "vlm": TranscriptPipeline(lab7a, lab8a),
    "ocr": OcrTranscriptPipeline(lab7a, settings.ocr_engine, settings.ocr_languages, settings.dpi),
}
app = FastAPI(
    title=f"{settings.app_name} — Transcript",
    description="Upload -> preprocessing -> Typhoon OCR -> Qwen JSON -> postprocessing",
    version="1.0.0",
)
# ไม่ใช่ API แต่เป็นการบอกว่า URL ที่ขึ้นต้น /static/... ให้ไปหยิบไฟล์จริงในโฟลเดอร์ static/ มาให้ ไม่ต้องเขียน route เอง
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False) # ขอข้อมูล web ตอนพิมพ์ URL
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html") # ตอบด้วย index.html ใน folder static


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ocr_model": settings.ocr_model,
        "text_model": settings.text_model,
        "max_upload_mb": settings.max_upload_mb,
        "ocr_engine": settings.ocr_engine,
        "tesseract_available": shutil.which("tesseract") is not None,
        "lab8a_module": str(Path(lab8a.__file__).resolve()),
    }


@app.post("/api/transcript/extract", response_model=TranscriptResponse) # response model บังคับตาม schema.py
async def extract_transcript(
    file: UploadFile = File(...), # รับไฟล์แบบ multipath ... เป็นการบังคับให้ใส่ ถ้าไม่ใส่ FastAPI จะตอบ 422
    preprocessing: str = Form(default="none"), # รับค่าจาก Form ถ้าไม่ส่งเป็น none
    method: str = Form(default="vlm"), # vlm = Typhoon-OCR + Qwen, ocr = OCR ปกติ (tesseract)
    include_markdown: bool = Form(default=False),
) -> dict:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES: # นามสกุลไม่รองรับ
        raise HTTPException(status_code=415, detail="รองรับเฉพาะ PDF/PNG/JPG/TIFF")

    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        # ไฟล์ใหญ่เกิน
        raise HTTPException(
            status_code=413,
            detail=f"ไฟล์ใหญ่เกิน {settings.max_upload_mb} MB",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="lab10_transcript_") as temp_dir:
            path = Path(temp_dir) / f"upload{suffix}"
            path.write_bytes(content)
            if method not in METHODS:
                raise ValueError(f"method ต้องเป็น {', '.join(METHODS)}")
            result = await run_in_threadpool(
                pipelines[method].extract, path, preprocessing, include_markdown
            )
            result["filename"] = file.filename or path.name
            result["method"] = method
            return result
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
