from .pipeline_service import process_transcript
from .schemas import TranscriptResponse
from .config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, WEB_DIR
from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse


app = FastAPI(title="Transcript OCR API")


@app.get("/")
def root():
    return {
        "message": "Transcript OCR API is running"
    }


@app.get("/web")
def web():
    return FileResponse(
        WEB_DIR / "index.html"
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/api/transcript/upload",
    response_model=TranscriptResponse
)
async def upload_transcript(
    file: UploadFile = File(...)
):
    # ตรวจสอบชื่อไฟล์
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="กรุณาเลือกไฟล์ PDF"
        )

    # ตรวจสอบประเภทไฟล์
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="รองรับเฉพาะไฟล์ PDF"
        )

    temp_path = None

    try:
        # สร้าง temporary file
        suffix = Path(file.filename).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            content = await file.read()
            max_size = MAX_FILE_SIZE_MB * 1024 * 1024

            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"ไฟล์มีขนาดเกิน {MAX_FILE_SIZE_MB} MB"
                )

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="ไฟล์ว่างหรือไม่สามารถอ่านไฟล์ได้"
                )

            temp.write(content)
            temp_path = Path(temp.name)

        # เรียก OCR ผ่าน service
        result = process_transcript(temp_path)

        # ส่งผลลัพธ์กลับ Web
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "OCR completed successfully",
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            f"[ERROR] Transcript OCR failed: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="ไม่สามารถประมวลผล Transcript ได้"
        )

    finally:
        # ลบ temporary file
        if temp_path is not None:
            temp_path.unlink(
                missing_ok=True
            )
@app.post("/api/transcript/extract", response_model=TranscriptResponse)
async def extract_transcript(file: UploadFile = File(...)):
    return await upload_transcript(file)