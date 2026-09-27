"""Preprocessing -> OCR -> structured extraction -> postprocessing."""

import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _preprocess_pages(pages: list[bytes], method: str,
                      lab8: ModuleType) -> list[bytes]:
    if method == "none":
        return pages
    output: list[bytes] = []
    for page in pages:
        image = cv2.imdecode(np.frombuffer(page, dtype=np.uint8), cv2.IMREAD_COLOR) # decode เป็น array
        if image is None:
            raise ValueError("OpenCV อ่านหน้าของเอกสารไม่ได้")
        cleaned = lab8.clean_image(image, method) # เรียก method clean image (none / light / heavy)
        ok, encoded = cv2.imencode(".png", cleaned) # encode กลับเป็น png
        if not ok:
            raise ValueError("OpenCV แปลงภาพหลัง preprocessing ไม่สำเร็จ")
        output.append(encoded.tobytes())
    return output


def _postprocess(record: dict, lab8: ModuleType) -> list[str]:
    """Apply deterministic constraints to critical transcript fields."""
    changes: list[str] = []
    transcript = record.get("transcript_detail") or {}
    for si, semester in enumerate(transcript.get("semesters") or []):
        for ci, subject in enumerate(semester.get("subject") or []):
            fields = {
                # เรียก lab8.... แล้วบันทึกว่าแก้อะไรไปบ้าง
                "subject_id": lab8.snap_code(subject.get("subject_id"), None),
                "credit": lab8.snap_credits(subject.get("credit")),
                "grade_earn": lab8.snap_grade(subject.get("grade_earn")),
            }
            for key, value in fields.items():
                if subject.get(key) != value:
                    changes.append(
                        f"semesters[{si}].subject[{ci}].{key}: "
                        f"{subject.get(key)!r} -> {value!r}"
                    )
                    subject[key] = value
    return changes


class TranscriptPipeline: # ลำดับงานทั้งหมด
    def __init__(self, lab7: ModuleType, lab8: ModuleType):
        self.lab7 = lab7
        self.lab8 = lab8

    def extract(self, path: Path, preprocessing: str,
                include_markdown: bool = False) -> dict[str, Any]:
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("รองรับเฉพาะ PDF, PNG, JPG และ TIFF")
        if preprocessing not in {"none", "light", "heavy"}:
            raise ValueError("preprocessing ต้องเป็น none, light หรือ heavy")

        started = time.time()
        self.lab7.assert_offline() # กันข้อมูลออกนอกเครื่อง (PDPA)
        pages = self.lab7.load_pages(str(path)) # pdf -> png
        cleaned_pages = _preprocess_pages(pages, preprocessing, self.lab8) # clean

        # markdown, record เก่า
        # markdown = self.lab7.ocr_pages_to_markdown(cleaned_pages) # OCR แล้วจัดเป็น json
        # record = self.lab7.structure_markdown(markdown) # OCR แล้วจัดเป็น json

        # markdown, record ใหม่
        md_path = path.with_suffix(".md") if include_markdown else None
        record = self.lab7.pipeline_vlm(cleaned_pages, save_md=md_path)
        markdown = md_path.read_text(encoding="utf-8") if md_path and md_path.exists() else None # .exists() ระบุว่า ไฟล์/โฟลเดอร์นี้มีอยู่ไหม ส่งกลับมาเป็น True/Fasle

        changes = _postprocess(record, self.lab8)
        verification = self.lab7.verify_internal(record)

        return {
            "filename": path.name,
            "pages": len(pages),
            "preprocessing": preprocessing,
            "header_detail": record.get("header_detail") or {},
            "transcript_detail": record.get("transcript_detail") or {},
            "footer_detail": record.get("footer_detail") or {},
            "warnings": verification.get("issues") or [],
            "postprocessing_changes": changes,
            "processing_seconds": round(time.time() - started, 1),
            "markdown": markdown if include_markdown else None,
        }
