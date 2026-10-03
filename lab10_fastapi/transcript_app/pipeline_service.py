"""Preprocessing -> OCR -> structured extraction -> postprocessing."""

import shutil
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
METHODS = ("vlm", "ocr")     # vlm = Typhoon-OCR + Qwen (ช้า แต่ทนเค้าโครงหลากหลาย) · ocr = OCR ปกติ (เร็ว)


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

        verification = self.lab7.verify_internal(record)
        warnings = list(verification.get("issues") or [])
        # กฎ snap ของ Lab 8A อยู่ใน lab8a_denoise.py ถ้าเป็นเวอร์ชันที่ไม่มีฟังก์ชันเหล่านี้ ห้ามให้ทั้ง request ล้ม
        # แต่ต้องบอกผู้ใช้ชัด ๆ ว่าขั้นนี้ถูกข้าม (ไม่ข้ามเงียบ ๆ)
        missing = [n for n in ("snap_code", "snap_credits", "snap_grade") if not hasattr(self.lab8, n)]
        if missing:
            changes = []
            warnings.append("ข้าม postprocessing ของ Lab 8A: lab8a_denoise.py ไม่มี " + ", ".join(missing))
        else:
            changes = _postprocess(record, self.lab8)

        return {
            "filename": path.name,
            "pages": len(pages),
            "preprocessing": preprocessing,
            "header_detail": record.get("header_detail") or {},
            "transcript_detail": record.get("transcript_detail") or {},
            "footer_detail": record.get("footer_detail") or {},
            "warnings": warnings,
            "postprocessing_changes": changes,
            "processing_seconds": round(time.time() - started, 1),
            "markdown": markdown if include_markdown else None,
        }


class OcrTranscriptPipeline:
    """โหมด OCR ปกติ: tesseract (ocr_system.pipeline) -> สกัดข้อมูลด้วยกฎ (ocr_system.transcript_extraction)

    ผลลัพธ์มีโครงเดียวกับ TranscriptPipeline เพื่อให้หน้าเว็บใช้ร่วมกันได้
    ทุกไฟล์ที่ run_ocr เขียน (ภาพหน้า, *_ocr.json, *_ocr.txt) ถูกชี้ไปไว้ในโฟลเดอร์ชั่วคราวข้าง ๆ ไฟล์อัปโหลด
    ซึ่ง main.py ลบทิ้งเมื่อ request จบ (PDPA) ส่วนการรันผ่าน CLI ไม่ได้ผ่านที่นี่ จึงยังเขียนที่ outputs/ ตามเดิม
    """

    def __init__(self, lab7: ModuleType, engine: str, languages: str, dpi: int):
        self.lab7 = lab7
        self.engine_name = engine
        self.languages = languages
        self.dpi = dpi
        self._engine = None
        self._lock = threading.Lock()   # สร้าง engine ครั้งเดียว แม้มีหลาย request เข้าพร้อมกัน

    def _engine_for(self, config):
        with self._lock:
            if self._engine is None:
                from ocr_system.engine_factory import build_engine
                self._engine = build_engine(config)
            return self._engine

    def extract(self, path: Path, preprocessing: str,
                include_markdown: bool = False) -> dict[str, Any]:
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("รองรับเฉพาะ PDF, PNG, JPG และ TIFF")
        if preprocessing not in {"none", "light", "heavy"}:
            raise ValueError("preprocessing ต้องเป็น none, light หรือ heavy")
        if self.engine_name == "tesseract" and shutil.which("tesseract") is None:
            raise RuntimeError("ไม่พบโปรแกรม tesseract ในเครื่อง (ติดตั้งด้วย brew install tesseract tesseract-lang)")

        # import ตรงนี้ เพื่อให้โหมด VLM ใช้ได้แม้เครื่องไม่มีไลบรารี OCR ของโหมดนี้
        from ocr_system.config import OCRConfig
        from ocr_system.pipeline import run_ocr
        from ocr_system.transcript_extraction import extract_transcript_from_ocr

        started = time.time()
        work = path.parent / "ocr_work"            # อยู่ใต้โฟลเดอร์ชั่วคราวของ request นี้
        # ใช้ช่อง preprocessing เดิม แต่ความหมายตามของ ocr_system:
        #   none = ภาพดิบ · light = แค่ปรับภาพเอียง (deskew) · heavy = ปรับภาพเอียง + ทำความสะอาดเต็มรูปแบบ
        config = OCRConfig(
            input_path=path, output_dir=work, page_image_dir=work / "pages",
            engine=self.engine_name, languages=self.languages, dpi=self.dpi,
            preprocess=(preprocessing == "heavy"), deskew=True,
            deskew_only=(preprocessing == "light"),
        )
        try:
            result = run_ocr(config, self._engine_for(config))
            record = extract_transcript_from_ocr(result.to_dict())
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:                   # เช่น tesseract ล้ม ให้ตอบ 422 พร้อมสาเหตุ ไม่ใช่ 500
            raise RuntimeError(f"โหมด OCR ล้มเหลว: {exc}") from exc

        verification = self.lab7.verify_internal(record)
        return {
            "filename": path.name,
            "pages": len(result.pages),
            "preprocessing": preprocessing,
            "header_detail": record.get("header_detail") or {},
            "transcript_detail": record.get("transcript_detail") or {},
            "footer_detail": record.get("footer_detail") or {},
            "warnings": verification.get("issues") or [],
            "postprocessing_changes": [],          # โหมดนี้ไม่ผ่านกฎ snap ของ Lab 8A
            "processing_seconds": round(time.time() - started, 1),
            "markdown": result.text if include_markdown else None,   # โหมดนี้ไม่มี Markdown ใช้ข้อความ OCR แทน
        }
