from pathlib import Path
from typing import Dict, Any

from .config import OCRConfig
from .pipeline import run_ocr


def extract_full_ocr(
    pdf_path: str,
    engine: str = "ensemble",
    paddle_lang: str = "th",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    รับ Path ของไฟล์ PDF/Image แล้วรัน OCR Engine 
    เพื่อสร้างข้อมูล Full OCR JSON สำหรับนำไปใช้งานต่อ
    """
    input_p = Path(pdf_path)
    
    # 1. ตั้งค่า OCRConfig สำหรับ Pipeline
    config = OCRConfig(
        input_path=input_p,
        engine=engine,  # สามารถเลือก 'paddle', 'tesseract', 'ensemble' ได้
        paddle_lang=paddle_lang,
        device=device
    )

    # 2. ประมวลผลภาพ/PDF ผ่าน Pipeline OCR[cite: 9, 15]
    ocr_result = run_ocr(config)

    # 3. จัดรูปทรงข้อมูลตัวเต็ม (Full OCR Format)
    raw_text = ocr_result.text
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    full_result = {
        "source_path": str(input_p),
        "engine": engine,
        "text": raw_text,  # ข้อความทั้งหมด
        "lines": lines     # ข้อความแยกบรรทัด
    }

    return full_result