from pathlib import Path
from dataclasses import dataclass
from paddleocr import PaddleOCR
import cv2

from .config import OCRConfig
from .document_loader import load_document_pages


@dataclass
class OCRResult:
    text: str


def process_paddle_result(paddle_result: list, image_width: int = 2000) -> str:
    """
    ฟังก์ชันช่วยจัดเรียงบรรทัดและสกัดข้อความจากผลลัพธ์ของ PaddleOCR
    """
    if not paddle_result or not paddle_result[0]:
        return ""

    lines = []
    # ดึงข้อมูล Bounding Box และ Text
    for res in paddle_result[0]:
        box, (text, score) = res[0], res[1]
        top_y = box[0][1]
        left_x = box[0][0]
        lines.append((top_y, left_x, text))

    # เรียงลำดับจากบนลงล่าง (Y) และซ้ายไปขวา (X)
    lines.sort(key=lambda item: (item[0], item[1]))

    return "\n".join([item[2] for item in lines])


def run_ocr(config: OCRConfig, image_paths: list[Path] = None) -> OCRResult:
    """
    ประมวลผล OCR จากไฟล์ภาพที่แปลงเรียบร้อยแล้ว
    """
    # 1. หากไม่ได้ส่ง image_paths มา ให้เรียกใช้ load_document_pages อัตโนมัติ[cite: 20]
    if not image_paths:
        image_paths = load_document_pages(
            input_path=config.input_path,
            output_dir=config.page_image_dir,
            dpi=config.dpi,
        )

    # 2. เริ่มต้นการทำงานของ PaddleOCR Engine[cite: 26]
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang=config.paddle_lang,
        show_log=False
    )

    all_page_texts = []

    # 3. วนลูปอ่าน OCR ทีละภาพ[cite: 20, 26]
    for img_path in image_paths:
        img_path_str = str(img_path)

        # อ่านขนาดภาพ[cite: 26]
        img = cv2.imread(img_path_str)
        image_width = img.shape[1] if img is not None else 2000

        # รัน PaddleOCR[cite: 26]
        paddle_result = ocr_engine.ocr(img_path_str, cls=True)

        # ดึงข้อความ[cite: 26]
        page_text = process_paddle_result(paddle_result, image_width=image_width)
        all_page_texts.append(page_text)

    # รวมข้อความทุกหน้า
    combined_text = "\n".join(all_page_texts)

    return OCRResult(text=combined_text)