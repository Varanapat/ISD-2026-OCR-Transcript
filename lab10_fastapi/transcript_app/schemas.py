"""Response models for transcript extraction."""

from typing import Any

from pydantic import BaseModel


class TranscriptResponse(BaseModel): # เป็น Pydantic model บอกว่า response จะมีคีย์อะไร + ชนิดอะไร
    filename: str
    method: str
    pages: int
    preprocessing: str
    header_detail: dict[str, Any]
    transcript_detail: dict[str, Any]
    footer_detail: dict[str, Any]
    warnings: list[str]
    postprocessing_changes: list[str]
    processing_seconds: float
    markdown: str | None = None

# Fast API ใช้สำหรับ : ตรวจว่าเราส่งข้อมูลครบจริง และ สร้างหน้า /docs ให้อัตโนมัติ
# ถ้าฟังก์ชันคืนคีย์ไม่ตรง FastAPI จะ error ทันที ซึ่งเป็นการกันบั๊กแบบ "ส่งข้อมูลผิดรูปแล้วหน้าเว็บพัง