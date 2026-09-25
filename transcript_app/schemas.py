"""Response models for transcript extraction."""

from typing import Any

from pydantic import BaseModel


class TranscriptResponse(BaseModel):
    filename: str
    pages: int
    preprocessing: str
    header_detail: dict[str, Any]
    transcript_detail: dict[str, Any]
    footer_detail: dict[str, Any]
    warnings: list[str]
    postprocessing_changes: list[str]
    processing_seconds: float
    markdown: str | None = None

