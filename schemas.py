from typing import Any

from pydantic import BaseModel


class TranscriptResponse(BaseModel):
    filename: str
    content_type: str | None = None
    message: str
    result: dict[str, Any]