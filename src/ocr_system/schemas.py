from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class OCRLine:
    text: str
    confidence: float | None = None
    box: Any | None = None
    engine: str | None = None
    page: int | None = None
    region: str | None = None  # layout region name when --layout is used


@dataclass
class OCRPageResult:
    page: int
    text: str
    lines: list[OCRLine]
    image_path: str
    regions: list[dict] | None = None  # layout regions (name + box) when --layout is used


@dataclass
class OCRDocumentResult:
    source_path: str
    engine: str
    text: str
    pages: list[OCRPageResult]

    def to_dict(self) -> dict:
        return asdict(self)
