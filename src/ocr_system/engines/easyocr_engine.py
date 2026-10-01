import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine. Thai ("th") can only be combined with English ("en")."""

    name = "easyocr"

    def __init__(self, languages: str = "th,en", device: str = "cpu"):
        import easyocr
        self.reader = easyocr.Reader(
            [lang.strip() for lang in languages.split(",") if lang.strip()],
            gpu=device != "cpu",
            verbose=False,
        )

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for box, text, conf in self.reader.readtext(image):
            if not text.strip():
                continue
            box = [[float(x), float(y)] for x, y in box]
            lines.append(OCRLine(text=text, confidence=float(conf), box=box, engine=self.name, page=page))
        return lines
