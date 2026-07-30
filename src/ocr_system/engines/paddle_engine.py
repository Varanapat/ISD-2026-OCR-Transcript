import cv2
import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class PaddleOCREngine(BaseOCREngine):
    name = "paddle"
    # PP-OCRv5's detector silently misses text above this height regardless of
    # text_det_limit_side_len, so tall page scans must be downscaled before predict().
    MAX_SIDE = 2000

    def __init__(self, lang: str = "th"):
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(use_angle_cls=True, lang=lang)

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = min(1.0, self.MAX_SIDE / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        results = self.model.predict(image)
        lines: list[OCRLine] = []
        for res in results or []:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", [])
            for text, score, poly in zip(texts, scores, polys):
                box = poly.tolist() if hasattr(poly, "tolist") else poly
                if scale < 1.0 and box:
                    box = [[x / scale, y / scale] for x, y in box]
                lines.append(OCRLine(text=text, confidence=float(score), box=box, engine=self.name, page=page))
        return lines
