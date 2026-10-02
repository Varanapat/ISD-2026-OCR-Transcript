import cv2
import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR 3.x engine.

    det_model: "PP-OCRv5_mobile_det" (fast, ~10s/page on CPU) or
    "PP-OCRv5_server_det" (more accurate, ~200s/page on CPU).
    """

    name = "paddle"

    def __init__(self, lang: str = "th", det_model: str = "PP-OCRv5_mobile_det"):
        from paddleocr import PaddleOCR
        # Passing a model name makes PaddleOCR ignore `lang` and fall back to a recognition
        # model without Thai, so resolve the language's recognition model ourselves
        # (e.g. th -> th_PP-OCRv5_mobile_rec, en -> en_PP-OCRv5_mobile_rec).
        _, rec_model = PaddleOCR._get_ocr_model_names(None, lang, None)
        if rec_model is None:
            raise ValueError(f"PaddleOCR has no recognition model for lang={lang!r}")
        self.model = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        # PaddleOCR 3.x requires a 3-channel image; preprocessing returns grayscale.
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lines: list[OCRLine] = []
        for res in self.model.predict(image):
            for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                if not text.strip():
                    continue
                box = [[float(x), float(y)] for x, y in poly]
                lines.append(OCRLine(text=text, confidence=float(score), box=box, engine=self.name, page=page))

        # Paddle does not guarantee reading order; sort top-to-bottom, then left-to-right.
        lines.sort(key=lambda x: (min(p[1] for p in x.box), min(p[0] for p in x.box)))
        return lines
