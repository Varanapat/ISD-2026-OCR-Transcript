import cv2
import numpy as np
from PIL import Image
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class SuryaOCREngine(BaseOCREngine):
    """Surya OCR 2 (datalab) engine, a vision-language model that reads the whole page.

    Surya runs the model in a local `llama-server` (llama.cpp) on Mac/CPU, or vLLM
    on an NVIDIA GPU. It starts the server on first use (first run downloads ~1.5GB)
    and stops it when the program exits.
    """

    name = "surya"

    def __init__(self):
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        self.predictor = RecognitionPredictor(SuryaInferenceManager())
        self.predictor.disable_tqdm = True

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        from bs4 import BeautifulSoup

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.predictor([Image.fromarray(rgb)], full_page=True)[0]
        lines: list[OCRLine] = []
        for block in sorted(result.blocks, key=lambda b: b.reading_order):
            if block.skipped or block.error or not block.html:
                continue
            # Surya returns each layout block as HTML (paragraphs, tables, ...); keep only the text.
            text = BeautifulSoup(block.html, "html.parser").get_text("\n")
            text = "\n".join(t.strip() for t in text.splitlines() if t.strip())
            if not text:
                continue
            box = [[float(x), float(y)] for x, y in block.polygon]
            lines.append(OCRLine(text=text, confidence=None, box=box, engine=self.name, page=page))
        return lines
