import cv2
import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class DocTREngine(BaseOCREngine):
    """docTR (mindee) engine.

    The pretrained recognition models only cover Latin-script languages, so docTR
    works on English documents but cannot read Thai.
    """

    name = "doctr"

    def __init__(self, det_arch: str = "fast_base", reco_arch: str = "crnn_vgg16_bn", device: str = "cpu"):
        from doctr.models import ocr_predictor
        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        if device != "cpu":
            self.model = self.model.to(device)

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.model([rgb]).pages[0]
        h, w = result.dimensions
        lines: list[OCRLine] = []
        for block in result.blocks:
            for line in block.lines:
                text = " ".join(word.value for word in line.words).strip()
                if not text:
                    continue
                conf = float(np.mean([word.confidence for word in line.words]))
                # docTR geometry is relative (0-1): ((x0, y0), (x1, y1)) for straight pages.
                (x0, y0), (x1, y1) = line.geometry[0], line.geometry[-1]
                x0, x1, y0, y1 = x0 * w, x1 * w, y0 * h, y1 * h
                box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                lines.append(OCRLine(text=text, confidence=conf, box=box, engine=self.name, page=page))
        return lines
