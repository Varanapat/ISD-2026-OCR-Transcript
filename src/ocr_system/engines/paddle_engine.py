import cv2
import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class PaddleOCREngine(BaseOCREngine):
    name = "paddle"
    # PP-OCRv5's detector silently misses text above this height regardless of
    # text_det_limit_side_len, so tall page scans must be downscaled before predict().
    # Raising it does not help and actively breaks detection: on a 2481x3508 scan,
    # 2000 reads 72 lines while 2500/3000/3508 collapse to 3-6.
    MAX_SIDE = 2000
    # Downscaling a full page to MAX_SIDE costs real text: the largest heading and the
    # signature block at the foot of a transcript are both dropped, and tone marks thin
    # out. Cutting the page into overlapping horizontal bands keeps every band under the
    # detector's limit without shrinking the glyphs, which recovers them. The bands
    # overlap because a line sitting on a cut would otherwise be halved and lost.
    BAND_HEIGHT = 1200
    BAND_OVERLAP = 200

    def __init__(self, lang: str = "th", tile: bool = False, unwarp: bool | None = None):
        from paddleocr import PaddleOCR
        # Unwarping dewarps the page before reading, which makes it the stronger reader
        # on average (63.3% vs 61.7% field accuracy over 12 transcripts) -- but it
        # returns boxes in the DEWARPED frame, which does not map back to the image we
        # passed in: the same line lands 140px higher at the top of the page and 186px
        # lower at the foot. So it stays on except when this engine's own boxes are used
        # against the original image, which is what tiling does.
        if unwarp is None:
            unwarp = not tile
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_doc_unwarping=unwarp,
            use_doc_orientation_classify=unwarp,
        )
        self.unwarp = unwarp
        self.tile = tile

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        if not self.tile or image.shape[0] <= self.BAND_HEIGHT:
            return self._recognize_whole(image, page=page)
        return self._recognize_tiled(image, page=page)

    def _recognize_tiled(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        height = image.shape[0]
        step = self.BAND_HEIGHT - self.BAND_OVERLAP
        lines: list[OCRLine] = []
        for top in range(0, height, step):
            bottom = min(top + self.BAND_HEIGHT, height)
            if bottom - top < self.BAND_OVERLAP:
                # A trailing sliver this short is already inside the previous band.
                break
            for line in self._recognize_whole(image[top:bottom, :], page=page):
                if line.box:
                    line.box = [[x, y + top] for x, y in line.box]
                lines.append(line)
        return _dedupe_overlapping(lines)

    def _recognize_whole(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
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


def _box_top(box) -> float:
    if not box:
        return 10 ** 9
    try:
        return min(float(p[1]) for p in box)
    except (TypeError, ValueError, IndexError):
        return 10 ** 9


def _dedupe_overlapping(lines: list[OCRLine], y_tol: int = 20) -> list[OCRLine]:
    """Drop the second reading of a line that fell inside a band overlap.

    The same physical line read from two adjacent bands lands at almost the same
    absolute y once the band offset is applied, so identical text within y_tol is a
    duplicate rather than a genuine repeat. The strongest reading wins.
    """
    kept: list[OCRLine] = []
    for line in sorted(lines, key=lambda l: -(l.confidence or 0.0)):
        y = _box_top(line.box)
        if any(k.text == line.text and abs(_box_top(k.box) - y) <= y_tol for k in kept):
            continue
        kept.append(line)
    kept.sort(key=lambda l: _box_top(l.box))
    return kept
