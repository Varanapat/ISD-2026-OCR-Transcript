import numpy as np
from .base import BaseOCREngine
from .paddle_engine import PaddleOCREngine
from .tesseract_engine import TesseractOCREngine
from ocr_system.schemas import OCRLine


class EnsembleOCREngine(BaseOCREngine):
    """PaddleOCR as the primary reader, Tesseract filling the bands it left blank.

    The two engines fail in different places. PaddleOCR reads the grade table well but
    must downscale a tall page before its detector will run, and at that size it misses
    the largest heading and the signature block entirely. Tesseract reads those, needs
    no downscale, and is an order of magnitude faster -- but it drops Thai vowels and
    scatters spaces inside words, so it is the weaker reader wherever Paddle sees text.

    Hence the rule: keep every Paddle line, and add a Tesseract line only where no
    Paddle line occupies that horizontal band. Taking the better of two readings of the
    SAME line would mix the engines' different row layouts inside the subject table and
    leave the extractor parsing a shape neither engine produces.
    """

    name = "ensemble"
    # A Tesseract line counts as already covered when this much of its height overlaps
    # some Paddle line. Well under half, because a band Paddle read at all is a band
    # where its reading should win.
    COVERAGE = 0.35

    def __init__(self, paddle_lang: str = "th", tesseract_languages: str = "tha+eng",
                 tile: bool = False):
        self.paddle = PaddleOCREngine(lang=paddle_lang, tile=tile)
        self.tesseract = TesseractOCREngine(languages=tesseract_languages)
        self.engines = [self.paddle, self.tesseract]

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        return self.merge(
            self.paddle.recognize(image, page=page),
            self.tesseract.recognize(image, page=page),
        )

    def merge(
        self,
        paddle_result: list[OCRLine],
        tesseract_result: list[OCRLine] | None = None,
        trocr_result: list[OCRLine] | None = None,
    ) -> list[OCRLine]:
        primary = [l for l in (paddle_result or []) if l.text.strip()]
        spans = [_y_span(l.box) for l in primary]
        spans = [s for s in spans if s]

        filled: list[OCRLine] = list(primary)
        for line in (tesseract_result or []) + (trocr_result or []):
            if not line.text.strip():
                continue
            span = _y_span(line.box)
            if span is None or _covered(span, spans, self.COVERAGE):
                continue
            filled.append(line)

        filled.sort(key=lambda l: (_y_span(l.box) or (10 ** 9, 10 ** 9))[0])
        return filled


def _y_span(box) -> tuple[float, float] | None:
    if not box:
        return None
    try:
        ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return None
    return (min(ys), max(ys))


def _covered(span: tuple[float, float], spans: list[tuple[float, float]], threshold: float) -> bool:
    top, bottom = span
    height = bottom - top
    if height <= 0:
        return False
    for other_top, other_bottom in spans:
        overlap = min(bottom, other_bottom) - max(top, other_top)
        if overlap > 0 and overlap / height >= threshold:
            return True
    return False
