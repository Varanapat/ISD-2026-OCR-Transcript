import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class EnsembleOCREngine(BaseOCREngine):
    name = "ensemble"

    def __init__(self, engines: list[BaseOCREngine]):
        self.engines = engines

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        return self.merge(*(engine.recognize(image, page=page) for engine in self.engines))

    @staticmethod
    def merge(*results: list[OCRLine]) -> list[OCRLine]:
        candidates: list[OCRLine] = [line for result in results for line in (result or [])]

        # Simple production-safe default: keep all lines, sorted top-to-bottom if boxes exist.
        # Dedup exact repeated text while preserving stronger confidence.
        best: dict[str, OCRLine] = {}
        for line in candidates:
            key = line.text.strip()
            if not key:
                continue
            if key not in best:
                best[key] = line
            else:
                old_conf = best[key].confidence or 0.0
                new_conf = line.confidence or 0.0
                if new_conf > old_conf:
                    best[key] = line

        lines = list(best.values())
        lines.sort(key=lambda x: _box_top(x.box))
        return lines


def _box_top(box) -> float:
    if not box:
        return 10**9
    try:
        return min(float(p[1]) for p in box)
    except Exception:
        return 10**9
