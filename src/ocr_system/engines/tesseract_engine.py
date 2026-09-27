import cv2
import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class TesseractOCREngine(BaseOCREngine):
    name = "tesseract"

    def __init__(self, languages: str = "tha+eng", psm: int = 6):
        import pytesseract
        self.pytesseract = pytesseract
        self.languages = languages
        self.config = f"--oem 3 --psm {psm}"

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = self.pytesseract.image_to_data(
            rgb,
            lang=self.languages,
            config=self.config,
            output_type=self.pytesseract.Output.DICT,
        )
        return self._group_into_lines(data, page=page)

    def _group_into_lines(self, data: dict, page: int | None = None) -> list[OCRLine]:
        """Rebuild text lines from Tesseract's word boxes.

        Tesseract splits Thai into one "word" per glyph cluster, so joining its tokens
        with a space yields "พ ี ช ค ณิ ต" instead of "พีชคณิต" and every Thai keyword
        the extractor anchors on stops matching. The pixel gap between neighbouring
        boxes tells the two apart: a real word break is wide, a split glyph is not.
        """
        rows: dict[tuple, list[dict]] = {}
        for i, raw in enumerate(data["text"]):
            text = (raw or "").strip()
            if not text:
                continue
            rows.setdefault(
                (data["block_num"][i], data["par_num"][i], data["line_num"][i]), []
            ).append({
                "text": text,
                "left": data["left"][i], "top": data["top"][i],
                "width": data["width"][i], "height": data["height"][i],
                "conf": data["conf"][i],
            })

        lines: list[OCRLine] = []
        for _, words in sorted(rows.items()):
            words.sort(key=lambda w: w["left"])
            parts = [words[0]["text"]]
            for prev, word in zip(words, words[1:]):
                gap = word["left"] - (prev["left"] + prev["width"])
                # A gap under a third of the glyph height is a split character, not a
                # space between words.
                parts.append(word["text"] if gap <= prev["height"] * 0.33 else " " + word["text"])
            confs = []
            for w in words:
                try:
                    confs.append(float(w["conf"]) / 100.0)
                except (TypeError, ValueError):
                    pass
            x0 = min(w["left"] for w in words)
            y0 = min(w["top"] for w in words)
            x1 = max(w["left"] + w["width"] for w in words)
            y1 = max(w["top"] + w["height"] for w in words)
            lines.append(OCRLine(
                text="".join(parts).strip(),
                confidence=sum(confs) / len(confs) if confs else None,
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                engine=self.name,
                page=page,
            ))
        return lines
