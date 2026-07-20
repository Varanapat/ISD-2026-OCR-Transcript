# import numpy as np
# from .base import BaseOCREngine
# from ocr_system.schemas import OCRLine


# class PaddleOCREngine(BaseOCREngine):
#     name = "paddle"

#     def __init__(self, lang: str = "th"):
#         from paddleocr import PaddleOCR
#         self.model = PaddleOCR(use_angle_cls=True, lang=lang)

#     def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
#         result = self.model.ocr(image, cls=True)

#         lines: list[OCRLine] = []
#         for block in result or []:
#             for item in block or []:
#                 box = item[0]
#                 text = item[1][0]
#                 conf = float(item[1][1])
#                 lines.append(OCRLine(text=text, confidence=conf, box=box, engine=self.name, page=page))
#         return lines

import numpy as np
from .base import BaseOCREngine
from ocr_system.schemas import OCRLine


class PaddleOCREngine(BaseOCREngine):
    name = "paddle"

    def __init__(self, lang: str = "th"):
        from paddleocr import PaddleOCR
        # 3.x: use_angle_cls -> use_textline_orientation
        # self.model = PaddleOCR(use_textline_orientation=True, lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False)
        self.model = PaddleOCR(lang=lang,
                               use_textline_orientation=True,
                               use_doc_orientation_classify=False,
                               use_doc_unwarping=False,
                               text_det_limit_side_len=1536,
                               text_det_limit_type="max",
    )

    def recognize(self, image: np.ndarray, page: int | None = None) -> list[OCRLine]:
        # 3.x: เมธอด ocr() ไม่รับ cls แล้ว
        # result = self.model.ocr(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        result = self.model.ocr(image)
        lines: list[OCRLine] = []

        for res in result or []:
            # ---- PaddleOCR 3.x: res เป็น dict ----
            if hasattr(res, "get") and "rec_texts" in res:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                boxes = res.get("rec_polys")
                if boxes is None:
                    boxes = res.get("dt_polys", [])
                for text, conf, box in zip(texts, scores, boxes):
                    box_list = box.tolist() if hasattr(box, "tolist") else box
                    lines.append(OCRLine(text=text, confidence=float(conf),
                                         box=box_list, engine=self.name, page=page))
            # ---- PaddleOCR 2.x: res เป็น list ของ [box, (text, conf)] ----
            else:
                for item in res or []:
                    box = item[0]
                    text = item[1][0]
                    conf = float(item[1][1])
                    lines.append(OCRLine(text=text, confidence=conf,
                                         box=box, engine=self.name, page=page))
        return lines