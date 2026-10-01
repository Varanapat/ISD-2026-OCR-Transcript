from .config import OCRConfig
from .engines.base import BaseOCREngine

ENSEMBLE_MEMBERS = ["paddle", "tesseract", "easyocr"]


def build_engine(config: OCRConfig, name: str | None = None) -> BaseOCREngine:
    # Engines are imported lazily so that e.g. tesseract runs without paddle/easyocr loaded.
    name = name or config.engine
    if name == "paddle":
        from .engines.paddle_engine import PaddleOCREngine
        return PaddleOCREngine(lang=config.paddle_lang, det_model=config.paddle_det_model)
    if name == "tesseract":
        from .engines.tesseract_engine import TesseractOCREngine
        return TesseractOCREngine(languages=config.languages)
    if name == "easyocr":
        from .engines.easyocr_engine import EasyOCREngine
        return EasyOCREngine(languages=config.easyocr_languages, device=config.device)
    if name == "doctr":
        from .engines.doctr_engine import DocTREngine
        return DocTREngine(device=config.device)
    if name == "surya":
        from .engines.surya_engine import SuryaOCREngine
        return SuryaOCREngine()
    if name == "trocr":
        from .engines.trocr_engine import TrOCREngine
        return TrOCREngine(model_name=config.trocr_model_name, device=config.device)
    if name == "ensemble":
        from .engines.ensemble_engine import EnsembleOCREngine
        return EnsembleOCREngine([build_engine(config, member) for member in ENSEMBLE_MEMBERS])
    raise ValueError(f"Unknown OCR engine: {name}")
