from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EngineName = Literal["paddle", "tesseract", "easyocr", "doctr", "surya", "trocr", "ensemble"]


@dataclass
class OCRConfig:
    input_path: Path
    output_dir: Path = Path("outputs")
    engine: EngineName = "ensemble"
    languages: str = "tha+eng"
    paddle_lang: str = "th"
    paddle_det_model: str = "PP-OCRv5_mobile_det"
    easyocr_languages: str = "th,en"
    trocr_model_name: str = "microsoft/trocr-base-printed"
    dpi: int = 300
    preprocess: bool = True
    deskew: bool = True
    deskew_only: bool = False  # straighten the page but skip all other preprocessing
    save_debug_images: bool = False
    min_confidence: float = 0.0
    device: str = "cpu"
    layout: bool = False
    layout_mode: str = "auto"  # "auto" | "assign" (OCR page, then label boxes) | "crop" (OCR each region)
    page_image_dir: Path = field(default_factory=lambda: Path("outputs/pages"))
