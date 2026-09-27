from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EngineName = Literal["paddle", "tesseract", "trocr", "ensemble"]
CleanMethod = Literal["none", "light", "heavy"]


@dataclass
class OCRConfig:
    input_path: Path
    output_dir: Path = Path("outputs")
    engine: EngineName = "ensemble"
    languages: str = "tha+eng"
    paddle_lang: str = "th"
    trocr_model_name: str = "microsoft/trocr-base-printed"
    dpi: int = 300
    clean_method: CleanMethod = "none"
    deskew: bool = True
    save_debug_images: bool = False
    min_confidence: float = 0.0
    device: str = "cpu"
    tile: bool = False
    page_image_dir: Path = field(default_factory=lambda: Path("outputs/pages"))
