from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import OCRConfig
from .document_loader import load_document_pages
from .engine_factory import build_engine
from .engines.base import BaseOCREngine
from .layout import Region, assign_regions, detect_regions, group_text_by_region, recognize_by_crop, save_layout_image
from .preprocessing import deskew_image, read_image, preprocess_image, save_debug_image
from .schemas import OCRDocumentResult, OCRLine, OCRPageResult
from .utils.io import ensure_dir, save_json, save_text

# Engines whose boxes are large (long lines / whole blocks) and often span several
# layout regions: with --layout-mode auto they OCR each region separately (method A).
# All other engines OCR the whole page and their boxes are assigned to regions (method B).
CROP_ENGINES = {"doctr", "surya", "trocr"}


@dataclass
class PreparedPage:
    page: int
    image_path: Path
    image: np.ndarray  # image passed to the OCR engine (after preprocessing, if enabled)
    regions: list[Region] | None = None  # layout regions, only with --layout


def prepare_pages(config: OCRConfig) -> list[PreparedPage]:
    """Load the document, preprocess each page and (with --layout) detect its regions."""
    output_dir = ensure_dir(config.output_dir)
    page_dir = ensure_dir(config.page_image_dir)
    pages = load_document_pages(config.input_path, page_dir, dpi=config.dpi)

    prepared = []
    for page_no, image_path in enumerate(pages, start=1):
        image = read_image(image_path)
        if config.deskew_only:
            image_for_ocr = deskew_image(image)
        elif config.preprocess:
            image_for_ocr = preprocess_image(image, deskew=config.deskew)
        else:
            image_for_ocr = image
        if config.save_debug_images and image_for_ocr is not image:
            debug_path = output_dir / "debug" / f"page_{page_no:03d}_preprocessed.png"
            save_debug_image(image_for_ocr, debug_path)

        regions = None
        if config.layout:
            # Detect on the same image the engine sees, so box coordinates match.
            regions = detect_regions(image_for_ocr)
            if config.save_debug_images:
                save_layout_image(image_for_ocr, regions, output_dir / "debug" / f"page_{page_no:03d}_layout.png")
        prepared.append(PreparedPage(page_no, image_path, image_for_ocr, regions))
    return prepared


def layout_mode_for(config: OCRConfig, engine_name: str) -> str:
    if config.layout_mode != "auto":
        return config.layout_mode
    return "crop" if engine_name in CROP_ENGINES else "assign"


def recognize_page(engine: BaseOCREngine, page: PreparedPage, config: OCRConfig) -> list[OCRLine]:
    if page.regions is None:
        return engine.recognize(page.image, page=page.page)
    if layout_mode_for(config, engine.name) == "crop":
        return recognize_by_crop(engine, page.image, page.regions, page=page.page)
    return assign_regions(engine.recognize(page.image, page=page.page), page.regions)


def build_result(
    config: OCRConfig,
    engine_name: str,
    page_lines: list[tuple[PreparedPage, list[OCRLine]]],
) -> OCRDocumentResult:
    """Filter lines by confidence, join them into text, and save *_ocr.json / *_ocr.txt."""
    page_results: list[OCRPageResult] = []
    for page, lines in page_lines:
        if config.min_confidence > 0:
            lines = [x for x in lines if x.confidence is None or x.confidence >= config.min_confidence]
        if page.regions is None:
            text = "\n".join(line.text for line in lines if line.text.strip())
            regions = None
        else:
            text = group_text_by_region(lines, page.regions)
            regions = [r.to_dict() for r in page.regions]
        page_results.append(
            OCRPageResult(page=page.page, text=text, lines=lines, image_path=str(page.image_path), regions=regions)
        )

    full_text = "\n\n".join(f"--- Page {p.page} ---\n{p.text}" for p in page_results)
    result = OCRDocumentResult(
        source_path=str(config.input_path),
        engine=engine_name,
        text=full_text,
        pages=page_results,
    )

    output_dir = ensure_dir(config.output_dir)
    stem = Path(config.input_path).stem
    save_json(result.to_dict(), output_dir / f"{stem}_ocr.json")
    save_text(result.text, output_dir / f"{stem}_ocr.txt")
    return result


def run_ocr(config: OCRConfig, engine: BaseOCREngine | None = None) -> OCRDocumentResult:
    pages = prepare_pages(config)
    engine = engine or build_engine(config)
    page_lines = [(page, recognize_page(engine, page, config)) for page in pages]
    return build_result(config, engine.name, page_lines)
