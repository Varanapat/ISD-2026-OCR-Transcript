from pathlib import Path
import numpy as np
from .config import OCRConfig
from .document_loader import load_document_pages
from .engine_factory import build_engine
from .engines.base import BaseOCREngine
from .preprocessing import read_image, preprocess_image, save_debug_image
from .schemas import OCRDocumentResult, OCRLine, OCRPageResult
from .utils.io import ensure_dir, save_json, save_text


def prepare_pages(config: OCRConfig) -> list[tuple[int, Path, np.ndarray]]:
    """Load the document and return (page_no, page_image_path, image_for_ocr) per page."""
    output_dir = ensure_dir(config.output_dir)
    page_dir = ensure_dir(config.page_image_dir)
    pages = load_document_pages(config.input_path, page_dir, dpi=config.dpi)

    prepared = []
    for page_no, image_path in enumerate(pages, start=1):
        image = read_image(image_path)
        if config.preprocess:
            image_for_ocr = preprocess_image(image, deskew=config.deskew)
            if config.save_debug_images:
                debug_path = output_dir / "debug" / f"page_{page_no:03d}_preprocessed.png"
                save_debug_image(image_for_ocr, debug_path)
        else:
            image_for_ocr = image
        prepared.append((page_no, image_path, image_for_ocr))
    return prepared


def build_result(
    config: OCRConfig,
    engine_name: str,
    page_lines: list[tuple[int, Path, list[OCRLine]]],
) -> OCRDocumentResult:
    """Filter lines by confidence, join them into text, and save *_ocr.json / *_ocr.txt."""
    page_results: list[OCRPageResult] = []
    for page_no, image_path, lines in page_lines:
        if config.min_confidence > 0:
            lines = [x for x in lines if x.confidence is None or x.confidence >= config.min_confidence]
        text = "\n".join(line.text for line in lines if line.text.strip())
        page_results.append(OCRPageResult(page=page_no, text=text, lines=lines, image_path=str(image_path)))

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
    page_lines = [
        (page_no, image_path, engine.recognize(image, page=page_no))
        for page_no, image_path, image in pages
    ]
    return build_result(config, engine.name, page_lines)
