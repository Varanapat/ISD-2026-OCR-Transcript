from pathlib import Path
import cv2
from .config import OCRConfig
from .document_loader import load_document_pages
from .engine_factory import build_engine
from .preprocessing import read_image, clean_image, save_debug_image
from .schemas import OCRDocumentResult, OCRPageResult
from .transcript_extraction import extract_transcript_fields, merge_extracted_fields
from .utils.io import ensure_dir, save_json, save_text


def run_ocr(config: OCRConfig) -> OCRDocumentResult:
    output_dir = ensure_dir(config.output_dir)
    page_dir = ensure_dir(config.page_image_dir)
    pages = load_document_pages(config.input_path, page_dir, dpi=config.dpi)
    engine = build_engine(config)

    page_results: list[OCRPageResult] = []
    for page_no, image_path in enumerate(pages, start=1):
        image = read_image(image_path)
        # The cleaning method is chosen explicitly, never inferred from the engine:
        # comparing none/light/heavy only means something when the engine is held fixed.
        image_for_ocr = clean_image(image, method=config.clean_method, deskew=config.deskew)
        if config.save_debug_images and config.clean_method != "none":
            debug_path = output_dir / "debug" / f"page_{page_no:03d}_{config.clean_method}.png"
            save_debug_image(image_for_ocr, debug_path)

        lines = engine.recognize(image_for_ocr, page=page_no)
        if config.min_confidence > 0:
            lines = [x for x in lines if x.confidence is None or x.confidence >= config.min_confidence]

        text = "\n".join(line.text for line in lines if line.text.strip())
        page_results.append(OCRPageResult(page=page_no, text=text, lines=lines, image_path=str(image_path)))

    full_text = "\n\n".join(f"--- Page {p.page} ---\n{p.text}" for p in page_results)
    result = OCRDocumentResult(
        source_path=str(config.input_path),
        engine=engine.name,
        text=full_text,
        pages=page_results,
    )

    stem = Path(config.input_path).stem
    save_json(result.to_dict(), output_dir / f"{stem}_ocr.json")
    save_text(result.text, output_dir / f"{stem}_ocr.txt")
    return result


# The reads the field ensemble combines, primary first. PaddleOCR with unwarping is the
# strongest single reader on the grade table; with unwarping off it keeps the page's own
# coordinates and so still sees the heading and signature block that the unwarped pass
# drops; Tesseract is last because it loses Thai vowels, but it reads what both miss.
ENSEMBLE_PASSES = [
    {"engine": "paddle", "unwarp": True},
    {"engine": "paddle", "unwarp": False},
    {"engine": "tesseract"},
]


def extract_fields(config: OCRConfig, result: OCRDocumentResult) -> dict:
    """Fields for one document. For "ensemble", read it once per pass and merge.

    The merge is at field level, not line level: the engines disagree on how a subject
    row is split across lines, so combining their lines hands the parser a row shape
    neither produced. Measured over 12 transcripts this never scored below the primary
    pass alone, and averaged 2.4 points above it.
    """
    if config.engine != "ensemble":
        return extract_transcript_fields(result.pages)

    from .engines.paddle_engine import PaddleOCREngine
    from .engines.tesseract_engine import TesseractOCREngine

    per_pass: list[dict] = []
    for spec in ENSEMBLE_PASSES:
        if spec["engine"] == "paddle":
            engine = PaddleOCREngine(lang=config.paddle_lang, tile=config.tile,
                                     unwarp=spec["unwarp"])
        else:
            engine = TesseractOCREngine(languages=config.languages)
        pages: list[OCRPageResult] = []
        for page in result.pages:
            image = clean_image(read_image(page.image_path), method=config.clean_method,
                                deskew=config.deskew)
            lines = engine.recognize(image, page=page.page)
            text = "\n".join(l.text for l in lines if l.text.strip())
            pages.append(OCRPageResult(page=page.page, text=text, lines=lines,
                                       image_path=page.image_path))
        per_pass.append(extract_transcript_fields(pages))
    return merge_extracted_fields(per_pass[0], *per_pass[1:])
