import argparse
import json
from pathlib import Path
from rich import print
from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_from_files
from .field_extraction import extract_common_fields
from .benchmark import run_benchmark
from .utils.io import save_json


ENGINES = ["paddle", "tesseract", "easyocr", "doctr", "surya", "trocr", "ensemble"]


def add_engine_options(p):
    p.add_argument("--languages", default="tha+eng", help="Tesseract languages, e.g. tha+eng")
    p.add_argument("--paddle-lang", default="th", help="PaddleOCR language, e.g. th or en")
    p.add_argument(
        "--paddle-det-model",
        default="PP-OCRv5_mobile_det",
        help="PaddleOCR detection model: PP-OCRv5_mobile_det (fast) or PP-OCRv5_server_det (slow, more accurate)",
    )
    p.add_argument("--easyocr-langs", default="th,en", help="EasyOCR languages, comma separated (th only pairs with en)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--device", default="cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Thai-English OCR system")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="Run OCR on image or PDF")
    ocr.add_argument("input_path")
    ocr.add_argument("--output-dir", default="outputs")
    ocr.add_argument("--engine", choices=ENGINES, default="ensemble")
    ocr.add_argument("--save-debug-images", action="store_true")
    add_engine_options(ocr)

    bench = sub.add_parser("benchmark", help="Compare OCR engines on a folder of documents with ground truth")
    bench.add_argument("--input-dir", default="data/input")
    bench.add_argument("--ground-truth-dir", default="data/ground_truth")
    bench.add_argument("--output-dir", default="outputs/benchmark")
    bench.add_argument(
        "--engines",
        default="paddle,tesseract,easyocr,doctr,ensemble",
        help=f"Comma separated engines to compare, from: {','.join(ENGINES)}",
    )
    bench.add_argument("--limit", type=int, default=None, help="Only use the first N documents (for a quick test)")
    add_engine_options(bench)

    ev = sub.add_parser("evaluate", help="Evaluate OCR JSON against ground truth JSON")
    ev.add_argument("ground_truth_json")
    ev.add_argument("prediction_json")
    ev.add_argument("--output", default="outputs/evaluation_result.json")

    return parser.parse_args()


def build_config(args, input_path: Path, engine: str) -> OCRConfig:
    output_dir = Path(args.output_dir)
    return OCRConfig(
        input_path=input_path,
        output_dir=output_dir,
        page_image_dir=output_dir / "pages",
        engine=engine,
        languages=args.languages,
        paddle_lang=args.paddle_lang,
        paddle_det_model=args.paddle_det_model,
        easyocr_languages=args.easyocr_langs,
        dpi=args.dpi,
        preprocess=not args.no_preprocess,
        deskew=not args.no_deskew,
        save_debug_images=getattr(args, "save_debug_images", False),
        min_confidence=args.min_confidence,
        device=args.device,
    )


def main():
    args = parse_args()

    if args.command == "ocr":
        output_dir = Path(args.output_dir)
        config = build_config(args, Path(args.input_path), args.engine)
        result = run_ocr(config)
        fields = extract_common_fields(result.text)
        field_path = output_dir / f"{Path(args.input_path).stem}_fields.json"
        save_json(fields, field_path)
        print(f"[green]OCR done[/green]: {output_dir}")
        print(f"Extracted fields: {json.dumps(fields, ensure_ascii=False, indent=2)}")

    elif args.command == "benchmark":
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
        unknown = [e for e in engines if e not in ENGINES]
        if unknown:
            raise SystemExit(f"Unknown engine(s): {', '.join(unknown)}. Choose from: {', '.join(ENGINES)}")
        run_benchmark(
            build_config(args, Path(args.input_dir), engines[0]),
            input_dir=Path(args.input_dir),
            ground_truth_dir=Path(args.ground_truth_dir),
            engines=engines,
            limit=args.limit,
        )

    elif args.command == "evaluate":
        result = evaluate_from_files(args.ground_truth_json, args.prediction_json)
        save_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
