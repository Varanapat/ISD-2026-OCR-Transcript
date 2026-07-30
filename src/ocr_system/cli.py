import argparse
import json
from pathlib import Path
from rich import print
from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_from_files, evaluate_fields_from_files
from .transcript_extraction import extract_transcript_fields
from .augmentation import augment_document
from .utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Thai-English OCR system")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="Run OCR on image or PDF")
    ocr.add_argument("input_path")
    ocr.add_argument("--output-dir", default="outputs")
    ocr.add_argument("--engine", choices=["paddle", "tesseract", "trocr", "ensemble"], default="ensemble")
    ocr.add_argument("--languages", default="tha+eng", help="Tesseract languages, e.g. tha+eng")
    ocr.add_argument("--paddle-lang", default="th", help="PaddleOCR language, e.g. th or en")
    ocr.add_argument("--dpi", type=int, default=300)
    ocr.add_argument("--no-preprocess", action="store_true")
    ocr.add_argument("--no-deskew", action="store_true")
    ocr.add_argument("--save-debug-images", action="store_true")
    ocr.add_argument("--min-confidence", type=float, default=0.0)
    ocr.add_argument("--device", default="cpu")

    ev = sub.add_parser("evaluate", help="Evaluate OCR JSON against ground truth JSON")
    ev.add_argument("ground_truth_json")
    ev.add_argument("prediction_json")
    ev.add_argument("--output", default="outputs/evaluation_result.json")

    aug = sub.add_parser("augment", help="Generate rotated/skewed/blurred/noisy image variants for OCR robustness testing")
    aug.add_argument("input_path")
    aug.add_argument("--output-dir", default="data/augmented")
    aug.add_argument("--ground-truth-dir", default="data/ground_truth/ground_truth")
    aug.add_argument("--dpi", type=int, default=300)
    aug.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ocr":
        output_dir = Path(args.output_dir)
        config = OCRConfig(
            input_path=Path(args.input_path),
            output_dir=output_dir,
            page_image_dir=output_dir / "pages",
            engine=args.engine,
            languages=args.languages,
            paddle_lang=args.paddle_lang,
            dpi=args.dpi,
            preprocess=not args.no_preprocess,
            deskew=not args.no_deskew,
            save_debug_images=args.save_debug_images,
            min_confidence=args.min_confidence,
            device=args.device,
        )
        result = run_ocr(config)
        fields = extract_transcript_fields(result.pages)
        field_path = output_dir / f"{Path(args.input_path).stem}_fields.json"
        save_json(fields, field_path)
        print(f"[green]OCR done[/green]: {output_dir}")
        print(f"Extracted fields: {json.dumps(fields, ensure_ascii=False, indent=2)}")

    elif args.command == "evaluate":
        with open(args.ground_truth_json, encoding="utf-8") as f:
            gt_preview = json.load(f)
        if isinstance(gt_preview, dict) and "header_detail" in gt_preview:
            result = evaluate_fields_from_files(args.ground_truth_json, args.prediction_json)
        else:
            result = evaluate_from_files(args.ground_truth_json, args.prediction_json)
        save_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "augment":
        manifest = augment_document(
            args.input_path,
            args.output_dir,
            ground_truth_dir=args.ground_truth_dir,
            dpi=args.dpi,
            seed=args.seed,
        )
        num_variants = sum(len(p["variants"]) for p in manifest["pages"])
        print(f"[green]Augmentation done[/green]: {Path(args.output_dir) / Path(args.input_path).stem}")
        print(f"Generated {num_variants} variants across {len(manifest['pages'])} page(s)")


if __name__ == "__main__":
    main()
