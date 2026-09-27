import argparse
import json
from pathlib import Path
from rich import print
from .config import OCRConfig
from .pipeline import extract_fields, run_ocr
from .evaluation import evaluate_from_files, evaluate_fields_from_files
from .augmentation import augment_document
from .batch_evaluation import run_augmented_dataset
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
    ocr.add_argument(
        "--clean",
        choices=["none", "light", "heavy"],
        default="none",
        help="Image cleaning method applied before OCR (Lab 8A part 3)",
    )
    ocr.add_argument("--no-deskew", action="store_true")
    ocr.add_argument("--save-debug-images", action="store_true")
    ocr.add_argument("--min-confidence", type=float, default=0.0)
    ocr.add_argument("--device", default="cpu")
    ocr.add_argument(
        "--tile",
        action="store_true",
        help="Read the page in overlapping horizontal bands (slower, recovers the "
             "heading and signature block that whole-page downscaling drops)",
    )

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

    batch = sub.add_parser(
        "batch-evaluate",
        help="Run OCR + field extraction over every document/variant under an augmented dataset and "
             "evaluate at field/page/category level",
    )
    batch.add_argument("--augmented-dir", default="data/augmented")
    batch.add_argument("--ground-truth-dir", default="data/ground_truth/ground_truth")
    batch.add_argument("--output", default="outputs/batch_evaluation_report.json")
    batch.add_argument("--limit-documents", type=int, default=None)

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
            clean_method=args.clean,
            deskew=not args.no_deskew,
            save_debug_images=args.save_debug_images,
            min_confidence=args.min_confidence,
            device=args.device,
            tile=args.tile,
        )
        result = run_ocr(config)
        fields = extract_fields(config, result)
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

    elif args.command == "batch-evaluate":
        report = run_augmented_dataset(
            args.augmented_dir,
            args.ground_truth_dir,
            args.output,
            limit_documents=args.limit_documents,
        )
        print(f"[green]Batch evaluation done[/green]: {args.output}")
        print(f"Documents: {report['num_documents']}, pages evaluated: {report['num_pages_evaluated']}")
        for row in report["category_level"]:
            print(
                f"  {row['category']:>10} [{row['language']}]: "
                f"acc {row['avg_accuracy']:.1%} | strict {row['avg_strict_accuracy']:.1%} | "
                f"non-null {row['avg_non_null_accuracy']:.1%} | "
                f"F1 {row['avg_f1']:.1%} (n={row['num_pages']})"
            )


if __name__ == "__main__":
    main()
