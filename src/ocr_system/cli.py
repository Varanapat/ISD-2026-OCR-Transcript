import argparse
import json
from pathlib import Path
from rich import print
from rich.markup import escape
from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_from_files
from .field_extraction import extract_common_fields
from .transcript_extraction import extract_transcript_from_file, extract_transcript_from_ocr
from .benchmark import run_benchmark
from .augmentation import AUGMENTATIONS, augment_dataset
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
    p.add_argument(
        "--deskew-only",
        action="store_true",
        help="Only straighten tilted pages; skip denoise/contrast/threshold (recommended for digital PDFs)",
    )
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--layout", action="store_true", help="Split each page into regions (header, info, table columns, footer)")
    p.add_argument(
        "--layout-mode",
        choices=["auto", "assign", "crop"],
        default="auto",
        help="assign: OCR whole page then label boxes by region; crop: OCR each region separately; "
        "auto: crop for doctr/surya/trocr, assign for the others",
    )


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

    aug = sub.add_parser("augment", help="Create an augmented, labeled dataset (one image per augmentation)")
    aug.add_argument("--input-dir", default="data/input")
    aug.add_argument("--ground-truth-dir", default="data/ground_truth")
    aug.add_argument("--output-dir", default="data/Augmentation_input")
    aug.add_argument(
        "--augmentations",
        default=",".join(AUGMENTATIONS),
        help=f"Comma separated, from: {','.join(AUGMENTATIONS)}",
    )
    aug.add_argument("--seed", type=int, default=42, help="Same seed = same random values = same images")
    aug.add_argument("--dpi", type=int, default=300)
    aug.add_argument("--no-original", action="store_true", help="Do not add the clean page as control image")

    ext = sub.add_parser("extract", help="Extract a structured transcript JSON from an OCR result (*_ocr.json)")
    ext.add_argument("ocr_json", help="OCR result from the ocr command, e.g. outputs/71010001_ocr.json")
    ext.add_argument("--output", default=None, help="Default: <ocr_json folder>/<name>_transcript.json")
    ext.add_argument("--language", choices=["th", "en"], default=None, help="Default: detect from the text")

    ev = sub.add_parser("evaluate", help="Evaluate OCR JSON or transcript JSON against ground truth JSON")
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
        preprocess=not args.no_preprocess and not args.deskew_only,
        deskew=not args.no_deskew,
        deskew_only=args.deskew_only,
        save_debug_images=getattr(args, "save_debug_images", False),
        min_confidence=args.min_confidence,
        device=args.device,
        layout=args.layout,
        layout_mode=args.layout_mode,
    )


def print_transcript_summary(transcript: dict, path: Path) -> None:
    header = transcript["header_detail"]
    semesters = transcript["transcript_detail"]["semesters"]
    subjects = sum(len(s["subject"]) for s in semesters)
    print(f"[green]Transcript extracted[/green]: {path}")
    print(f"  student_id={header.get('student_id')}  name={header.get('name')}  "
          f"semesters={len(semesters)}  subjects={subjects}")


def print_evaluation_summary(result: dict) -> None:
    fields = result if "field_accuracy" in result else result.get("fields")
    if "field_recall" in result:
        print(f"Text:   field_recall={result['field_recall']:.3f} ({result['field_found']}/{result['field_total']})  "
              f"cer={result['cer']:.3f}")
    elif "cer" in result:
        print(f"Text:   cer={result['cer']:.3f}  wer={result['wer']:.3f}  exact_match={result['exact_match']}")
    if fields:
        print(f"Fields: field_accuracy={fields['field_accuracy']:.3f} ({fields['fields_correct']}/{fields['fields_total']})")
        for section, s in fields["by_section"].items():
            print(f"  {section:18} {s['accuracy']:.3f} ({s['correct']}/{s['total']})")
        for m in fields["mismatches"][:10]:
            print(f"  [red]x[/red] {escape(m['field'])}: expected={escape(repr(m['expected']))} predicted={escape(repr(m['predicted']))}")
        if len(fields["mismatches"]) > 10:
            print(f"  ... {len(fields['mismatches']) - 10} more mismatches")


def main():
    args = parse_args()
    if getattr(args, "deskew_only", False) and getattr(args, "no_deskew", False):
        raise SystemExit("--deskew-only and --no-deskew cannot be used together")

    if args.command == "ocr":
        output_dir = Path(args.output_dir)
        config = build_config(args, Path(args.input_path), args.engine)
        result = run_ocr(config)
        stem = Path(args.input_path).stem
        fields = extract_common_fields(result.text)
        save_json(fields, output_dir / f"{stem}_fields.json")
        transcript = extract_transcript_from_ocr(result.to_dict())
        save_json(transcript, output_dir / f"{stem}_transcript.json")
        print(f"[green]OCR done[/green]: {output_dir}")
        print(f"Extracted fields: {json.dumps(fields, ensure_ascii=False, indent=2)}")
        print_transcript_summary(transcript, output_dir / f"{stem}_transcript.json")

    elif args.command == "augment":
        names = [a.strip() for a in args.augmentations.split(",") if a.strip()]
        unknown = [a for a in names if a not in AUGMENTATIONS]
        if unknown:
            raise SystemExit(f"Unknown augmentation(s): {', '.join(unknown)}. Choose from: {', '.join(AUGMENTATIONS)}")
        augment_dataset(
            Path(args.input_dir),
            Path(args.ground_truth_dir),
            Path(args.output_dir),
            augmentations=names,
            seed=args.seed,
            dpi=args.dpi,
            include_original=not args.no_original,
        )

    elif args.command == "extract":
        ocr_json = Path(args.ocr_json)
        output = Path(args.output) if args.output else ocr_json.with_name(ocr_json.stem.removesuffix("_ocr") + "_transcript.json")
        transcript = extract_transcript_from_file(ocr_json, language=args.language)
        save_json(transcript, output)
        print_transcript_summary(transcript, output)

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
        print_evaluation_summary(result)
        print(f"Full result (incl. every mismatched field): {args.output}")


if __name__ == "__main__":
    main()
