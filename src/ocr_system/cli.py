import argparse
import json
from pathlib import Path
from rich import print
from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_from_files
from .field_extraction import extract_common_fields
from .utils.io import save_json
from .transcript_extraction import extract_transcript_from_file
from .dataset_builder import build_dataset

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

    tr = sub.add_parser("transcript", help="Extract transcript structure from OCR JSON")
    tr.add_argument("ocr_json")
    tr.add_argument("--output", default=None)
    tr.add_argument("--language", default=None)
    tr.add_argument("--ground-truth", default=None)

    ds = sub.add_parser("dataset", help="สร้าง augmented dataset จาก PDF + ground truth")
    ds.add_argument("input_dir", help="โฟลเดอร์ PDF เช่น data/input")
    ds.add_argument("output_dir", help="โฟลเดอร์ผลลัพธ์ เช่น dataset/augmented")
    ds.add_argument("--ground-truth-dir", default="data/ground_truth")
    ds.add_argument("--n", type=int, default=8, help="จำนวนเวอร์ชันต่อภาพ")
    ds.add_argument("--dpi", type=int, default=300)

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
        fields = extract_common_fields(result.text)
        field_path = output_dir / f"{Path(args.input_path).stem}_fields.json"
        save_json(fields, field_path)
        print(f"[green]OCR done[/green]: {output_dir}")
        print(f"Extracted fields: {json.dumps(fields, ensure_ascii=False, indent=2)}")

    elif args.command == "evaluate":
        result = evaluate_from_files(args.ground_truth_json, args.prediction_json)
        save_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "transcript":
        result = extract_transcript_from_file(args.ocr_json, language=args.language)
        out_path = args.output or Path(args.ocr_json).with_name(
            Path(args.ocr_json).stem + "_transcript.json"
        )
        save_json(result, out_path)
        print(f"[green]Transcript extracted[/green]: {out_path}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "dataset":
        build_dataset(args.input_dir, args.output_dir,
                      ground_truth_dir=args.ground_truth_dir,
                      n=args.n, dpi=args.dpi)


if __name__ == "__main__":
    main()
