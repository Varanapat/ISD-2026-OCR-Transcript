import argparse
import json
from pathlib import Path
from rich import print

from .config import OCRConfig
from .pipeline import run_ocr
from .evaluation import evaluate_from_files
from .field_extraction import extract_fields_from_full
from .document_loader import load_document_pages
from .utils.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Thai-English OCR system")
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ocr
    ocr = sub.add_parser("ocr", help="Run OCR on image or PDF")
    ocr.add_argument("input_path", help="Path to input PDF or Image file")
    ocr.add_argument("--output-dir", default="outputs", help="Output directory")
    ocr.add_argument(
        "--engine",
        choices=["paddle", "tesseract", "trocr", "ensemble"],
        default="paddle",
        help="OCR engine to use",
    )
    ocr.add_argument(
        "--languages", default="tha+eng", help="Tesseract languages, e.g. tha+eng"
    )
    ocr.add_argument(
        "--paddle-lang", type=str, default="th", help="Language for PaddleOCR"
    )
    ocr.add_argument("--dpi", type=int, default=300, help="DPI for PDF conversion")
    ocr.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")
    ocr.add_argument("--no-deskew", action="store_true", help="Disable deskewing")
    ocr.add_argument(
        "--save-debug-images", action="store_true", help="Save intermediate debug images"
    )
    ocr.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Filter text by confidence threshold",
    )
    ocr.add_argument("--device", default="cpu", help="Device to run model (cpu/cuda)")

    # Subcommand: evaluate
    ev = sub.add_parser(
        "evaluate", help="Evaluate OCR JSON against ground truth JSON"
    )
    ev.add_argument("ground_truth_json", help="Path to Ground Truth JSON")
    ev.add_argument("prediction_json", help="Path to Prediction JSON")
    ev.add_argument(
        "--output",
        default="outputs/evaluation_result.json",
        help="Path to save evaluation result",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ocr":
        input_path = Path(args.input_path)
        output_dir = Path(args.output_dir)
        page_image_dir = output_dir / "pages"

        # สร้าง Output Directory หากยังไม่มี
        output_dir.mkdir(parents=True, exist_ok=True)
        page_image_dir.mkdir(parents=True, exist_ok=True)

        # 1. แปลงไฟล์ PDF เป็นรูปภาพ JPEG (หรือคืนค่าเดิมถ้าอินพุตเป็นรูปภาพอยู่แล้ว)[cite: 20]
        saved_page_images = load_document_pages(
            input_path=input_path, output_dir=page_image_dir, dpi=args.dpi
        )

        # 2. ตั้งค่า OCRConfig[cite: 18, 19]
        config = OCRConfig(
            input_path=input_path,
            output_dir=output_dir,
            page_image_dir=page_image_dir,
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

        # 3. รัน OCR โดยส่ง list รูปภาพที่แปลงได้เข้าไปประมวลผล[cite: 18, 20]
        result = run_ocr(config, image_paths=saved_page_images)

        # 4. บันทึกผลลัพธ์ Full OCR Text ลงไฟล์ _full.json[cite: 18, 26]
        full_text_json_path = output_dir / f"{input_path.stem}_full.json"
        full_ocr_data = {
            "source_path": str(input_path),
            "engine": args.engine,
            "text": result.text,
            "lines": result.text.split("\n"),
        }
        save_json(full_ocr_data, full_text_json_path)

        # 5. สกัดโครงสร้างฟิลด์ลงไฟล์ _fields.json
        fields = extract_fields_from_full(full_ocr_data)
        field_path = output_dir / f"{input_path.stem}_fields.json"
        save_json(fields, field_path)

        # แสดงสรุปผลการทำงาน
        print(f"\n[bold green]✓ OCR Process Finished Successfully![/bold green]")
        print(f"[cyan]• Input File:[/cyan] {input_path}")
        print(f"[cyan]• Extracted Image Pages:[/cyan] {[str(p) for p in saved_page_images]}")
        print(f"[cyan]• Full OCR JSON Saved:[/cyan] {full_text_json_path}")
        print(f"[cyan]• Fields JSON Saved:[/cyan] {field_path}\n")

    elif args.command == "evaluate":
        result = evaluate_from_files(args.ground_truth_json, args.prediction_json)
        save_json(result, args.output)
        print("\n[bold green]✓ Evaluation Completed![/bold green]")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()