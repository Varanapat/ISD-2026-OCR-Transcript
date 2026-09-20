"""
dataset_builder.py  (วางที่ src/ocr_system/dataset_builder.py)
-------------------------------------------------------------
คุมงานสร้าง dataset สำหรับ Lab5:
  1. แปลง PDF -> jpg  (ใช้ document_loader ของโปรเจกต์ ไม่เขียนใหม่)
  2. จับคู่ภาพกับ ground truth (label) โดยดูจากรหัสในชื่อไฟล์
  3. augment แต่ละภาพ N เวอร์ชัน + copy label ให้ครบ

ผลลัพธ์: โฟลเดอร์ที่มี <ชื่อ>.jpg คู่กับ <ชื่อ>.json พร้อมอัป Drive ไปส่ง
"""

import json
import re
import shutil
from pathlib import Path

import cv2

from .document_loader import load_document_pages
from .augmentation import augment_once


def _find_ground_truth(stem: str, gt_dir: Path) -> Path | None:
    """หา label ที่ชื่อมีรหัสเดียวกับ pdf เช่น 71010001 -> Json_71010001_th.json"""
    if not gt_dir or not gt_dir.exists():
        return None
    for p in gt_dir.glob("*.json"):
        if stem in p.stem:
            return p
    return None


def build_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    ground_truth_dir: str | Path | None = None,
    n: int = 8,
    dpi: int = 300,
    keep_original: bool = True,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    page_tmp = output_dir / "_pages_tmp"
    page_tmp.mkdir(exist_ok=True)
    gt_dir = Path(ground_truth_dir) if ground_truth_dir else None

    pdfs = sorted(input_dir.glob("*.pdf")) + sorted(input_dir.glob("*.PDF"))
    made = 0
    missing_label = []

    for pdf in pdfs:
        stem = pdf.stem                       # เช่น 71010001
        pages = load_document_pages(pdf, page_tmp, dpi=dpi)   # PDF -> [jpg paths]
        label = _find_ground_truth(stem, gt_dir)
        if label is None:
            missing_label.append(stem)

        for page_no, img_path in enumerate(pages, start=1):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # ถ้า pdf มีหลายหน้า ใส่เลขหน้าในชื่อ, หน้าเดียวไม่ต้อง
            base = stem if len(pages) == 1 else f"{stem}_p{page_no:02d}"

            if keep_original:
                cv2.imwrite(str(output_dir / f"{base}.jpg"), img)
                if label:
                    shutil.copy(label, output_dir / f"{base}.json")

            for i in range(1, n + 1):
                name = f"{base}_aug{i:02d}"
                cv2.imwrite(str(output_dir / f"{name}.jpg"), augment_once(img),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                if label:
                    shutil.copy(label, output_dir / f"{name}.json")
                made += 1

        print(f"[ok] {pdf.name} -> {n} aug/หน้า"
              + ("" if label else "  (ยังไม่มี label!)"))

    # เก็บกวาดภาพชั่วคราว
    shutil.rmtree(page_tmp, ignore_errors=True)

    summary = {"pdfs": len(pdfs), "augmented_images": made,
               "missing_label": missing_label, "output_dir": str(output_dir)}
    print(f"\nเสร็จ: {made} ภาพ ที่ {output_dir}")
    if missing_label:
        print(f"เตือน: ไม่พบ label ของ {', '.join(missing_label)} "
              f"-> ต้องมีไฟล์ json ในโฟลเดอร์ ground truth ที่ชื่อมีรหัสเหล่านี้")
    return summary