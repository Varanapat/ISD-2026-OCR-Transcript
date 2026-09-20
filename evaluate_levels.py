"""
evaluate_levels.py  (Lab 6)
===========================
รัน dataset ทั้งชุดผ่าน OCR + transcript extraction แล้ว evaluate 3 ระดับ:

  Field Level     - ตรงกี่ field จากทุก field ทุกภาพ (micro-average)
  Page Level      - แต่ละภาพ/หน้า ดึงถูกเฉลี่ยกี่ % และกี่หน้าที่ถูกครบ 100%
  Category Level  - แยกตามหมวด header_detail / transcript_detail / footer_detail

วางไฟล์นี้ที่โฟลเดอร์ ocr_system (ระดับเดียวกับ src/, data/)

วิธีใช้
    # dataset = โฟลเดอร์ที่มีคู่ <ชื่อ>.jpg + <ชื่อ>.json (ground truth)
    python evaluate_levels.py dataset/augmented
    python evaluate_levels.py dataset/augmented --lang th --output outputs/lab6_result.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2  # noqa: E402
from ocr_system.engines.paddle_engine import PaddleOCREngine  # noqa: E402
from ocr_system.transcript_extraction import extract_transcript  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def flatten(obj, prefix=""):
    """แบนโครงสร้าง nested เป็น {path: value_as_str}"""
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flat.update(flatten(v, f"{prefix}[{i}]"))
    else:
        flat[prefix] = "" if obj is None else str(obj)
    return flat


def category_of(field_path: str) -> str:
    """หมวด = key ระดับบนสุด เช่น 'transcript_detail.semesters[0]...' -> 'transcript_detail'"""
    return field_path.split(".")[0].split("[")[0]


def compare_one(pred: dict, gt: dict) -> dict:
    """เทียบ 1 ภาพ คืนสถิติระดับ field พร้อม tag หมวด"""
    fp, fg = flatten(pred), flatten(gt)
    keys = sorted(set(fp) | set(fg))
    per_cat = defaultdict(lambda: [0, 0])   # cat -> [matched, total]
    matched = 0
    for k in keys:
        ok = fp.get(k) == fg.get(k)
        matched += ok
        c = category_of(k)
        per_cat[c][0] += ok
        per_cat[c][1] += 1
    return {
        "matched": matched,
        "total": len(keys),
        "accuracy": matched / len(keys) if keys else 0.0,
        "exact": matched == len(keys),
        "per_category": {c: {"matched": m, "total": t} for c, (m, t) in per_cat.items()},
    }


def ocr_to_prediction(engine, image_path: Path, lang: str) -> dict:
    """OCR ภาพ -> โครงสร้าง transcript (predicted)"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"อ่านภาพไม่ได้: {image_path}")
    lines = engine.recognize(img, page=1)
    text = "\n".join(l.text for l in lines if l.text.strip())
    ocr_lines = [{"text": l.text, "box": l.box, "page": l.page,
                  "confidence": l.confidence} for l in lines]
    return extract_transcript(text, ocr_lines, language=lang)


def aggregate(per_page: list[dict]) -> dict:
    """รวมผลทุกภาพเป็น 3 ระดับ"""
    # Field Level (micro): รวม matched/total ทุกภาพ
    tot_m = sum(p["matched"] for p in per_page)
    tot_t = sum(p["total"] for p in per_page)

    # Page Level: เฉลี่ย accuracy ต่อหน้า + นับหน้าที่ถูกครบ
    n = len(per_page)
    mean_page_acc = sum(p["accuracy"] for p in per_page) / n if n else 0.0
    exact_pages = sum(p["exact"] for p in per_page)

    # Category Level: รวมตามหมวด
    cat = defaultdict(lambda: [0, 0])
    for p in per_page:
        for c, v in p["per_category"].items():
            cat[c][0] += v["matched"]
            cat[c][1] += v["total"]

    return {
        "field_level": {
            "matched": tot_m, "total": tot_t,
            "accuracy": round(tot_m / tot_t, 4) if tot_t else 0.0,
        },
        "page_level": {
            "pages": n,
            "mean_accuracy": round(mean_page_acc, 4),
            "exact_match_pages": exact_pages,
            "exact_match_rate": round(exact_pages / n, 4) if n else 0.0,
        },
        "category_level": {
            c: {"matched": m, "total": t, "accuracy": round(m / t, 4) if t else 0.0}
            for c, (m, t) in sorted(cat.items())
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Lab6: evaluate dataset ที่ 3 ระดับ")
    ap.add_argument("dataset_dir", help="โฟลเดอร์ที่มีคู่ <ชื่อ>.jpg + <ชื่อ>.json")
    ap.add_argument("--lang", default="th")
    ap.add_argument("--output", default="outputs/lab6_result.json")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนภาพ (ทดสอบ)")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    pairs = []
    for img in sorted(ds.rglob("*")):
        if img.suffix.lower() in IMG_EXTS and img.with_suffix(".json").exists():
            pairs.append((img, img.with_suffix(".json")))
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"ไม่พบคู่ภาพ+label ใน {ds}")
        return

    print(f"พบ {len(pairs)} คู่ | กำลังโหลดโมเดล PaddleOCR ...")
    engine = PaddleOCREngine(lang=args.lang)

    per_page = []
    for idx, (img, gt_path) in enumerate(pairs, 1):
        gt = json.load(open(gt_path, encoding="utf-8"))
        pred = ocr_to_prediction(engine, img, args.lang)
        stat = compare_one(pred, gt)
        stat["file"] = img.name
        per_page.append(stat)
        print(f"  [{idx}/{len(pairs)}] {img.name:35s} acc={stat['accuracy']*100:5.1f}%")

    report = aggregate(per_page)
    report["per_page"] = [
        {"file": p["file"], "accuracy": round(p["accuracy"], 4), "exact": p["exact"]}
        for p in per_page
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    fl, pl, cl = report["field_level"], report["page_level"], report["category_level"]
    print("\n" + "=" * 55)
    print("LAB 6 EVALUATION")
    print("=" * 55)
    print(f"Field Level    : {fl['accuracy']*100:5.1f}%  ({fl['matched']}/{fl['total']} fields)")
    print(f"Page Level     : {pl['mean_accuracy']*100:5.1f}%  เฉลี่ยต่อหน้า | "
          f"ถูกครบ {pl['exact_match_pages']}/{pl['pages']} หน้า")
    print("Category Level :")
    for c, v in cl.items():
        print(f"    {c:20s} {v['accuracy']*100:5.1f}%  ({v['matched']}/{v['total']})")
    print("=" * 55)
    print(f"\nบันทึกผลละเอียด -> {args.output}")


if __name__ == "__main__":
    main()