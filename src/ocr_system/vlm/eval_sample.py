"""
สุ่มเอกสารชนิดละ N ฉบับ (ปริญญาตรี/บัณฑิต x ไทย/อังกฤษ) รัน pipeline VLM แล้วเทียบกับเฉลย

    python src/ocr_system/vlm/eval_sample.py                 # สุ่ม 4 ฉบับ (ชนิดละ 1)
    python src/ocr_system/vlm/eval_sample.py --dry-run       # ดูว่าสุ่มได้ไฟล์ไหน ยังไม่รัน
    python src/ocr_system/vlm/eval_sample.py --seed 7        # สุ่มชุดอื่น (ค่าเริ่มต้น seed=1)
    python src/ocr_system/vlm/eval_sample.py --per-type 2    # ชนิดละ 2 ฉบับ
    python src/ocr_system/vlm/eval_sample.py --docs 73046011,74166001   # ระบุเอง ไม่สุ่ม

ค่าเริ่มต้นไม่สุ่มเอกสาร 4 ฉบับที่ใช้ปรับระบบ (ใช้ --include-dev ถ้าต้องการรวม)
ใช้เงื่อนไขเดียวกับหน้าเว็บ: DPI 300, ไม่ทำความสะอาดภาพ (none)

ผลลัพธ์ (มีแต่ตัวเลข ไม่มีชื่อ/เกรดนักศึกษา) บันทึกที่ outputs/eval_sample/ ซึ่งถูก git ยกเว้นอยู่แล้ว
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("LAB7_DPI", "300")          # ต้องตั้งก่อน import lab7a (มันอ่านค่าตอน import)
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data"
DEV_DOCS = {"71010001", "72100002", "73036003", "74106005"}      # ชุดที่ใช้ปรับ prompt/normalizer
# (โฟลเดอร์ PDF, โฟลเดอร์เฉลย, ชื่อกลุ่ม)
SOURCES = [("input", "ground_truth", "undergrad"), ("input_G", "ground_truth_G", "grad")]
# ฟิลด์สำคัญ: ผิดแล้วข้อมูลเสียหายจริง (ชื่อกลุ่มตาม stats ของ lab7a.evaluate)
CRITICAL = {"student_id": "รหัสนศ.", "subject_id": "รหัสวิชา", "credit": "หน่วยกิต",
            "grade": "เกรด", "gpa": "GPS/GPA", "summary": "สรุปสะสม"}
USABLE_THRESHOLD = 0.95        # เอกสาร "ใช้ได้จริง" = ฟิลด์สำคัญถูกอย่างน้อยเท่านี้ (แก้ได้ด้วย --usable)


def discover() -> tuple[dict, list[str]]:
    """คืน ({(group, lang): [(doc_id, pdf, gt)]}, รายการที่จับคู่ไม่ได้)"""
    types: dict[tuple[str, str], list] = {}
    problems: list[str] = []
    for in_dir, gt_dir, group in SOURCES:
        pdfs = {p.stem: p for p in (DATA / in_dir).glob("*.pdf")}
        gts = {}
        for g in (DATA / gt_dir).glob("Json_*_*.json"):
            _, doc_id, lang = g.stem.split("_")
            gts[doc_id] = (g, lang.lower())
        for doc_id in sorted(pdfs.keys() - gts.keys()):
            problems.append(f"{group}: มี PDF {doc_id} แต่ไม่มีเฉลย")
        for doc_id in sorted(gts.keys() - pdfs.keys()):
            problems.append(f"{group}: มีเฉลย {doc_id} แต่ไม่มี PDF")
        for doc_id in sorted(pdfs.keys() & gts.keys()):
            g, lang = gts[doc_id]
            types.setdefault((group, lang), []).append((doc_id, pdfs[doc_id], g))
    return types, problems


def pick(types: dict, per_type: int, seed: int, include_dev: bool) -> list[tuple]:
    rng = random.Random(seed)
    chosen = []
    for key in sorted(types):
        pool = [d for d in types[key] if include_dev or d[0] not in DEV_DOCS]
        if len(pool) < per_type:
            print(f"⚠ ชนิด {key}: เหลือให้สุ่ม {len(pool)} ฉบับ น้อยกว่า {per_type}")
        for d in rng.sample(pool, min(per_type, len(pool))):
            chosen.append((key, *d))
    return chosen


def critical_metrics(stats: dict) -> dict:
    """ความถูกต้องของฟิลด์สำคัญ รวม (micro) และแยกรายฟิลด์"""
    per = {}
    for k in CRITICAL:
        st = stats.get(k)
        per[k] = round(st.n_exact / st.n_items, 4) if st and st.n_items else None
    items = sum(stats[k].n_items for k in CRITICAL if k in stats)
    exact = sum(stats[k].n_exact for k in CRITICAL if k in stats)
    return {"crit_acc": round(exact / items, 4) if items else None,
            "crit_items": items, **{f"crit_{k}": v for k, v in per.items()}}


def run_one(lab7, L8, doc_id: str, pdf: Path, gt_path: Path, outdir: Path, usable: float) -> dict:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    pages = lab7.load_pages(str(pdf))
    t0 = time.time()
    pred = lab7.pipeline_vlm(pages, save_md=outdir / f"{doc_id}.md")
    seconds = time.time() - t0
    (outdir / f"{doc_id}_pred.json").write_text(
        json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    stats, align = lab7.evaluate(pred, gt)
    om = lab7.overall_metrics(stats)
    fa = L8.field_accuracy(pred, gt)
    ver = lab7.verify_internal(pred)
    sub = align["subject"]
    crit = critical_metrics(stats)
    return {
        "doc_id": doc_id,
        "seconds": round(seconds, 1),
        "accuracy": om["accuracy"], "cer": om["cer"], "wer": om["wer"],
        "fields": om["fields"], "fields_exact": om["fields_exact"],
        "ref_chars": om["ref_chars"], "ref_words": om["ref_words"],
        "wer_tokenizer": om["wer_tokenizer"],
        "hallucinated": fa["hallucinated"],
        "subj_precision": round(sub["precision"], 4), "subj_recall": round(sub["recall"], 4),
        "subj_f1": round(sub["f1"], 4),
        "subj_missed": sub["missed"], "subj_spurious": sub["spurious"],
        "internal_ok": ver["ok"], "grade_null": ver["n_grade_null"],
        **crit,
        "usable": crit["crit_acc"] is not None and crit["crit_acc"] >= usable,
    }


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--docs", default="", help="doc_id คั่นด้วย , (ข้ามการสุ่ม)")
    ap.add_argument("--include-dev", action="store_true", help="ให้สุ่มรวมเอกสารที่ใช้ปรับระบบด้วย")
    ap.add_argument("--usable", type=float, default=USABLE_THRESHOLD,
                    help="เกณฑ์ฟิลด์สำคัญถูก (สัดส่วน) ที่นับว่าเอกสาร 'ใช้ได้จริง' (ค่าเริ่มต้น %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="แค่แสดงว่าสุ่มได้ไฟล์ไหน")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "eval_sample"))
    args = ap.parse_args()

    types, problems = discover()
    for p in problems:
        print("⚠", p)
    if args.docs:
        want = [d.strip() for d in args.docs.split(",") if d.strip()]
        chosen = [(k, *d) for k, v in sorted(types.items()) for d in v if d[0] in want]
        missing = set(want) - {c[1] for c in chosen}
        if missing:
            raise SystemExit(f"❌ ไม่พบ (หรือจับคู่ PDF/เฉลยไม่ได้): {', '.join(sorted(missing))}")
    else:
        chosen = pick(types, args.per_type, args.seed, args.include_dev)

    print(f"\nเอกสารที่เลือก ({len(chosen)} ฉบับ, seed={args.seed}):")
    for (group, lang), doc_id, pdf, gt in chosen:
        mark = "  (ชุดที่ใช้ปรับระบบ)" if doc_id in DEV_DOCS else ""
        print(f"  {group:9s} {lang}  {doc_id}{mark}")
    if args.dry_run:
        return

    import lab7a_transcript as lab7
    import lab8a_denoise as L8
    lab7.assert_offline()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (group, lang), doc_id, pdf, gt in chosen:
        print(f"\n{'=' * 70}\n  {doc_id}  ({group}/{lang})\n{'=' * 70}")
        try:
            r = run_one(lab7, L8, doc_id, pdf, gt, outdir, args.usable)
        except Exception as e:                      # เอกสารหนึ่งล้ม ไม่ควรทำให้ทั้งชุดล้ม
            print(f"  ❌ {doc_id} ล้มเหลว: {e}")
            r = {"doc_id": doc_id, "error": str(e)}
        r.update(group=group, lang=lang)
        rows.append(r)

    ok = [r for r in rows if "error" not in r]
    print(f"\n{'=' * 78}\n  สรุป (เงื่อนไข: DPI {lab7.DPI}, ไม่ทำความสะอาดภาพ)\n{'=' * 78}")
    head = (f"{'doc':9s} {'type':12s} {'acc':>6s} {'crit':>6s} {'grade':>6s} {'cer':>6s} {'wer':>6s} "
            f"{'hall':>5s} {'subjF1':>6s} {'miss':>4s} {'spur':>4s} {'sec':>6s} {'ok?':>4s}")
    print(head + "\n" + "-" * len(head))
    for r in rows:
        if "error" in r:
            print(f"{r['doc_id']:9s} {r['group'] + '/' + r['lang']:12s}  ล้มเหลว")
            continue
        print(f"{r['doc_id']:9s} {r['group'] + '/' + r['lang']:12s} {fmt(r['accuracy']):>6s} {fmt(r['crit_acc']):>6s} "
              f"{fmt(r['crit_grade']):>6s} {fmt(r['cer']):>6s} "
              f"{fmt(r['wer']):>6s} {r['hallucinated']:>5d} {fmt(r['subj_f1']):>6s} "
              f"{r['subj_missed']:>4d} {r['subj_spurious']:>4d} {r['seconds']:>6.0f} {'✓' if r['usable'] else '✗':>4s}")
    if ok:
        n = len(ok)
        w = [r["wer"] for r in ok if r["wer"] is not None]
        print("-" * len(head))
        crit_vals = [r["crit_acc"] for r in ok if r["crit_acc"] is not None]
        grade_vals = [r["crit_grade"] for r in ok if r["crit_grade"] is not None]
        print(f"{'เฉลี่ย':9s} {'(macro)':12s} {sum(r['accuracy'] for r in ok) / n:>6.3f} "
              f"{(sum(crit_vals) / len(crit_vals)) if crit_vals else float('nan'):>6.3f} "
              f"{(sum(grade_vals) / len(grade_vals)) if grade_vals else float('nan'):>6.3f} "
              f"{sum(r['cer'] for r in ok) / n:>6.3f} {(sum(w) / len(w)) if w else float('nan'):>6.3f} "
              f"{sum(r['hallucinated'] for r in ok) / n:>5.1f} {sum(r['subj_f1'] for r in ok) / n:>6.3f} "
              f"{'':>4s} {'':>4s} {sum(r['seconds'] for r in ok) / n:>6.0f}")
        n_ok = sum(1 for r in ok if r["usable"])
        print(f"\n  เอกสารที่ใช้ได้จริง (ฟิลด์สำคัญถูก >= {args.usable:.0%}): {n_ok}/{len(rows)} ฉบับ"
              f"   [ฟิลด์สำคัญ: {', '.join(CRITICAL.values())}]")
        print(f"  ช้าสุด {max(r['seconds'] for r in ok):.0f} วินาที | ต่ำสุดของ accuracy {min(r['accuracy'] for r in ok):.3f}"
              f" | WER ใช้ตัวตัดคำ: {ok[0]['wer_tokenizer']}")
        if "fallback" in str(ok[0]["wer_tokenizer"]):
            print("  ⚠ WER ของภาษาไทยยังหยาบ (ไม่ได้ติดตั้ง pythainlp)")

    (outdir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = sorted({k for r in rows for k in r})
    with open(outdir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n  บันทึก: {outdir / 'summary.csv'} , {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
