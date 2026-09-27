"""Lab 8A entry point: check / selftest / noise / sweep.

Runs both as a module (python -m ocr_system.lab8a_denoise) and as a plain script
(python src/ocr_system/lab8a_denoise.py), which is how the lab sheet invokes it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Direct script execution: relative imports need the package root on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ocr_system"

from ocr_system.evaluation import char_error_rate, compare_fields, strip_diacritics
from ocr_system.lab8a_sweep import CLEAN_METHODS, format_grid, run_sweep
from ocr_system.noise_levels import NOISE_LEVELS, apply_level, generate_noisy_document
from ocr_system.preprocessing import clean_image


# ------------------------------------------------ constrained correction
# Lab 10 calls these after the model has produced JSON: each pulls one field back
# onto the set of values that can actually appear on a transcript. They are
# deliberately deterministic -- a rule that can be checked by reading it, not a
# second model whose mistakes would be as hard to audit as the first model's.
#
# All three return None rather than a guess when the value cannot be reached by a
# known correction. A null says "this was not read"; a plausible-looking wrong value
# says "this was read, and it is this" -- which is the failure that hallucination
# counts exist to catch.

SUBJECT_CODE_LENGTH = 8
MAX_CREDITS = 12
VALID_GRADES = {"a", "b+", "b", "c+", "c", "d+", "d", "f",
                "w", "s", "u", "i", "p", "t"}
# OCR reads these letters as their lookalike digits, especially on a binarised page.
GRADE_DIGIT_FIX = {"0": "d", "8": "b", "5": "s", "1": "i", "6": "g"}


def snap_grade(value) -> str | None:
    """Pull a grade onto the set of grades a transcript can carry."""
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value)).lower()
    if not text:
        return None
    if text in VALID_GRADES:
        return text
    swapped = GRADE_DIGIT_FIX.get(text[0], text[0]) + text[1:]
    return swapped if swapped in VALID_GRADES else None


def snap_credits(value) -> int | None:
    """Credits are a small whole number; anything else did not survive the read."""
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    credits = int(match.group())
    return credits if 0 <= credits <= MAX_CREDITS else None


def snap_code(value, known_codes=None) -> str | None:
    """Keep a subject code only at its real length; with a course list, repair one digit.

    Without `known_codes` there is nothing to check a code against beyond its shape, so
    a code of the wrong length is dropped rather than padded into something plausible.
    With a list, a code that differs from exactly one known code in a single position is
    that code misread -- but two candidates means the repair is a coin flip, so it is
    left alone.
    """
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if known_codes:
        if digits in known_codes:
            return digits
        close = [c for c in known_codes
                 if len(c) == len(digits)
                 and sum(a != b for a, b in zip(c, digits)) == 1]
        if len(close) == 1:
            return close[0]
    return digits if len(digits) == SUBJECT_CODE_LENGTH else None


# ----------------------------------------------------------------- check
def cmd_check() -> int:
    ok = True
    print("Python:", sys.version.split()[0], "" if sys.version_info >= (3, 10) else "<- ต้อง 3.10+")
    for mod, fix in [("cv2", "pip install opencv-python"), ("numpy", "pip install numpy"),
                     ("paddleocr", "pip install paddleocr paddlepaddle"),
                     ("pdf2image", "pip install pdf2image  (+ brew install poppler)"),
                     ("jiwer", "pip install jiwer"), ("Levenshtein", "pip install python-Levenshtein")]:
        try:
            __import__(mod)
            print(f"  [ok]      {mod}")
        except ImportError:
            ok = False
            print(f"  [ขาด]     {mod:<14} แก้ด้วย: {fix}")
    import shutil
    if shutil.which("pdftoppm"):
        print("  [ok]      poppler (pdftoppm)")
    else:
        ok = False
        print("  [ขาด]     poppler        แก้ด้วย: brew install poppler")
    print(f"  [info]    noise levels: {', '.join(NOISE_LEVELS)}")
    print(f"  [info]    clean methods: {', '.join(CLEAN_METHODS)}")
    print("\nพร้อมทำแล็บ" if ok else "\nยังไม่พร้อม -- ติดตั้งของที่ขาดก่อน")
    return 0 if ok else 1


# -------------------------------------------------------------- selftest
def _gt(subjects, major=None):
    return {"header_detail": {"student_id": "71010001", "major": major,
                              "name": "ทดสอบ"},
            "transcript_detail": {"semesters": [
                {"year": 2561, "sem_num": 1, "GPA": "2.50",
                 "subject": [{"subject_id": s, "subject_name": n, "type": None,
                              "credit": 3, "grade_earn": g} for s, n, g in subjects]}]}}


def _checks():
    import copy
    import numpy as np

    two = _gt([("03386101", "คณิตศาสตร์", "c"), ("03386102", "ฟิสิกส์", "b")])
    one = _gt([("03386101", "คณิตศาสตร์", "c")])
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    img[100:150, 50:350] = 20

    def tone_stripped(tree):
        if isinstance(tree, dict):
            return {k: tone_stripped(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [tone_stripped(v) for v in tree]
        return strip_diacritics(tree)

    swapped = copy.deepcopy(two)
    swapped["transcript_detail"]["semesters"][0]["subject"].reverse()
    extra = copy.deepcopy(two)
    extra["transcript_detail"]["semesters"][0]["subject"].append(
        {"subject_id": "99999999", "subject_name": "ขยะ", "type": None, "credit": 3, "grade_earn": "a"})
    filled = copy.deepcopy(two)
    filled["header_detail"]["major"] = "ใส่มาเอง"

    return [
        ("เทียบกับตัวเอง = 1.0", lambda: compare_fields(two, two)["accuracy"] == 1.0),
        ("ผลลัพธ์ว่าง = 0.0 (ไม่มีคะแนนฟรีจาก null)",
         lambda: compare_fields(two, {})["accuracy"] == 0.0),
        ("วิชาหาย 1 -> พังเฉพาะวิชานั้น",
         lambda: compare_fields(two, one)["total_fields"] - compare_fields(two, one)["matched_fields"] == 5),
        ("สลับลำดับวิชา -> ยังตรง 100%", lambda: compare_fields(two, swapped)["accuracy"] == 1.0),
        ("วิชาเกิน -> recall คงที่", lambda: compare_fields(two, extra)["accuracy"] == 1.0),
        ("วิชาเกิน -> precision ตก", lambda: compare_fields(two, extra)["precision"] < 1.0),
        ("วิชาเกิน -> extra_fields = 5", lambda: len(compare_fields(two, extra)["extra_fields"]) == 5),
        ("hall นับช่องที่เฉลยว่างแต่ถูกกรอก",
         lambda: len(compare_fields(two, filled)["hallucinated_fields"]) == 1),
        ("hall = 0 เมื่อตรงกันเป๊ะ",
         lambda: len(compare_fields(two, two)["hallucinated_fields"]) == 0),
        ("วรรณยุกต์หาย -> acc = 1.0",
         lambda: compare_fields(two, tone_stripped(copy.deepcopy(two)))["accuracy"] == 1.0),
        ("วรรณยุกต์หาย -> strict < 1.0",
         lambda: compare_fields(two, tone_stripped(copy.deepcopy(two)))["strict_accuracy"] < 1.0),
        ("normalize=False -> acc = strict",
         lambda: (lambda r: r["accuracy"] == r["strict_accuracy"])(
             compare_fields(two, tone_stripped(copy.deepcopy(two)), normalize=False))),
        ("strip_diacritics ลบวรรณยุกต์", lambda: strip_diacritics("เกล้า") == "เกลา"),
        ("strip_diacritics ลบทัณฑฆาต", lambda: strip_diacritics("ศาสตราจารย์") == "ศาสตราจารย"),
        ("strip_diacritics ไม่แตะสระ", lambda: strip_diacritics("อำ") != strip_diacritics("อา")),
        ("strip_diacritics ปล่อยค่าที่ไม่ใช่ str",
         lambda: strip_diacritics(3) == 3 and strip_diacritics(None) is None),
        ("CER เหมือนกัน = 0", lambda: char_error_rate("abcd", "abcd") == 0.0),
        ("CER ผิด 1 ตัว = 0.25", lambda: abs(char_error_rate("abcd", "abcf") - 0.25) < 1e-9),
        ("clean none = ภาพเดิมทุกพิกเซล",
         lambda: bool((clean_image(img, "none") == img).all())),
        ("clean light = ภาพเทา (2 มิติ)", lambda: clean_image(img, "light").ndim == 2),
        ("clean heavy = ขาวดำล้วน",
         lambda: set(map(int, set(clean_image(img, "heavy").ravel().tolist()))) <= {0, 255}),
        ("noise L4 ย่อเหลือ 50% ของต้นฉบับ",
         lambda: apply_level(img, "L4_rescan")[0].shape[1] == round(img.shape[1] * 0.50)),
        ("noise ให้ผลเดิมทุกครั้ง (ไม่มีการสุ่ม)",
         lambda: bool((apply_level(img, "L3_titled_copy")[0] == apply_level(img, "L3_titled_copy")[0]).all())),
    ]


def cmd_selftest() -> int:
    checks = _checks()
    passed = failed = 0
    for name, fn in checks:
        try:
            good = bool(fn())
        except Exception as exc:
            good = False
            name = f"{name}  [{type(exc).__name__}: {exc}]"
        if good:
            passed += 1
        else:
            failed += 1
            print(f"  ไม่ผ่าน: {name}")
    print(f"\nผ่าน {passed} · ไม่ผ่าน {failed}  (ทั้งหมด {len(checks)})")
    return 0 if failed == 0 else 1


# ----------------------------------------------------------------- noise
def cmd_noise(args) -> int:
    manifest = generate_noisy_document(args.input, args.output, dpi=args.dpi,
                                       levels=args.levels.split(",") if args.levels else None)
    for level, info in manifest["levels"].items():
        w, h = info["pages"][0]["size"]
        print(f"  {level:<16} {w}x{h}  ({len(info['pages'])} หน้า)  {info['params']['simulates']}")
    print(f"\nเขียนลง {args.output}  (พารามิเตอร์อยู่ใน manifest.json)")
    return 0


# ----------------------------------------------------------------- sweep
def cmd_sweep(args) -> int:
    report = run_sweep(
        args.input, args.ground_truth, args.output,
        levels=args.levels.split(",") if args.levels else None,
        methods=args.methods.split(",") if args.methods else None,
        lang=args.lang, tile=args.tile,
    )
    print("\nความถูกต้องระดับฟิลด์ (ยิ่งสูงยิ่งดี)")
    print(format_grid(report, "accuracy"))
    print("\nCER (ยิ่งต่ำยิ่งดี)")
    print(format_grid(report, "cer"))
    print(f"\nรายละเอียดทั้งหมด: {Path(args.output) / 'sweep.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="lab8a_denoise", description="Lab 8A")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="ตรวจว่าเครื่องพร้อมหรือยัง")
    sub.add_parser("selftest", help="ทดสอบฟังก์ชันวัดผลด้วยข้อมูลที่รู้คำตอบ")

    n = sub.add_parser("noise", help="สร้างภาพเสื่อมสภาพ 5 ระดับ")
    n.add_argument("-i", "--input", required=True)
    n.add_argument("-o", "--output", required=True)
    n.add_argument("--dpi", type=int, default=300)
    n.add_argument("--levels", default=None, help="คั่นด้วย comma")

    s = sub.add_parser("sweep", help="clean x OCR x วัดผล ทุกคู่")
    s.add_argument("-i", "--input", required=True, help="โฟลเดอร์ที่ได้จากคำสั่ง noise")
    s.add_argument("-g", "--ground-truth", required=True)
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--levels", default=None)
    s.add_argument("--methods", default=None)
    s.add_argument("--lang", default="th")
    s.add_argument("--tile", action="store_true")

    args = parser.parse_args()
    if args.command == "check":
        return cmd_check()
    if args.command == "selftest":
        return cmd_selftest()
    if args.command == "noise":
        return cmd_noise(args)
    return cmd_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
