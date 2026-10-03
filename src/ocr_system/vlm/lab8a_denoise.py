#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 lab8a_denoise.py
 Lab 8A — เอกสารไม่สะอาด กับคำถามที่ว่า LLM ช่วยได้แค่ไหน
 Document Cleaning · Controlled Degradation · Constrained Correction
================================================================================

 วิชา 06026240 การพัฒนาระบบอัจฉริยะ

 ใช้ต่อจาก Lab 7A — pipeline อ่านเอกสาร (Typhoon-OCR -> qwen3 -> JSON)
 และตัววัดผลทั้งหมด import มาจาก lab7a_transcript.py / lab7_metrics.py
 เพื่อให้ตัวเลขของแล็บนี้เทียบกับ Lab 7A ได้ตรง ๆ

 --------------------------------------------------------------------------
 วิธีใช้
 --------------------------------------------------------------------------
   python3 lab8a_denoise.py check        # ตรวจเครื่อง
   python3 lab8a_denoise.py selftest     # ตรวจตรรกะวัดผล (ต้องผ่าน 23 · ไม่ผ่าน 0)

   # ส่วนที่ 1 — สร้างภาพเสื่อมสภาพ 5 ระดับ
   python3 lab8a_denoise.py noise -i data/71010001.pdf -o work/noisy/

   # ส่วนที่ 2-3 — ทำความสะอาด x OCR x วัดผล
   python3 lab8a_denoise.py sweep -i work/noisy -g gt/Json_71010001_th.json -o work/out
   python3 lab8a_denoise.py sweep -i work/noisy -g gt/Json_71010001_th.json -o work/out \
       --levels L3_tilted_copy,L4_rescan --methods light,heavy

   # สรุปตาราง + กราฟจาก sweep.json ที่รันไว้แล้ว (ไม่เรียกโมเดล)
   python3 lab8a_denoise.py report -o work/out
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# อยู่ในโฟลเดอร์เดียวกัน (src/ocr_system/vlm)
# รองรับทั้งรันไฟล์ตรง ๆ และ  python -m ocr_system.vlm.lab8a_denoise
try:
    from . import lab7_metrics as M
    from . import lab7a_transcript as L7
except ImportError:
    import lab7_metrics as M  # type: ignore[no-redef]
    import lab7a_transcript as L7  # type: ignore[no-redef]


# ==============================================================================
#  ส่วนที่ 0 — ค่าคงที่ของการทดลอง
# ==============================================================================

# seed เดียวทั้งแล็บ — ทุกคนที่รันคำสั่งเดียวกันต้องได้ภาพเหมือนกันทุกพิกเซล
SEED = 8240

# ink bleed ของ augraphy: ช่วงค่าแคบมากและผูก seed ไว้ จึงให้ผลเดิมทุกครั้ง
_INK_LIGHT = {"intensity_range": (0.10, 0.20), "kernel_size": (3, 3), "severity": (0.15, 0.25)}
_INK_COPIER = {"intensity_range": (0.30, 0.45), "kernel_size": (3, 3), "severity": (0.30, 0.40)}

# ห้าระดับตามตาราง 3.3 ในเอกสารแล็บ
#   skew  : องศาที่หน้าเอียง (บวก = ทวนเข็ม)
#   scale : ย่อความละเอียดแล้วขยายกลับขนาดเดิม -> รายละเอียดหายถาวร เหมือนสแกน dpi ต่ำ
#   jpeg  : ช่วง quality ตอนบันทึกไฟล์ (L3/L4 เอกสารไม่ได้ระบุ จึงลดลงตามลำดับ)
#   marks : ลายน้ำที่ใส่   dark : ความมืดด้านที่แสงน้อยที่สุด (0 = ไม่มืด)
LEVELS: dict[str, dict] = {
    "L0_clean":       {"desc": "ต้นฉบับดิจิทัล", "skew": 0.0, "scale": 1.00,
                       "ink": None, "jpeg": None, "marks": [], "dark": 0.0},
    "L1_light":       {"desc": "สแกนเนอร์ดี กระดาษเรียบ", "skew": 0.4, "scale": 1.00,
                       "ink": _INK_LIGHT, "jpeg": [75, 90], "marks": [], "dark": 0.0},
    "L2_watermark":   {"desc": "ถ่ายด้วยมือถือแล้วติดลายน้ำ", "skew": 1.1, "scale": 0.85,
                       "ink": _INK_LIGHT, "jpeg": [75, 90], "marks": ["crest"], "dark": 0.0},
    "L3_tilted_copy": {"desc": "ถ่ายซ้ำหลายรอบ", "skew": 2.3, "scale": 0.65,
                       "ink": _INK_LIGHT, "jpeg": [60, 75], "marks": ["copy"], "dark": 0.0},
    "L4_rescan":      {"desc": "เครื่องถ่ายเอกสารสำนักงาน", "skew": 3.8, "scale": 0.50,
                       "ink": _INK_COPIER, "jpeg": [45, 60], "marks": ["crest", "copy"], "dark": 0.45},
}

# ตาราง 3.3 ในเอกสารสะกด "titled" แต่คำสั่ง sweep สะกด "tilted" — รับได้ทั้งคู่
LEVEL_ALIASES = {"L3_titled_copy": "L3_tilted_copy"}

METHODS = ["none", "light", "heavy"]


def _need_cv():
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise SystemExit("❌ ไม่พบ OpenCV/numpy  -->  pip install opencv-python numpy")
    return cv2, np


def resolve_levels(text: str | None) -> list[str]:
    """แปลง '--levels a,b' เป็นรายชื่อระดับ ตรวจชื่อผิดตั้งแต่ต้น ไม่ใช่ไปพังตอนชั่วโมงที่สอง"""
    if not text:
        return list(LEVELS)
    out = []
    for name in (t.strip() for t in text.split(",") if t.strip()):
        name = LEVEL_ALIASES.get(name, name)
        if name not in LEVELS:
            raise SystemExit(f"❌ ไม่รู้จักระดับ '{name}'  (มี: {', '.join(LEVELS)})")
        out.append(name)
    return out


def resolve_methods(text: str | None) -> list[str]:
    if not text:
        return list(METHODS)
    out = [t.strip() for t in text.split(",") if t.strip()]
    for m in out:
        if m not in METHODS:
            raise SystemExit(f"❌ ไม่รู้จักวิธี '{m}'  (มี: {', '.join(METHODS)})")
    return out


def decode_png(png: bytes):
    cv2, np = _need_cv()
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def encode_png(img) -> bytes:
    cv2, _ = _need_cv()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("เข้ารหัส PNG ไม่สำเร็จ")
    return buf.tobytes()


def rotate(img, angle: float, border):
    """หมุนรอบจุดกึ่งกลาง ขนาดภาพคงเดิม  angle บวก = ทวนเข็มนาฬิกา"""
    cv2, _ = _need_cv()
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# ==============================================================================
#  ส่วนที่ 1 — CONTROLLED DEGRADATION
# ==============================================================================
#
#  ลำดับการเติม noise เรียงตาม "เวลาที่มันเกิดจริง" (ตาราง 3.1)
#     หมึกบนกระดาษ -> ลายน้ำบนกระดาษ -> วางกระดาษเอียง -> แสงไม่เท่ากัน
#     -> ความละเอียดต่ำ -> บีบอัด JPEG ตอนบันทึก
#  ถ้าสลับลำดับ เช่น ใส่ JPEG ก่อนหมุน คลื่นสี่เหลี่ยมจะเอียงตามหน้า
#  ซึ่งไม่มีทางเกิดในโลกจริง
# ==============================================================================


def _draw_crest(layer, np, cv2) -> None:
    """ลายน้ำตราทรงกลมกลางหน้า (ตราสมมติ ไม่ใช่ตราของสถาบันจริง)"""
    h, w = layer.shape
    c = (w // 2, int(h * 0.48))
    r = int(min(h, w) * 0.30)
    t = max(2, w // 250)
    cv2.circle(layer, c, r, 255, t * 3)
    cv2.circle(layer, c, int(r * 0.80), 255, t)
    cv2.circle(layer, c, int(r * 0.35), 255, t * 2)
    for k in range(24):                                  # ซี่รอบวง
        a = 2 * np.pi * k / 24
        p1 = (int(c[0] + r * 0.40 * np.cos(a)), int(c[1] + r * 0.40 * np.sin(a)))
        p2 = (int(c[0] + r * 0.75 * np.cos(a)), int(c[1] + r * 0.75 * np.sin(a)))
        cv2.line(layer, p1, p2, 255, t)
    txt, font, scale = "CERTIFIED", cv2.FONT_HERSHEY_TRIPLEX, r / 120
    (tw, th), _ = cv2.getTextSize(txt, font, scale, t * 2)
    cv2.putText(layer, txt, (c[0] - tw // 2, c[1] + th // 2), font, scale, 255, t * 2, cv2.LINE_AA)


def _draw_copy(layer, np, cv2) -> None:
    """คำว่า COPY ตัวใหญ่แนวทแยง — แบบที่เครื่องถ่ายเอกสารบางรุ่นประทับให้"""
    h, w = layer.shape
    txt, font = "COPY", cv2.FONT_HERSHEY_DUPLEX
    t = max(3, w // 70)
    (tw, _), _ = cv2.getTextSize(txt, font, 1.0, t)
    scale = w * 0.75 / tw
    (tw, th), _ = cv2.getTextSize(txt, font, scale, t)
    tmp = np.zeros_like(layer)
    cv2.putText(tmp, txt, ((w - tw) // 2, (h + th) // 2), font, scale, 255, t, cv2.LINE_AA)
    layer[:] = np.maximum(layer, rotate(tmp, 35, 0))


# ความเข้มของลายน้ำ (สัดส่วนที่แสงถูกบัง) — ต้องจางกว่าหมึกจริง ไม่งั้นมนุษย์ก็อ่านไม่ออก
MARK_DENSITY = {"crest": 0.18, "copy": 0.28}


def degrade(img, level: str, seed: int):
    """
    เติม noise ตามระดับที่กำหนด  คืนภาพ BGR ขนาดเท่าเดิม
    ฟังก์ชันนี้ deterministic: seed เดิม + ภาพเดิม = ผลเดิมทุกพิกเซล
    """
    cv2, np = _need_cv()
    p = LEVELS[level]
    out = img.copy()
    if level == "L0_clean":
        return out

    # augraphy และ numpy ใช้ตัวสุ่ม global — ต้องตั้ง seed ก่อนเรียกทุกครั้ง
    random.seed(seed)
    np.random.seed(seed % (2**32))
    rng = np.random.default_rng(seed)

    # 1) หมึกซึม — เกิดตอนพิมพ์
    if p["ink"]:
        try:
            from augraphy import InkBleed
        except ImportError:
            raise SystemExit("❌ ไม่พบ augraphy  -->  pip install augraphy")
        out = InkBleed(**p["ink"], p=1)(out)

    h, w = out.shape[:2]

    # 2) ลายน้ำ — อยู่บนกระดาษก่อนถูกถ่าย จึงต้องเอียงไปพร้อมกับตัวหนังสือ
    for mark in p["marks"]:
        layer = np.zeros((h, w), np.uint8)             # OpenCV วาดข้อความได้เฉพาะภาพ uint8
        (_draw_crest if mark == "crest" else _draw_copy)(layer, np, cv2)
        layer = cv2.GaussianBlur(layer, (5, 5), 0).astype(np.float32) / 255
        out = (out.astype(np.float32) * (1 - MARK_DENSITY[mark] * layer)[..., None]).astype(np.uint8)

    # 3) วางกระดาษเอียง — ขอบนอกกระดาษเป็นสีขาวของฝาสแกนเนอร์
    if p["skew"]:
        out = rotate(out, p["skew"], (255, 255, 255))

    # 4) แสงไม่เท่ากัน — มืดไล่จากมุมซ้ายบนไปขวาล่าง + noise ของเซนเซอร์
    if p["dark"]:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = 0.7 * xx / w + 0.3 * yy / h
        light = 0.92 - p["dark"] * ramp
        f = out.astype(np.float32) * light[..., None]
        f += rng.normal(0, 4.0, size=f.shape).astype(np.float32)
        out = np.clip(f, 0, 255).astype(np.uint8)

    # 5) ความละเอียดต่ำ — ย่อแล้วขยายกลับ ขนาดไฟล์เท่าเดิมแต่รายละเอียดหายไปแล้ว
    if p["scale"] < 1.0:
        small = cv2.resize(out, (int(w * p["scale"]), int(h * p["scale"])), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 6) JPEG ตอนบันทึกไฟล์
    if p["jpeg"]:
        q = int(rng.integers(p["jpeg"][0], p["jpeg"][1] + 1))
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return out


def level_seed(level: str, page_idx: int) -> int:
    return SEED + list(LEVELS).index(level) * 1000 + page_idx


def cmd_noise(args) -> None:
    cv2, np = _need_cv()
    src = Path(args.input)
    outdir = Path(args.out)
    levels = resolve_levels(args.levels)

    print(f"\nเตรียมภาพต้นฉบับจาก: {src}")
    pages = [decode_png(b) for b in L7.load_pages(str(src))]

    manifest = {
        "source": str(src), "dpi": L7.DPI, "seed": SEED, "n_pages": len(pages),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "levels": {},
    }
    for lv in levels:
        d = outdir / lv
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for i, img in enumerate(pages):
            seed = level_seed(lv, i)
            path = d / f"page_{i + 1:02d}.png"
            path.write_bytes(encode_png(degrade(img, lv, seed)))
            files.append({"file": str(path.relative_to(outdir)), "seed": seed})
        manifest["levels"][lv] = {**LEVELS[lv], "pages": files}
        print(f"  ✓ {lv:<16} {LEVELS[lv]['desc']:<28} -> {d}/")

    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    firsts = [(lv, cv2.imread(str(outdir / lv / "page_01.png"))) for lv in levels]
    cv2.imwrite(str(outdir / "contact_sheet.png"), contact_sheet(firsts))
    cv2.imwrite(str(outdir / "contact_sheet_full.png"), contact_sheet(firsts, crop=(0.0, 1.0, 0.0, 1.0), width=900))
    print(f"\n  บันทึกพารามิเตอร์: {outdir / 'manifest.json'}")
    print(f"  ภาพเทียบทุกระดับ (ส่วนหัวของหน้าแรก): {outdir / 'contact_sheet.png'}")
    print(f"  ภาพเทียบทุกระดับ (เต็มหน้า):          {outdir / 'contact_sheet_full.png'}")
    print("  --> เปิดดูด้วยตาตัวเอง แล้วจดว่าระดับไหนที่ 'คนเริ่มอ่านลำบาก'\n")


def contact_sheet(items: list[tuple[str, object]], crop=(0.0, 0.42, 0.0, 0.62), width: int = 620):
    """ครอปบริเวณเดียวกันของทุกระดับมาเรียงกัน เพื่อเทียบด้วยตาได้ในภาพเดียว"""
    cv2, np = _need_cv()
    tiles = []
    for label, img in items:
        h, w = img.shape[:2]
        part = img[int(h * crop[0]):int(h * crop[1]), int(w * crop[2]):int(w * crop[3])]
        part = cv2.resize(part, (width, int(part.shape[0] * width / part.shape[1])), interpolation=cv2.INTER_AREA)
        if part.ndim == 2:
            part = cv2.cvtColor(part, cv2.COLOR_GRAY2BGR)
        bar = np.full((36, width, 3), 40, np.uint8)
        cv2.putText(bar, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, part]))
    gap = np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [gap, t]
    return np.hstack(row)


# ==============================================================================
#  ส่วนที่ 2 — ทำความสะอาดภาพด้วย OpenCV
# ==============================================================================
#
#  ลำดับเปลี่ยนไม่ได้ (หัวข้อ 5.2):
#    1. deskew            ขั้นถัดไปทั้งหมดสมมติว่าบรรทัดอยู่แนวนอน
#    2. background divide ลบเงา  แล้วดันพื้นหลังให้ขาว (ลายน้ำที่จางกว่าหมึกหายไปด้วย)
#    3. fastNlMeans       ลด noise แบบรักษาขอบ
#    4. (heavy) adaptiveThreshold + morphological opening
# ==============================================================================


def estimate_skew(gray, max_angle: float = 6.0) -> float:
    """
    หามุมเอียงด้วย projection profile
      หมุนภาพทีละมุม แล้วดูผลรวมพิกเซลหมึกของแต่ละแถว
      มุมที่ถูกต้องคือมุมที่บรรทัดตั้งตรง -> แถวที่เป็นตัวหนังสือกับช่องว่าง
      ต่างกันชัดที่สุด (ผลต่างระหว่างแถวติดกันสูงสุด)
    ทนต่อลายน้ำและตารางได้ดีกว่า minAreaRect ที่ใช้กล่องล้อมหมึกทั้งหมด
    คืนค่ามุมที่หน้าเอียงอยู่ (บวก = ทวนเข็ม) เพื่อหมุนกลับด้วย -angle
    """
    cv2, np = _need_cv()
    h, w = gray.shape
    s = 800 / w if w > 800 else 1.0
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ink = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    def score(a: float) -> float:
        rows = rotate(ink, -a, 0).sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(rows) ** 2))

    coarse = max(np.arange(-max_angle, max_angle + 1e-9, 0.25), key=score)
    fine = max(np.arange(coarse - 0.25, coarse + 0.25 + 1e-9, 0.05), key=score)
    return round(float(fine), 2)


def remove_background(gray, white_point: int = 235):
    """
    background division:
      closing ด้วย kernel ใหญ่กว่าตัวอักษรมาก -> ตัวอักษรถูกกลืน เหลือแต่ระดับแสง
      ภาพเดิม ÷ พื้นหลัง -> บริเวณที่มืดเพราะเงาถูกดึงกลับขึ้นมา
    ทำ closing บนภาพย่อ 4 เท่าเพราะ kernel ใหญ่บนภาพเต็มช้ามาก และพื้นหลังไม่มีรายละเอียดอยู่แล้ว

    white_point: หลังหารแล้ว ดันทุกค่าที่สว่างกว่านี้ให้เป็นขาว
      ลายน้ำจางกว่าหมึก จึงจางลงมากในขั้นนี้  แต่หมึกยังคงระดับสีเทาไว้ครบ
      ⚠️ ตั้งต่ำกว่านี้ (เช่น 195) ลายน้ำหายหมดก็จริง แต่ตัวอักษรเส้นบางที่เบลอจาก
         ความละเอียดต่ำ (L4) หายไปด้วย เพราะความเข้มของทั้งสองอย่างใกล้กันมาก
    """
    cv2, np = _need_cv()
    h, w = gray.shape
    small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    k = max(5, (min(h, w) // 100) | 1)                    # ~51px บนภาพเต็มขนาด 1240px
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, 5)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return np.clip(flat.astype(np.float32) * 255.0 / white_point, 0, 255).astype(np.uint8)


def clean_image(img, method: str):
    """
    none  : คืนภาพเดิม
    light : deskew -> ลบเงา/ลายน้ำ -> ลด noise   (คงระดับสีเทา)
    heavy : light + adaptiveThreshold + opening (ขาวดำล้วน)
    """
    cv2, np = _need_cv()
    if method == "none":
        return img
    if method not in METHODS:
        raise ValueError(f"ไม่รู้จักวิธี {method}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    angle = estimate_skew(gray)
    if abs(angle) >= 0.1:
        paper = int(np.percentile(gray, 95))
        gray = rotate(gray, -angle, paper)

    out = remove_background(gray)

    # h=10 ลด noise JPEG ได้โดยขอบตัวอักษรยังคม — GaussianBlur จะเบลอขอบไปด้วย
    out = cv2.fastNlMeansDenoising(out, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if method == "heavy":
        # adaptive ไม่ใช่ Otsu: ค่าตัดคำนวณทีละบริเวณ ครึ่งหน้าที่มืดจึงไม่ดำสนิท
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
        # opening บนภาพกลับสี (หมึก = ขาว) -> ลบจุดหมึกเล็ก ๆ ที่เป็น noise
        # ที่ 150 DPI เส้นตัวอักษรหนาแค่ 1-2 px  opening ตรง ๆ จะกัดตัวอักษรขาดทั้งหน้า
        # จึงใช้ "opening by reconstruction": ก้อนหมึกที่ยังเหลือส่วนใดส่วนหนึ่งหลัง opening
        # ได้คืนมาทั้งก้อน  ก้อนที่หายหมดถือเป็นจุด noise แล้วลบทิ้ง
        # ⚠️ สระและวรรณยุกต์ไทยเป็นก้อนเล็กแยกจากพยัญชนะ จึงถูกลบไปด้วยได้ — สังเกตผลตรงนี้
        ink = 255 - out
        opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        _, labels = cv2.connectedComponents(ink, connectivity=8)
        keep = np.unique(labels[opened > 0])
        out = np.where(np.isin(labels, keep[keep > 0]), 0, 255).astype(np.uint8)
    return out


# ==============================================================================
#  ส่วนที่ 3 — อ่านเอกสาร + วัดผล (ใช้ของ Lab 7A ทั้งหมด)
# ==============================================================================


def extract_one_page(pages: list[bytes], md_path: Path | None = None) -> dict:
    """
    Typhoon-OCR อ่านภาพ -> Markdown -> qwen3 จัดเป็น JSON  (pipeline_vlm ของ Lab 7A)
    transcript ในแล็บนี้มีหน้าเดียว แต่ถ้ามีหลายหน้าก็ส่งเข้าไปพร้อมกันได้
    """
    return L7.pipeline_vlm(pages, save_md=md_path)


def field_accuracy(pred: dict, gt: dict) -> dict:
    """
    สรุปผล 1 เงื่อนไขเป็นตัวเลขชุดเดียว โดยใช้ evaluate() ของ Lab 7A
      accuracy : ช่องที่ตรงเฉลยเป๊ะ (หลัง normalize) ÷ ช่องทั้งหมด
                 วิชา/ภาคที่อ่านตก ถูกนับเป็นช่องผิด ไม่ถูกตัดทิ้ง
      cer      : micro CER ของทุกช่องรวมกัน
      hall     : ช่องที่เฉลยว่าง แต่โมเดลใส่ค่ามา
    """
    stats, align = L7.evaluate(pred, gt)

    # evaluate() นับรายวิชาของภาคที่อ่านตกทั้งภาคแล้ว แต่ไม่นับ ปี/ภาค/GPA/GPS ของภาคนั้น
    # ถ้าไม่เติมตรงนี้ noise ยิ่งหนัก ตัวหารยิ่งเล็ก -> accuracy ดูดีเกินจริง
    missed = M.align_by_key(
        (gt.get("transcript_detail") or {}).get("semesters") or [],
        (pred.get("transcript_detail") or {}).get("semesters") or [],
        key_fn=lambda s: f"{s.get('year')}/{s.get('sem_num')}",
    ).missed
    stats["sem_missed"] = M.FieldStat("ปี/ภาค/GPA ของภาคที่ตก")
    for s in missed:
        for f in ("year", "sem_num", "GPA", "GPS", "pass_reason"):
            stats["sem_missed"].add(s.get(f), None, f"{f}@{s.get('year')}/{s.get('sem_num')}", track_wer=False)

    n = sum(s.n_items for s in stats.values())
    exact = sum(s.n_exact for s in stats.values())
    err = sum(s.c_err for s in stats.values())
    ref = sum(s.c_ref for s in stats.values())
    return {
        "accuracy": round(exact / n, 4) if n else 0.0,
        "cer": round(err / ref, 4) if ref else 0.0,
        "hall": sum(s.n_hallucinated for s in stats.values()),
        "n_fields": n,
        "n_exact": exact,
        "subject_f1": align["subject"]["f1"],
        "per_attribute": {k: {"acc": round(s.acc, 4), "cer": round(s.cer, 4), "hall": s.n_hallucinated}
                          for k, s in stats.items() if s.n_items},
    }


def load_sweep(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"results": {}}


def cmd_sweep(args) -> None:
    cv2, _ = _need_cv()
    noisy = Path(args.input)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    levels = resolve_levels(args.levels)
    methods = resolve_methods(args.methods)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    # ตรวจว่ามีภาพครบทุกระดับก่อนเริ่ม — ไม่ใช่ไปรู้ตอนรอบที่ 9
    pages_of: dict[str, list[Path]] = {}
    for lv in levels:
        files = sorted((noisy / lv).glob("page_*.png"))
        if not files:
            raise SystemExit(f"❌ ไม่พบภาพใน {noisy / lv}  -->  รันคำสั่ง noise ก่อน")
        pages_of[lv] = files

    L7.assert_offline()
    sweep_path = outdir / "sweep.json"
    combos = [(lv, m) for lv in levels for m in methods]
    print(f"\nจะรัน {len(combos)} เงื่อนไข: {len(levels)} ระดับ x {len(methods)} วิธี")

    for k, (lv, method) in enumerate(combos, 1):
        tag = f"{lv}__{method}"
        d = outdir / tag
        d.mkdir(parents=True, exist_ok=True)
        pred_path = d / "extracted.json"
        print(f"\n{'─' * 70}\n  [{k}/{len(combos)}] {tag}\n{'─' * 70}")

        # รันค้างไว้แล้ว -> ใช้ผลเดิม (sweep เต็มชุดใช้หลายชั่วโมง ต้องต่อจากที่หยุดได้)
        if pred_path.exists() and not args.force:
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
            seconds = pred.get("_meta", {}).get("seconds")
            print(f"  ใช้ผลเดิมจาก {pred_path}  (สั่ง --force เพื่อรันใหม่)")
            error = None
        else:
            t0 = time.time()
            pages = []
            for f in pages_of[lv]:
                clean = clean_image(cv2.imread(str(f)), method)
                cv2.imwrite(str(d / f.name), clean)
                pages.append(encode_png(clean))
            t_clean = time.time() - t0
            try:
                pred = extract_one_page(pages, d / "intermediate.md")
                error = None
            except Exception as e:                   # เงื่อนไขเดียวพัง ไม่ควรทำให้ทั้ง sweep หยุด
                print(f"  ❌ อ่านเอกสารไม่สำเร็จ: {e}")
                pred, error = {}, str(e)
            seconds = round(time.time() - t0, 1)
            pred["_meta"] = {"level": lv, "method": method, "seconds": seconds,
                             "clean_seconds": round(t_clean, 1), "error": error,
                             "models": {"ocr": L7.MODEL_OCR, "text": L7.MODEL_TEXT}}
            pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

        r = field_accuracy(pred, gt)
        r.update({"level": lv, "method": method, "seconds": seconds,
                  "error": pred.get("_meta", {}).get("error", error),
                  "updated_at": datetime.now().isoformat(timespec="seconds")})
        print(f"  accuracy={r['accuracy']:.3f}  CER={r['cer']:.3f}  hall={r['hall']}  "
              f"seconds={seconds}  (subject F1={r['subject_f1']:.3f})")

        # merge ทีละเงื่อนไข -> ถ้าไฟดับกลางทาง ผลที่ได้แล้วไม่หาย และไม่ทับผลของรอบก่อน
        sweep = load_sweep(sweep_path)
        sweep["results"][tag] = r
        sweep["gt"] = str(args.gt)
        sweep_path.write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")

    report(outdir)


# ==============================================================================
#  ส่วนที่ 4 — รายงาน
# ==============================================================================

METRICS = [  # (key, หัวตาราง, best = max/min, รูปแบบ)
    ("accuracy", "ความถูกต้องระดับฟิลด์ (ยิ่งสูงยิ่งดี)", max, "{:.3f}"),
    ("cer", "CER (ยิ่งต่ำยิ่งดี)", min, "{:.3f}"),
    ("hall", "hall — ช่องที่แต่งขึ้นมา (ยิ่งต่ำยิ่งดี)", min, "{}"),
    ("seconds", "เวลา วินาที (ยิ่งต่ำยิ่งเร็ว)", min, "{}"),
]


def report(outdir: Path) -> None:
    results = load_sweep(outdir / "sweep.json")["results"]
    if not results:
        raise SystemExit(f"❌ ยังไม่มีผลใน {outdir / 'sweep.json'}")
    levels = [lv for lv in LEVELS if any(r["level"] == lv for r in results.values())]
    methods = [m for m in METHODS if any(r["method"] == m for r in results.values())]

    for key, title, best, fmt in METRICS:
        print(f"\n  {title}")
        print(f"  {'ระดับ noise':<18}" + "".join(f"{m:>10}" for m in methods))
        for lv in levels:
            vals = {m: results.get(f"{lv}__{m}", {}).get(key) for m in methods}
            have = [v for v in vals.values() if v is not None]
            b = best(have) if len(have) > 1 else None
            cells = []
            for m in methods:
                v = vals[m]
                txt = "—" if v is None else fmt.format(v) + (" *" if v == b else "  ")
                cells.append(f"{txt:>10}")
            print(f"  {lv:<18}" + "".join(cells))
    print("  * = ดีที่สุดของแถวนั้น")

    errors = [t for t, r in results.items() if r.get("error")]
    if errors:
        print(f"\n  ⚠ เงื่อนไขที่อ่านเอกสารไม่สำเร็จ (ตัวเลขเป็น 0 เพราะไม่มีผล): {', '.join(errors)}")

    with open(outdir / "sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["level", "method", "accuracy", "cer", "hall", "seconds", "n_fields", "n_exact", "subject_f1"])
        for lv in levels:
            for m in methods:
                r = results.get(f"{lv}__{m}")
                if r:
                    w.writerow([lv, m, r["accuracy"], r["cer"], r["hall"], r["seconds"],
                                r["n_fields"], r["n_exact"], r["subject_f1"]])
    print(f"\n  ตาราง CSV: {outdir / 'sweep.csv'}")
    plot(results, levels, methods, outdir / "sweep_plot.png")


def plot(results: dict, levels: list[str], methods: list[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (ไม่มี matplotlib จึงข้ามกราฟ  -->  pip install matplotlib)")
        return
    colors = {"none": "#7a7a7a", "light": "#1f77b4", "heavy": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, label) in zip(axes, [("accuracy", "field accuracy"), ("cer", "CER"),
                                       ("hall", "hallucinated fields")]):
        for m in methods:
            xs = [i for i, lv in enumerate(levels) if f"{lv}__{m}" in results]
            ys = [results[f"{levels[i]}__{m}"][key] for i in xs]
            ax.plot(xs, ys, marker="o", label=m, color=colors[m])
        ax.set_xticks(range(len(levels)), [lv.split("_")[0] for lv in levels])
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].legend(title="cleaning")
    fig.suptitle("Lab 8A — noise level vs. extraction quality")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  กราฟ: {path}")


# ==============================================================================
#  ส่วนที่ 5 — CHECK / SELFTEST
# ==============================================================================


def cmd_check(_args) -> bool:
    ok = True
    print("\n" + "=" * 70 + "\n  Lab 8A — ตรวจความพร้อม\n" + "=" * 70)
    for mod, pip in [("cv2", "opencv-python"), ("numpy", "numpy"), ("fitz", "pymupdf"),
                     ("PIL", "pillow"), ("requests", "requests"), ("augraphy", "augraphy")]:
        try:
            m = __import__(mod)
            print(f"  ✓ python: {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  ✗ python: {mod}   --> pip install {pip}")
            ok = False
    try:
        __import__("matplotlib")
        print("  ✓ python: matplotlib (วาดกราฟ — ไม่บังคับ)")
    except ImportError:
        print("  ○ python: matplotlib ไม่มี — ไม่วาดกราฟ แต่ยังรันได้  (pip install matplotlib)")

    try:
        import requests
        L7.assert_offline()
        tags = [m["name"] for m in requests.get(f"{L7.OLLAMA_HOST}/api/tags", timeout=5).json().get("models", [])]
        print(f"  ✓ Ollama ทำงานที่ {L7.OLLAMA_HOST}")
        for tag, role in [(L7.MODEL_OCR, "อ่านภาพ"), (L7.MODEL_TEXT, "จัด JSON")]:
            # ollama เติม ":latest" ให้เอง จึงเทียบแบบเดียวกับ Lab 7A
            hit = any(t == tag or t.startswith(tag + ":") or t.split(":")[0] == tag for t in tags)
            print(f"  {'✓' if hit else '✗'} [{role}] {tag}" + ("" if hit else f"   --> ollama pull {tag}"))
            ok &= hit
    except SystemExit:
        raise
    except Exception as e:
        print(f"  ✗ ต่อ Ollama ไม่ได้: {e}   --> ollama serve")
        ok = False

    print("=" * 70)
    print("  พร้อมทำแล็บ ✓" if ok else "  ยังไม่พร้อม — แก้ตามรายการ ✗ ด้านบน")
    print("=" * 70 + "\n")
    return ok


def _toy_gt() -> dict:
    return {
        "header_detail": {"student_id": "71010001", "prename": "นาย", "name": "ทดสอบระบบ",
                          "uni_name": "สถาบันสมมติ", "degree": "วิศวกรรมศาสตรบัณฑิต", "honor": 0,
                          "major": None, "grad_reason": None},
        "transcript_detail": {
            "semesters": [{
                "year": 2561, "sem_num": 1, "GPA": "3.50", "GPS": "3.50", "pass_reason": None,
                "subject": [
                    {"subject_id": "01006001", "subject_name": "calculus1", "type": None, "credit": 3, "grade_earn": "b+"},
                    {"subject_id": "01006002", "subject_name": "physics1", "type": None, "credit": 3, "grade_earn": "a"},
                ],
            }],
            "total_credits_earned": 6, "cumulative_gpa": "3.50",
        },
        "footer_detail": {"updated_at": None, "by": {"by_signature": None}},
    }


def _toy_page(np, cv2):
    img = np.full((420, 640, 3), 255, np.uint8)
    for i in range(10):
        cv2.putText(img, f"0600{i}240  Intelligent System {i}   3   B+", (30, 45 + i * 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def cmd_selftest(_args) -> bool:
    """ทดสอบตรรกะด้วยข้อมูลจิ๋วที่รู้คำตอบอยู่แล้ว — ไม่เรียกโมเดล"""
    import copy
    cv2, np = _need_cv()
    results: list[tuple[str, bool]] = []

    def t(name: str, cond) -> None:
        results.append((name, bool(cond)))

    gt = _toy_gt()
    same = field_accuracy(copy.deepcopy(gt), gt)

    # ---- วัดผล ----
    t("เหมือนเฉลยทุกช่อง -> accuracy = 1", same["accuracy"] == 1.0)
    t("เหมือนเฉลยทุกช่อง -> CER = 0", same["cer"] == 0.0)
    t("เหมือนเฉลยทุกช่อง -> hall = 0", same["hall"] == 0)

    p = copy.deepcopy(gt)
    p["transcript_detail"]["semesters"][0]["subject"][0]["grade_earn"] = "b"
    wrong = field_accuracy(p, gt)
    t("เกรดผิด 1 ช่อง -> ถูกน้อยลงพอดี 1 ช่อง", wrong["n_exact"] == same["n_exact"] - 1)
    t("เกรดผิด 1 ช่อง -> CER > 0", wrong["cer"] > 0)

    p = copy.deepcopy(gt)
    p["transcript_detail"]["semesters"][0]["pass_reason"] = "ผ่าน"
    p["header_detail"]["major"] = "วิศวกรรมคอมพิวเตอร์"
    t("เติมค่าลงช่องที่เฉลยว่าง 2 ช่อง -> hall = 2", field_accuracy(p, gt)["hall"] == 2)

    empty = field_accuracy({}, gt)
    t("โมเดลไม่ตอบอะไรเลย -> CER = 1", empty["cer"] == 1.0)
    t("โมเดลไม่ตอบอะไรเลย -> hall = 0 (ว่างไม่ใช่แต่ง)", empty["hall"] == 0)
    t("โมเดลไม่ตอบอะไรเลย -> จำนวนช่องที่วัดเท่าเดิม", empty["n_fields"] == same["n_fields"])

    p = copy.deepcopy(gt)
    p["transcript_detail"]["semesters"][0]["subject"][0]["grade_earn"] = " B+ "
    p["header_detail"]["student_id"] = "๗๑๐๑๐๐๐๑"
    t("ต่างแค่ตัวพิมพ์/ช่องว่าง/เลขไทย -> ยังนับว่าถูก", field_accuracy(p, gt)["accuracy"] == 1.0)

    p = copy.deepcopy(gt)
    p["transcript_detail"]["semesters"][0]["subject"].reverse()
    t("สลับลำดับวิชา -> ยังจับคู่ถูก accuracy = 1", field_accuracy(p, gt)["accuracy"] == 1.0)

    p = copy.deepcopy(gt)
    p["transcript_detail"]["semesters"][0]["subject"].pop()
    miss = field_accuracy(p, gt)
    t("อ่านตก 1 วิชา -> ถูกนับเป็นช่องผิด ไม่ถูกตัดทิ้ง",
      miss["n_fields"] == same["n_fields"] and miss["accuracy"] < 1.0)

    # ---- สร้าง noise ----
    page = _toy_page(np, cv2)
    t("L0_clean ไม่แตะภาพเลย", np.array_equal(degrade(page, "L0_clean", 1), page))
    a, b = degrade(page, "L3_tilted_copy", 7), degrade(page, "L3_tilted_copy", 7)
    t("seed เดิม -> ภาพเหมือนเดิมทุกพิกเซล", np.array_equal(a, b))
    t("ทุกระดับคงขนาดภาพเดิม", all(degrade(page, lv, 3).shape == page.shape for lv in LEVELS))
    t("L4_rescan มืดกว่า L0_clean", degrade(page, "L4_rescan", 3).mean() < page.mean() - 10)
    t("ชื่อ L3_titled_copy ในตารางเอกสาร ใช้ได้เหมือนกัน",
      resolve_levels("L3_titled_copy,L4_rescan") == ["L3_tilted_copy", "L4_rescan"])

    # ---- ทำความสะอาด ----
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    t("หามุมเอียง +2.3° ได้ใกล้เคียง", abs(estimate_skew(rotate(gray, 2.3, 255)) - 2.3) <= 0.3)
    t("หามุมเอียง -1.5° ได้ถูกทิศ", abs(estimate_skew(rotate(gray, -1.5, 255)) + 1.5) <= 0.3)

    h, w = gray.shape
    shade = np.linspace(1.0, 0.5, w, dtype=np.float32)[None, :]
    shadowed = (gray.astype(np.float32) * shade).astype(np.uint8)
    paper = gray > 200                                  # ตำแหน่งกระดาษเปล่าในภาพต้นฉบับ
    t("background division ทำให้พื้นกระดาษสว่างเท่ากันขึ้น",
      remove_background(shadowed)[paper].std() < shadowed[paper].std() / 3)

    t("none คืนภาพเดิม", clean_image(page, "none") is page)
    t("heavy ได้ภาพขาวดำล้วน", set(np.unique(clean_image(page, "heavy")).tolist()) <= {0, 255})
    t("light ยังคงระดับสีเทา", len(np.unique(clean_image(page, "light"))) > 2)

    n_pass = sum(ok for _, ok in results)
    print()
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  ผ่าน {n_pass} · ไม่ผ่าน {len(results) - n_pass}\n")
    return n_pass == len(results)


# ==============================================================================
#  MAIN
# ==============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Lab 8A — Document cleaning & controlled degradation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="ตรวจความพร้อมของเครื่อง")
    sub.add_parser("selftest", help="ทดสอบตรรกะวัดผลด้วยข้อมูลที่รู้คำตอบ")

    p = sub.add_parser("noise", help="สร้างภาพเสื่อมสภาพแบบควบคุมได้")
    p.add_argument("-i", "--input", required=True, help="transcript ต้นฉบับ (.pdf/.png)")
    p.add_argument("-o", "--out", default="work/noisy")
    p.add_argument("--levels", help="เช่น L1_light,L4_rescan (ค่าเริ่มต้น: ทุกระดับ)")

    p = sub.add_parser("sweep", help="ทำความสะอาด x OCR x วัดผล ทุกเงื่อนไข")
    p.add_argument("-i", "--input", default="work/noisy", help="โฟลเดอร์ผลของคำสั่ง noise")
    p.add_argument("-g", "--gt", required=True, help="ไฟล์เฉลย .json")
    p.add_argument("-o", "--out", default="work/out")
    p.add_argument("--levels")
    p.add_argument("--methods", help="none,light,heavy")
    p.add_argument("--force", action="store_true", help="รันใหม่แม้มี extracted.json อยู่แล้ว")

    p = sub.add_parser("report", help="สรุปตาราง + กราฟจาก sweep.json")
    p.add_argument("-o", "--out", default="work/out")

    args = ap.parse_args()
    if args.cmd == "check":
        sys.exit(0 if cmd_check(args) else 1)
    if args.cmd == "selftest":
        sys.exit(0 if cmd_selftest(args) else 1)
    if args.cmd == "noise":
        cmd_noise(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "report":
        report(Path(args.out))


if __name__ == "__main__":
    main()
