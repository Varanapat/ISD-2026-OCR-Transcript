#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab8a_denoise.py — Lab 8A : ทำเอกสารให้สะอาด และวัดว่า LLM "แก้" หรือ "ทำพัง"
วิชา 06026240 การพัฒนาระบบอัจฉริยะ · ภาควิชาเทคโนโลยีสารสนเทศ สจล.

ต่อยอดจาก Lab 7A ซึ่งสกัด transcript ได้แล้วบนเอกสารสะอาด
Lab 8A ตอบคำถามว่า "ถ้าเอกสารไม่สะอาดล่ะ จะเกิดอะไรขึ้น และแก้ได้แค่ไหน"

โครงของแล็บ
  1) สร้าง noise แบบควบคุมได้ 4 ระดับ L1-L4 (ground truth ไม่เปลี่ยน)
  2) รัน pipeline ของ Lab 7A บนทุกระดับ  → หา "จุดพัง"
  3) ทำความสะอาดด้วย OpenCV 3 ระดับ     → วัดว่ากู้คืนได้เท่าไร
  4) ให้ LLM แก้ข้อความหลัง OCR 2 แบบ    → วัดด้วย triage (แก้ถูก / ทำพัง)
  5) สรุปเป็นตารางเดียวเพื่อการตัดสินใจ

คำสั่งหลัก
  (รันจากรากโปรเจกต์ ocr_system/ โดยมี PYTHONPATH=src หรือ pip install -e .)
  python -m ocr_system.vlm.lab8a_denoise check
  python -m ocr_system.vlm.lab8a_denoise selftest
  python -m ocr_system.vlm.lab8a_denoise noise   --doc 71010001
  python -m ocr_system.vlm.lab8a_denoise denoise --doc 71010001 -m none [--level L3_tilted_copy]
  python -m ocr_system.vlm.lab8a_denoise sweep   --doc 71010001
  python -m ocr_system.vlm.lab8a_denoise fix     --doc 71010001
  python -m ocr_system.vlm.lab8a_denoise report  --doc 71010001      (หรือ report --all)

  --doc หา PDF ใน data/input*/ และเฉลยใน data/ground_truth*/ ให้เอง
  ผลลัพธ์อยู่ที่ work/<doc_id>/{noisy,denoise_out,out,fixed,report} และ work/summary/

หลักการที่ห้ามลืม
  ข้อมูล transcript เป็นข้อมูลส่วนบุคคลตาม PDPA — ทุกขั้นตอนรันบนเครื่องตัวเอง
  ห้ามอัปโหลดขึ้น Colab / cloud / API ภายนอกโดยเด็ดขาด
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
#  ค่าคงที่ของแล็บ
# ═══════════════════════════════════════════════════════════════════════

OLLAMA_URL = os.environ.get("LAB8_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_OCR = os.environ.get("LAB8_MODEL_OCR", "scb10x/typhoon-ocr1.5-3b")
MODEL_TEXT = os.environ.get("LAB8_MODEL_TEXT", "qwen3:4b")

# ระดับ noise — ชื่อสั้นเพื่อใช้เป็นชื่อโฟลเดอร์
NOISE_LEVELS = ["L0_clean", "L1_light", "L2_watermark", "L3_tilted_copy", "L4_rescan"]

# วิธีทำความสะอาด
CLEAN_METHODS = ["none", "light", "heavy"]

# วิธีให้ LLM แก้ข้อความ
FIX_METHODS = ["none", "free", "constrained"]

# เกรดที่ระบบทะเบียนยอมรับ — นี่คือ "closed vocabulary" ที่ใช้บังคับ LLM
VALID_GRADES = {
    "A", "B+", "B", "C+", "C", "D+", "D", "F",
    "W", "S", "U", "I", "P", "T", "V", "AU",
}

# ฟิลด์ที่ห้าม LLM เดาเด็ดขาด (แก้ผิดแล้วกู้ไม่ได้ และกระทบสิทธิ์นักศึกษา)
#   ชื่อต้องตรงกับ key ใน TRANSCRIPT_SCHEMA ของ Lab 7A (เทียบชื่อฟิลด์ตัวท้ายแบบตรงตัว)
CRITICAL_FIELDS = ("grade_earn", "credit", "subject_id", "student_id")

DPI = 300

# ผลลัพธ์สำหรับส่งเข้า discord
def make_compare_png(noisy_dir: Path, out_path: Path) -> None:
    """เรียงภาพหน้า 1 ของทั้ง 5 noise level เป็นภาพเดียว พร้อมป้ายกำกับ"""
    import cv2
    import numpy as np

    target_w = 700
    thumbs = []
    for level in NOISE_LEVELS:
        p = noisy_dir / level / "page_01.png"
        if not p.exists():
            print(f"  ! ข้าม {level} (ไม่พบ {p})")
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        scale = target_w / w
        thumb = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        band_h = 32
        band = np.full((band_h, target_w, 3), 255, dtype="uint8")
        cv2.putText(band, level, (6, band_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([band, thumb]))

    if not thumbs:
        print("  ! ไม่พบภาพใด ๆ สร้าง compare.png ไม่ได้")
        return

    max_h = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for t in thumbs]
    cv2.imwrite(str(out_path), np.hstack(thumbs))
    print(f"  บันทึก {out_path}")


def make_curve_png(rows: list[dict], out_path: Path) -> None:
    """กราฟเส้น accuracy เทียบระดับ noise แยกเส้นตาม clean method"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ! ต้องติดตั้ง matplotlib ก่อน: "
              "pip install matplotlib --break-system-packages")
        return

    levels = [lv for lv in NOISE_LEVELS if any(r["level"] == lv for r in rows)]
    if not levels:
        print("  ! ไม่มีข้อมูลพอสร้าง curve.png")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in CLEAN_METHODS:
        ys = [next((x["accuracy"] for x in rows
                    if x["level"] == lv and x["clean"] == method), None)
              for lv in levels]
        ax.plot(levels, ys, marker="o", label=method)

    ax.set_xlabel("noise level")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy across Noise Levels by Cleaning Method")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  บันทึก {out_path}")



# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 0 — ตรวจสภาพแวดล้อม
# ═══════════════════════════════════════════════════════════════════════

def check_environment() -> bool:
    """
    ตรวจว่าเครื่องพร้อมทำแล็บหรือยัง

    หลักการสำคัญ: dependency ที่ "พังเงียบ" ต้องเป็น required เสมอ
    - augraphy ขาด  -> สร้างชุด noise ไม่ได้เลย (พังดัง ไม่อันตราย)
    - opencv ขาด    -> ทำความสะอาดไม่ได้ (พังดัง)
    - Ollama ไม่ทำงาน -> ส่วนที่ 2-4 ทำไม่ได้ (พังดัง)
    ทั้งสามตัวพังแบบเห็นชัด จึงตรวจตรงนี้ทีเดียวจบ
    """
    print("=" * 68)
    print("  ตรวจสภาพแวดล้อม Lab 8A")
    print("=" * 68)
    ok = True

    # ── ไลบรารี Python ที่จำเป็น ────────────────────────────────────
    required = [
        ("cv2", "opencv-python", "ทำความสะอาดภาพ"),
        ("numpy", "numpy", "คำนวณเมทริกซ์ภาพ"),
        ("fitz", "pymupdf", "แปลง PDF เป็นภาพ"),
        ("augraphy", "augraphy", "สร้าง noise แบบควบคุมได้"),
        ("requests", "requests", "เรียก Ollama"),
    ]
    for mod, pipname, why in required:
        try:
            __import__(mod)
            print(f"  [ ok ] {pipname:<16} — {why}")
        except ImportError:
            print(f"  [FAIL] {pipname:<16} — {why}")
            print(f"         แก้ด้วย:  pip install {pipname}")
            ok = False

    # ── Ollama ──────────────────────────────────────────────────────
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        print(f"  [ ok ] Ollama ทำงานอยู่ที่ {OLLAMA_URL}")
        for want in (MODEL_OCR, MODEL_TEXT):
            hit = any(n == want or n.startswith(want.split(":")[0]) for n in names)
            if hit:
                print(f"  [ ok ] พบโมเดล {want}")
            else:
                print(f"  [FAIL] ไม่พบโมเดล {want}")
                print(f"         แก้ด้วย:  ollama pull {want}")
                ok = False
    except Exception as e:
        print(f"  [FAIL] ต่อ Ollama ไม่ได้ ({type(e).__name__})")
        print(f"         เปิด service ด้วย:  ollama serve")
        ok = False

    print("=" * 68)
    print("  พร้อมทำแล็บ" if ok else "  ยังไม่พร้อม — แก้ตามข้อความ [FAIL] ข้างบนก่อน")
    print("=" * 68)
    return ok


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 1 — สร้าง noise แบบควบคุมได้
# ═══════════════════════════════════════════════════════════════════════
#
#  ทำไมต้อง "สร้าง" noise เอง แทนที่จะไปหาเอกสารสแกนจริงมาใช้
#
#    เพราะเราต้องรู้เฉลย 100%
#    ถ้าไปเก็บเอกสารสแกนจริงมา เราจะไม่มีทางรู้ว่า "ค่าจริง" คืออะไร
#    นอกจากนั่งพิมพ์เองทุกช่อง ซึ่งช้าและผิดพลาดได้
#    แต่ถ้าเราเอาเอกสารสะอาดที่มีเฉลยอยู่แล้ว (จาก Lab 7A) มาเติม noise
#    เฉลยเดิมยังใช้ได้ทุกระดับ ทำให้เปรียบเทียบข้ามระดับได้อย่างยุติธรรม
#
#    นี่คือหลัก "controlled degradation" ที่ใช้ในงานวิจัย document analysis จริง
# ═══════════════════════════════════════════════════════════════════════

def build_noise_pipeline(level: str, seed: int = 42):
    """
    คืน Augraphy pipeline สำหรับระดับ noise ที่ระบุ

    เอกสารสมัยนี้ส่วนใหญ่เป็นไฟล์ดิจิทัล (export/scan/แชร์ต่อ) ไม่ใช่กระดาษ
    ที่ผ่านเครื่องพิมพ์/เครื่องถ่ายเอกสารซ้ำหลายรอบ จึงจำลองแค่สิ่งที่เกิดจริง
    ตอนไฟล์ถูกบีบอัดซ้ำ (SubtleNoise/Jpeg) ส่วนลายน้ำ "COPY" และมุมเอียง
    ถูกใส่แยกต่างหากใน make_noise_dataset() เพราะ Augraphy ไม่มี op นี้
    """
    import numpy as np
    from augraphy import AugraphyPipeline, Jpeg, SubtleNoise

    np.random.seed(seed)

    if level == "L0_clean":
        # ระดับ 0 ไม่ใส่อะไรเลย ใช้เป็นเส้นฐานเปรียบเทียบ
        return None

    if level == "L1_light":
        # แค่บันทึกไฟล์ซ้ำเป็น JPEG คุณภาพสูง — แทบไม่ต่างจากต้นฉบับ
        return AugraphyPipeline(
            ink_phase=[], paper_phase=[],
            post_phase=[SubtleNoise(subtle_range=4, p=1.0),
                        Jpeg(quality_range=(85, 95), p=1.0)],
        )

    if level == "L2_watermark":
        # ไฟล์ที่แชร์ต่อพร้อมลายน้ำ "COPY" จาง ๆ + เอียงเล็กน้อยตอนสแกน
        return AugraphyPipeline(
            ink_phase=[], paper_phase=[],
            post_phase=[SubtleNoise(subtle_range=5, p=1.0),
                        Jpeg(quality_range=(75, 90), p=1.0)],
        )

    if level == "L3_tilted_copy":
        # สำเนาที่ถูกบีบอัดซ้ำหลายต่อ + ลายน้ำเข้มขึ้น + เอียงมากขึ้น
        return AugraphyPipeline(
            ink_phase=[], paper_phase=[],
            post_phase=[SubtleNoise(subtle_range=6, p=1.0),
                        Jpeg(quality_range=(60, 80), p=1.0)],
        )

    if level == "L4_rescan":
        # สแกนซ้ำจากสำเนา (ความละเอียดลด) + ลายน้ำเข้มสุด + เอียงสุด
        return AugraphyPipeline(
            ink_phase=[], paper_phase=[],
            post_phase=[SubtleNoise(subtle_range=8, p=1.0),
                        Jpeg(quality_range=(45, 65), p=1.0)],
        )

    raise ValueError(f"ไม่รู้จักระดับ noise: {level}")


def add_watermark_circle(img, opacity: float):
    """
    วาดลายน้ำเป็นตราปั๊มวงกลมสีส้มขนาดใหญ่กลางหน้า (วงแหวนซ้อน + เส้นรัศมีโดยรอบ)

    จำลองสถานการณ์ที่พบได้จริงในปัจจุบัน (เอกสาร export/แชร์ต่อมีตราปั๊มทับ)
    ใช้กับระดับ L2_watermark และเป็นส่วนหนึ่งของ L4_rescan
    """
    import cv2
    import numpy as np

    if opacity <= 0:
        return img

    h, w = img.shape[:2]
    layer = np.zeros((h, w, 3), dtype="uint8")
    color = (0, 140, 255)  # BGR ส้ม
    cx, cy = w // 2, h // 2
    radius = int(min(h, w) * 0.38)

    # วงแหวนซ้อนกันแบบตราปั๊ม
    for r in (radius, int(radius * 0.86)):
        cv2.circle(layer, (cx, cy), r, color, thickness=6, lineType=cv2.LINE_AA)

    # เส้นรัศมีรอบวง ให้ดูเป็นลวดลาย/ตราปั๊ม
    for deg in range(0, 360, 12):
        rad = np.radians(deg)
        p1 = (int(cx + radius * 0.6 * np.cos(rad)), int(cy + radius * 0.6 * np.sin(rad)))
        p2 = (int(cx + radius * 0.86 * np.cos(rad)), int(cy + radius * 0.86 * np.sin(rad)))
        cv2.line(layer, p1, p2, color, thickness=4, lineType=cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((cx, cy), 20, 1.0)
    layer = cv2.warpAffine(layer, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))

    mask = (layer.sum(axis=2) > 0)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    blended = cv2.addWeighted(img, 1 - opacity, layer, opacity, 0)
    return np.where(mask3, blended, img).astype("uint8")


def add_watermark_tiled_copy(img, opacity: float):
    """
    วาดคำว่า COPY สีเทาตัวหนา เอียง 30 องศา กระจายเต็มหน้า

    จำลองไฟล์ที่ถูกประทับ "COPY" ซ้ำหลายจุดตอน export/แชร์
    ใช้กับระดับ L3_tilted_copy และเป็นส่วนหนึ่งของ L4_rescan
    """
    import cv2
    import numpy as np

    if opacity <= 0:
        return img

    h, w = img.shape[:2]
    layer = np.zeros((h, w, 3), dtype="uint8")
    color = (150, 150, 150)  # BGR เทา
    scale = w / 700
    thickness = max(4, int(scale * 6))
    step_x, step_y = int(w / 2.0), int(h / 4.0)
    for y in range(step_y, h + step_y, step_y):
        for x in range(-step_x, w + step_x, step_x):
            cv2.putText(layer, "COPY", (x, y), cv2.FONT_HERSHEY_DUPLEX,
                       scale, color, thickness, cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    layer = cv2.warpAffine(layer, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))

    mask = (layer.sum(axis=2) > 0)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    blended = cv2.addWeighted(img, 1 - opacity, layer, opacity, 0)
    return np.where(mask3, blended, img).astype("uint8")


def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = DPI) -> list[Path]:
    """แปลง PDF เป็นภาพ PNG หน้าละไฟล์ ที่ความละเอียด dpi"""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        p = out_dir / f"page_{i:02d}.png"
        pix.save(p)
        paths.append(p)
    doc.close()
    return paths


def apply_rotation(img, degrees: float):
    """
    หมุนภาพโดยเติมขอบด้วยสีขาว

    แยกออกมาจาก Augraphy เพราะเราต้องคุมมุมเองให้ได้แน่นอน
    เวลาทดสอบ deskew จะได้รู้ว่า "มุมจริง" คือเท่าไร
    """
    import cv2
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


# มุมเอียงประจำแต่ละระดับ (องศา) — คงที่เพื่อให้ทำซ้ำได้
LEVEL_SKEW = {"L0_clean": 0.0, "L1_light": 0.3, "L2_watermark": 1.0,
              "L3_tilted_copy": 2.5, "L4_rescan": 4.0}

# อัตราลดความละเอียดของแต่ละระดับ (จำลองการสแกนที่ dpi ต่ำ)
LEVEL_SCALE = {"L0_clean": 1.0, "L1_light": 1.0, "L2_watermark": 1.0,
               "L3_tilted_copy": 0.95, "L4_rescan": 0.85}

# ความเข้มของลายน้ำตราปั๊มประจำแต่ละระดับ (0 = ไม่มี)
LEVEL_WATERMARK = {"L0_clean": 0.0, "L1_light": 0.0, "L2_watermark": 0.35,
                   "L3_tilted_copy": 0.45, "L4_rescan": 0.55}


def make_noise_dataset(src: Path, out_dir: Path, seed: int = 42) -> dict:
    """
    สร้างชุดข้อมูล noise ระดับ L1-L4 จากเอกสารต้นฉบับหนึ่งฉบับ

    สร้าง L0_clean (ต้นฉบับ ไม่มี noise) ไว้ด้วย เพื่อให้ path เหมือน level อื่น
    ผลลัพธ์:  out_dir/L0_clean/page_01.png, out_dir/L1_light/page_01.png, out_dir/L2_watermark/page_01.png, ...
    เฉลย (ground truth) ใช้ไฟล์เดิมของ Lab 7A ได้ทุกระดับ
    """
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    # ขั้นที่ 1 — ได้ภาพสะอาดมาก่อน
    tmp = out_dir / "_source"
    if src.suffix.lower() == ".pdf":
        pages = pdf_to_images(src, tmp)
    elif src.is_dir():
        pages = sorted([p for p in src.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    else:
        pages = [src]
    print(f"  ต้นฉบับ {len(pages)} หน้า")

    manifest = {"source": str(src), "seed": seed, "dpi": DPI, "levels": {}}

    l0_dir = out_dir / "L0_clean"
    l0_dir.mkdir(parents=True, exist_ok=True)
    made0 = []
    for idx, p in enumerate(pages):
        img = cv2.imread(str(p))
        if img is None:
            print(f"    ! อ่านไม่ได้: {p}")
            continue
        dst = l0_dir / f"page_{idx + 1:02d}.png"
        cv2.imwrite(str(dst), img)
        made0.append(dst.name)
    manifest["levels"]["L0_clean"] = {
        "pages": made0, "skew_deg": 0.0, "scale": 1.0, "watermark_opacity": 0.0,
    }
    print(f"  [L0_clean      ] {len(made0)} หน้า  (ต้นฉบับ ไม่มี noise)")

    # ขั้นที่ 2 — วนสร้างทีละระดับ (ข้าม L0_clean เพราะคือต้นฉบับเดิม ไม่ต้องสร้างซ้ำ)
    for level in NOISE_LEVELS[1:]:
        lv_dir = out_dir / level
        lv_dir.mkdir(parents=True, exist_ok=True)
        pipe = build_noise_pipeline(level, seed=seed)
        made = []

        for idx, p in enumerate(pages):
            img = cv2.imread(str(p))
            if img is None:
                print(f"    ! อ่านไม่ได้: {p}")
                continue

            # ลำดับสำคัญ: ย่อ -> หมุน -> ลายน้ำ -> จบด้วยการบีบอัด JPEG
            # เพราะในโลกจริง การ export ซ้ำ/เอกสารเอียงเกิดก่อน แล้วลายน้ำ
            # ถูกประทับตอน share/print ก่อนที่ไฟล์จะถูกบีบอัดครั้งสุดท้าย
            s = LEVEL_SCALE[level]
            if s < 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            deg = LEVEL_SKEW[level]
            if abs(deg) > 1e-6:
                img = apply_rotation(img, deg)
            wm = LEVEL_WATERMARK[level]
            if level == "L2_watermark":
                img = add_watermark_circle(img, wm)
            elif level == "L3_tilted_copy":
                img = add_watermark_tiled_copy(img, wm)
            elif level == "L4_rescan":
                # L4 = ตราปั๊มวงกลมของ L2 + คำว่า COPY เอียงของ L3 ซ้อนกัน
                # แล้วเข้มขึ้นทั้งแผ่นเล็กน้อย จำลองการสแกนซ้ำจากสำเนา
                img = add_watermark_circle(img, LEVEL_WATERMARK["L2_watermark"])
                img = add_watermark_tiled_copy(img, LEVEL_WATERMARK["L3_tilted_copy"])
                img = cv2.convertScaleAbs(img, alpha=0.9, beta=0)
            if pipe is not None:
                img = pipe(img)
                if isinstance(img, dict):          # Augraphy บางรุ่นคืน dict
                    img = img["output"]
                img = np.asarray(img)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            dst = lv_dir / f"page_{idx + 1:02d}.png"
            cv2.imwrite(str(dst), img)
            made.append(dst.name)

        manifest["levels"][level] = {
            "pages": made,
            "skew_deg": LEVEL_SKEW[level],
            "scale": LEVEL_SCALE[level],
            "watermark_opacity": LEVEL_WATERMARK[level],
        }
        print(f"  [{level:<14}] {len(made)} หน้า  "
              f"(เอียง {LEVEL_SKEW[level]}° · ย่อ {LEVEL_SCALE[level]:.0%} · "
              f"ลายน้ำ {LEVEL_WATERMARK[level]:.0%})")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 2 — ทำความสะอาดภาพด้วย OpenCV
# ═══════════════════════════════════════════════════════════════════════
#
#  คำเตือนที่นักศึกษาจะพิสูจน์เองในส่วนที่ 5
#
#    "สะอาดขึ้นในสายตาคน" ไม่ได้แปลว่า "OCR อ่านได้ดีขึ้น"
#    Typhoon-OCR เป็น VLM ที่ฝึกมาบนเอกสารสแกนจริงจำนวนมาก
#    มันจึง "คุ้นเคย" กับ noise ระดับหนึ่งอยู่แล้ว
#    การ binarize แรง ๆ จะตัดข้อมูล grayscale ที่โมเดลใช้แยกสระ/วรรณยุกต์ไทยทิ้ง
#    ผลคือภาพดูคมขึ้นสำหรับคน แต่โมเดลอ่านแย่ลง
# ═══════════════════════════════════════════════════════════════════════

def estimate_skew(gray) -> float:
    """
    ประมาณมุมเอียงจากเส้นแนวตั้งที่ยาวที่สุดในหน้าเอกสารด้วย Hough transform

    กรองเฉพาะเส้นที่เอียงจากแนวตั้งไม่เกิน 10 องศา แล้วคืนมุมสำหรับ
    apply_rotation() โดยตรง ถ้าไม่พบเส้นแนวตั้งที่ยาวพอจะคืน 0.0
    """
    import cv2
    import numpy as np

    if gray is None or gray.size == 0:
        return 0.0

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    height, _ = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # ใช้ Probabilistic Hough เพื่อให้ได้ปลายเส้นและวัดความยาวจริงได้
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800,       # ความละเอียดมุม 0.1 องศา
        threshold=max(50, int(height * 0.05)),
        minLineLength=max(50, int(height * 0.20)),
        maxLineGap=max(10, int(height * 0.03)),
    )
    if lines is None:
        return 0.0

    longest = None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))

        # มุมเบี่ยงจากแนวตั้ง: 0 คือเส้นตั้งตรง และทำให้ไม่ขึ้นกับลำดับปลายเส้น
        deviation = float(np.degrees(np.arctan2(dx, dy)))
        while deviation <= -90.0:
            deviation += 180.0
        while deviation > 90.0:
            deviation -= 180.0

        if abs(deviation) <= 10.0 and (longest is None or length > longest[0]):
            longest = (length, deviation)

    if longest is None:
        return 0.0

    # deviation คือมุมที่ภาพเอียงอยู่ จึงหมุนด้วยเครื่องหมายตรงข้ามเพื่อแก้กลับ
    return -longest[1]


# def remove_watermark_tint(img):
#     """
#     ลบลายน้ำก่อนแปลงเป็น grayscale — ต้องทำก่อน estimate_skew() เสมอ
#     ไม่งั้นเส้นลายน้ำเอียง ๆ จะไปปนกับตัวอักษรจริงตอนประมาณมุมเอียง

#     ลายน้ำมีสองแบบ ต้องกรองคนละเงื่อนไข
#       วงกลม/ตราปั๊ม (L2, L4) — สีส้ม จึงกรองด้วย Hue ช่วงเหลือง/ส้ม
#       คำว่า COPY (L3, L4)   — สีเทา (satuation ต่ำ) ต่างจากตัวอักษรจริงที่เป็น
#                               สีดำเข้ม (value ต่ำมาก) จึงกรองด้วยช่วง value กลาง ๆ แทน
#     แล้วดึงบริเวณที่ตรงเงื่อนไขกลับเป็นสีขาวล้วน โดยไม่แตะตัวอักษรจริง
#     """
#     import cv2
#     import numpy as np
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     orange_stamp = (h >= 15) & (h <= 40) & (v >= 120)
#     gray_copy = (s <= 40) & (v >= 90) & (v <= 235)
#     watermark = orange_stamp | gray_copy
#     s = np.where(watermark, 0, s).astype("uint8")
#     v = np.where(watermark, 255, v).astype("uint8")
#     return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def remove_watermark_tint(img):
    """
    ลบลายน้ำก่อนแปลงเป็น grayscale — ต้องทำก่อน estimate_skew() เสมอ
    ไม่งั้นเส้นลายน้ำเอียง ๆ จะไปปนกับตัวอักษรจริงตอนประมาณมุมเอียง

    ลายน้ำมีสองแบบ ต้องกรองคนละเงื่อนไข
      วงกลม/ตราปั๊ม (L2, L4) — สีส้ม จึงกรองด้วย Hue ช่วงเหลือง/ส้ม
      คำว่า COPY (L3, L4)   — สีเทา (satuation ต่ำ) ต่างจากตัวอักษรจริงที่เป็น
                              สีดำเข้ม (value ต่ำมาก) จึงกรองด้วยช่วง value กลาง ๆ แทน
    แล้วดึงบริเวณที่ตรงเงื่อนไขกลับเป็นสีขาวล้วน โดยไม่แตะตัวอักษรจริง

    ปัญหาที่พบ: ขอบตัวอักษรไทยจาง ๆ จาก anti-aliasing/JPEG ก็ตกอยู่ในช่วงสี
    เดียวกับ gray_copy/orange_stamp พอดี ต่างกันแค่ "ความหนา" เท่านั้น —
    ลายน้ำวาดด้วย thickness >= 4px แต่ขอบตัวอักษรหนาแค่ 1-2px จึงใช้
    morphological opening กรองซ้ำอีกชั้น: จุดที่บางกว่า kernel จะถูก
    erosion กัดหายไปทั้งหมด (= ไม่ถูกนับเป็นลายน้ำ ตัวอักษรรอด) ส่วนลายน้ำที่
    หนากว่าจะรอด แล้ว dilate กลับคืนขนาดเดิมก่อนเอาไปลบ

    ข้อจำกัดที่รู้อยู่: ถ้าลายน้ำทับเนื้อหาโดยตรง (เช่นตราปั๊มวงกลมทับกลาง
    ตาราง) วิธีนี้ยังกู้คืนไม่ได้ดีพอ เพราะไม่รู้ค่าพิกเซลจริงใต้ mask
    """
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    orange_stamp_raw = (h >= 15) & (h <= 40) & (v >= 120)
    gray_copy_raw = (s <= 40) & (v >= 90) & (v <= 235)

    kernel = np.ones((3, 3), np.uint8)
    orange_stamp = cv2.morphologyEx(
        orange_stamp_raw.astype("uint8") * 255, cv2.MORPH_OPEN, kernel
    ).astype(bool)
    gray_copy = cv2.morphologyEx(
        gray_copy_raw.astype("uint8") * 255, cv2.MORPH_OPEN, kernel
    ).astype(bool)

    watermark = orange_stamp | gray_copy
    s = np.where(watermark, 0, s).astype("uint8")
    v = np.where(watermark, 255, v).astype("uint8")
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

# def remove_watermark_tint(img, opacity: float = 0.0):
#     """
#     ... (docstring เดิม) ...

#     วิธีฟอกขาว/inpaint ทำลาย/เดาข้อมูลตรงจุดที่ลายน้ำทับตัวอักษรจริง เพราะ
#     ไม่รู้ค่าพิกเซลต้นฉบับใต้ mask เลย — แต่จริง ๆ รู้สูตรผสมสีที่ใช้สร้าง
#     ลายน้ำอยู่แล้ว (blended = img*(1-opacity) + color*opacity) ถ้ารู้ opacity
#     (จาก LEVEL_WATERMARK) กับสีลายน้ำ ก็แก้สมการย้อนกลับหา img ต้นฉบับได้ตรง ๆ
#     แม่นกว่าเดา เพราะเป็นการ "คำนวณย้อนกลับ" ไม่ใช่ "เดา"
#     """
#     import cv2
#     import numpy as np
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     orange_stamp_raw = (h >= 15) & (h <= 40) & (v >= 120)
#     gray_copy_raw = (s <= 40) & (v >= 90) & (v <= 235)

#     kernel = np.ones((3, 3), np.uint8)
#     orange_stamp = cv2.morphologyEx(
#         orange_stamp_raw.astype("uint8") * 255, cv2.MORPH_OPEN, kernel
#     ).astype(bool)
#     gray_copy = cv2.morphologyEx(
#         gray_copy_raw.astype("uint8") * 255, cv2.MORPH_OPEN, kernel
#     ).astype(bool)

#     if opacity <= 0:
#         # ไม่รู้ opacity จริง (เรียกแบบเดิมไม่ส่งมา) — fallback ฟอกขาวแบบเดิม
#         watermark = orange_stamp | gray_copy
#         s = np.where(watermark, 0, s).astype("uint8")
#         v = np.where(watermark, 255, v).astype("uint8")
#         return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

#     out = img.astype("float32")
#     orange_color = np.array([0, 140, 255], dtype="float32")   # BGR ตรงกับ add_watermark_circle
#     gray_color = np.array([150, 150, 150], dtype="float32")   # BGR ตรงกับ add_watermark_tiled_copy
#     orange_mask3 = np.repeat(orange_stamp[:, :, None], 3, axis=2)
#     gray_mask3 = np.repeat((gray_copy & ~orange_stamp)[:, :, None], 3, axis=2)

#     out = np.where(orange_mask3, (out - orange_color * opacity) / (1 - opacity), out)
#     out = np.where(gray_mask3, (out - gray_color * opacity) / (1 - opacity), out)
#     return np.clip(out, 0, 255).astype("uint8")


# def clean_image(img, method: str):
#     """
#     ทำความสะอาดภาพตามระดับที่เลือก

#     none  — ไม่ทำอะไร (เส้นฐาน)
#     light — ลบลายน้ำ + แก้เอียง + ลด noise เบา ๆ  แต่ "คงระดับสีเทาไว้"
#     heavy — light + binarize แบบรักษาเส้นหมึก  ตัดเป็นขาวดำล้วน
#     """
#     import cv2
#     if method == "none":
#         return img

#     # 0) ลบลายน้ำสีก่อนแปลงเป็น grayscale — ทำตอนยังมีสีอยู่เท่านั้น
#     img = remove_watermark_tint(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

#     # 1) แก้ภาพเอียง — ทำก่อนเสมอ เพราะขั้นอื่นสมมติว่าบรรทัดอยู่แนวนอน
#     ang = estimate_skew(gray)
#     if abs(ang) > 0.1:
#         gray = apply_rotation(gray, ang)

#     # 2) ลด noise แบบรักษาขอบตัวอักษร
#     gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7,
#                                     searchWindowSize=21)

#     if method == "light":
#         return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

#     if method == "heavy":
#         # 3) binarize แบบปรับตามพื้นที่ — ลด C เพื่อไม่ให้ threshold
#         # กินขอบตัวอักษรบางๆ และใช้พื้นที่รอบข้างกว้างขึ้นเพื่อให้พื้นหลังนิ่ง
#         bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, blockSize=41, C=7)
#         # fastNlMeansDenoising ด้านบนเก็บ noise ไปแล้ว จึงไม่ทำ morphological
#         # opening ซ้ำ เพราะ kernel 2x2 สามารถทำให้เส้นอักษรขนาด 1–2 px ขาดได้
#         return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

#     raise ValueError(f"ไม่รู้จักวิธีทำความสะอาด: {method}")

def clean_image(img, method: str):
    """
    ทำความสะอาดภาพตามระดับที่เลือก

    none  — ไม่ทำอะไร (เส้นฐาน)
    light — ลบลายน้ำ + แก้เอียง + ลด noise เบา ๆ  แต่ "คงระดับสีเทาไว้"
    heavy — light + binarize แบบรักษาเส้นหมึก  ตัดเป็นขาวดำล้วน
    """
    import cv2
    if method == "none":
        return img

    # 0) ลบลายน้ำสีก่อนแปลงเป็น grayscale — ทำตอนยังมีสีอยู่เท่านั้น
    img = remove_watermark_tint(img)
    # img = remove_watermark_tint(img, watermark_opacity)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # 1) แก้ภาพเอียง — ทำก่อนเสมอ เพราะขั้นอื่นสมมติว่าบรรทัดอยู่แนวนอน
    ang = estimate_skew(gray)
    if abs(ang) > 0.1:
        gray = apply_rotation(gray, ang)

    # 2) ลด noise แบบรักษาขอบตัวอักษร
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7,
                                    searchWindowSize=21)

    if method == "light":
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if method == "heavy":
        # 3) binarize แบบปรับตามพื้นที่ — ลด C เพื่อไม่ให้ threshold
        # กินขอบตัวอักษรบางๆ และใช้พื้นที่รอบข้างกว้างขึ้นเพื่อให้พื้นหลังนิ่ง
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=41, C=7)
        # fastNlMeansDenoising ด้านบนเก็บ noise ไปแล้ว จึงไม่ทำ morphological
        # opening ซ้ำ เพราะ kernel 2x2 สามารถทำให้เส้นอักษรขนาด 1–2 px ขาดได้
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"ไม่รู้จักวิธีทำความสะอาด: {method}")


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 3 — เรียกใช้ main() ของ Lab 7A
# ═══════════════════════════════════════════════════════════════════════

_OLLAMA_VERSION: str | None = None


def ollama_version() -> str:
    """
    เวอร์ชันของ Ollama ที่กำลังรันอยู่ — บันทึกไว้คู่กับผลทุกครั้ง

    ทำไมต้องจด: โค้ด โมเดล และ PDF เดียวกันทุกไบต์ ยังได้ผลต่างกันได้
    เมื่อ Ollama อัปเดตตัวเอง (แอปบน macOS เช็คอัปเดตทุกชั่วโมง)
    เคยเจอจริง: 71010001 ได้ 0.925 ก่อนอัปเดต แต่ได้ 0.688 หลังอัปเดตเป็น 0.33.3
    """
    global _OLLAMA_VERSION
    if _OLLAMA_VERSION is None:
        try:
            import requests
            r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
            r.raise_for_status()
            _OLLAMA_VERSION = str(r.json().get("version") or "unknown")
        except Exception:
            _OLLAMA_VERSION = "unknown"
    return _OLLAMA_VERSION


def ollama_generate(model: str, prompt: str, images: list[str] | None = None,
                    fmt: str | None = None, timeout: int = 600) -> str:
    """เรียก Ollama บนเครื่องตัวเอง — ไม่มีการส่งข้อมูลออกนอกเครื่อง"""
    import base64
    import requests

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    if images:
        payload["images"] = [
            base64.b64encode(Path(p).read_bytes()).decode() for p in images
        ]
    if fmt:
        payload["format"] = fmt

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_json_loose(s: str) -> dict:
    """ดึง JSON ออกจากคำตอบของโมเดล แม้จะมี fence หรือข้อความหุ้มมาด้วย"""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    depth = 0
    for i in range(start, len(s)) if start >= 0 else []:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    break
    return {}


def _lab7_module():
    """โหลด Lab 7A แบบรองรับทั้ง package และการรันไฟล์ตรง"""
    try:
        from . import lab7a_transcript
    except ImportError:
        import lab7a_transcript
    return lab7a_transcript


# def ocr_one_page(img_path: Path, lab7=None) -> str:
#     """อ่านภาพหนึ่งหน้าเป็น Markdown โดยยังไม่โหลด text LLM"""
#     lab7 = lab7 or _lab7_module()
#     pages = lab7.load_pages(str(img_path))
#     return lab7.ocr_pages_to_markdown(pages)


# def structure_one_page(markdown: str, lab7=None) -> dict:
#     """จัด Markdown หนึ่งหน้าเป็น JSON โดยไม่เรียก vision model ซ้ำ"""
#     lab7 = lab7 or _lab7_module()
#     return lab7.structure_markdown(markdown)


# def extract_one_page(img_path: Path) -> tuple[str, dict]:
#     """อ่านภาพด้วย pipeline VLM ของ Lab 7A แล้วคืน Markdown และ JSON"""
#     lab7 = _lab7_module()
#     lab7.assert_offline()
#     t0 = time.time()
#     md = ocr_one_page(img_path, lab7)
#     rec = structure_one_page(md, lab7)
#     rec["_meta"] = {
#         "pipeline": "vlm",
#         "elapsed_sec": round(time.time() - t0, 1),
#         "models": {"ocr": lab7.MODEL_OCR, "text": lab7.MODEL_TEXT},
#         "dpi": lab7.DPI,
#     }
#     return md, rec

def extract_one_page(img_path: Path, save_md: Path | None = None) -> tuple[str, dict]:
    """อ่านภาพด้วย pipeline VLM ของ Lab 7A แล้วคืน Markdown และ JSON"""
    lab7 = _lab7_module()
    lab7.assert_offline()
    t0 = time.time()
    pages = lab7.load_pages(str(img_path))
    rec = lab7.pipeline_vlm(pages, save_md=save_md)
    md = save_md.read_text(encoding="utf-8") if save_md and save_md.exists() else ""
    rec["_meta"] = {**(rec.get("_meta") or {}), **run_meta(lab7, t0)}   # เก็บ detected_lang จาก Lab 7A ไว้ด้วย
    return md, rec


def run_meta(lab7, t0: float) -> dict:
    """ข้อมูลประกอบการรันที่ต้องจดคู่กับผลทุกครั้ง เพื่อให้ทำซ้ำ/อธิบายความต่างได้"""
    return {
        "pipeline": "vlm",
        "elapsed_sec": round(time.time() - t0, 1),
        "models": {"ocr": lab7.MODEL_OCR, "text": lab7.MODEL_TEXT},
        "dpi": lab7.DPI,
        "ollama_version": ollama_version(),
    }

# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 4 — ให้ LLM แก้ข้อความหลัง OCR
# ═══════════════════════════════════════════════════════════════════════

FREE_FIX_PROMPT = """ข้อมูลใบแสดงผลการศึกษาต่อไปนี้ได้มาจาก OCR ที่อ่านเอกสารคุณภาพต่ำ
จึงมีตัวสะกดผิดและอักขระเพี้ยนอยู่ กรุณาแก้ไขให้ถูกต้อง

ตอบกลับเป็น JSON โครงสร้างเดิมทั้งหมด ไม่ต้องอธิบาย

ข้อมูล:
"""

CONSTRAINED_FIX_PROMPT = """ข้อมูลใบแสดงผลการศึกษาต่อไปนี้ได้มาจาก OCR ที่อ่านเอกสารคุณภาพต่ำ
งานของคุณคือแก้เฉพาะ "ชื่อวิชา" และ "ชื่อบุคคล" ที่สะกดผิดจาก OCR เท่านั้น

กติกาที่ห้ามฝ่าฝืน
1. ห้ามแก้ค่าในฟิลด์ subject_id, credit, grade_earn, student_id เด็ดขาด — คัดลอกมาตามเดิมทุกตัวอักษร
2. ถ้าไม่มั่นใจว่าค่าที่ถูกคืออะไร ให้ใส่ null ห้ามเดา
3. ห้ามเพิ่มหรือลบรายวิชา จำนวนรายการต้องเท่าเดิม
4. ห้ามคำนวณหรือเติมค่าที่หายไป

ตอบกลับเป็น JSON โครงสร้างเดิมทั้งหมด ไม่ต้องอธิบาย

ข้อมูล:
"""


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def snap_grade(value: Any) -> Any:
    """
    บังคับให้เกรดอยู่ในชุดที่ระบบทะเบียนยอมรับ

    - ถ้าตรงอยู่แล้ว        -> คืนค่าเดิม
    - ถ้าใกล้เคียงตัวเดียว   -> แก้ให้ (เช่น "8+" -> "B+" เพราะ 8 กับ B สับสนกันบ่อย)
    - ถ้าใกล้เคียงหลายตัว   -> คืน None (ให้คนตรวจ) ปลอดภัยกว่าเดา
    """
    if value is None:
        return None
    v = str(value).strip().upper().replace(" ", "")
    if v in VALID_GRADES:
        return v
    # แก้ความสับสนของ OCR ที่พบบ่อยก่อน
    table = str.maketrans({"8": "B", "6": "G", "0": "D", "1": "I", "5": "S"})
    v2 = v.translate(table)
    if v2 in VALID_GRADES:
        return v2
    cands = [g for g in VALID_GRADES if similar(v, g) >= 0.75]
    return cands[0] if len(cands) == 1 else None


def snap_code(value: Any, registry: set[str] | None) -> Any:
    """
    บังคับให้รหัสวิชาเป็นเลข 8 หลัก และ (ถ้ามีทะเบียนวิชา) ต้องมีอยู่จริง

    ทะเบียนวิชาคือ "closed vocabulary" ที่ดีที่สุดที่เรามี
    ถ้ารหัสที่ OCR อ่านได้ต่างจากรหัสจริงแค่หลักเดียว เราแก้ให้ได้อย่างมั่นใจ
    แต่ถ้าต่างสองหลักขึ้นไป อาจกลายเป็นวิชาอื่นได้จริง — จึงคืน None
    """
    if value is None:
        return None
    v = re.sub(r"\D", "", str(value))
    if registry is None:
        return v if len(v) == 8 else None
    if v in registry:
        return v
    near = [c for c in registry if len(c) == len(v)
            and sum(x != y for x, y in zip(c, v)) == 1]
    return near[0] if len(near) == 1 else None


def snap_credits(value: Any) -> Any:
    """หน่วยกิตต้องเป็นจำนวนเต็ม 0-12 เท่านั้น นอกช่วงนี้ถือว่าอ่านผิด"""
    if value is None:
        return None
    try:
        n = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 12 else None


def apply_hard_constraints(rec: dict, registry: set[str] | None = None) -> dict:
    """
    บังคับข้อจำกัดกับ record ด้วยโค้ด ไม่ใช่ด้วย prompt

    นี่คือจุดต่างสำคัญของ pipeline "constrained"
    prompt เป็นเพียง "การขอร้อง" โมเดลจะทำตามหรือไม่ก็ได้
    แต่โค้ดตรงนี้เป็น "การบังคับ" ที่โมเดลข้ามไม่ได้
    """
    rec = copy.deepcopy(rec)
    for sem in _semesters(rec):
        for c in sem.get("subject") or []:
            c["grade_earn"] = snap_grade(c.get("grade_earn"))
            c["subject_id"] = snap_code(c.get("subject_id"), registry)
            c["credit"] = snap_credits(c.get("credit"))
    return rec


def _semesters(rec: Any) -> list:
    """ดึง list ภาคการศึกษาตาม TRANSCRIPT_SCHEMA ของ Lab 7A: transcript_detail.semesters"""
    if not isinstance(rec, dict):
        return []
    td = rec.get("transcript_detail")
    sems = td.get("semesters") if isinstance(td, dict) else None
    return [s for s in sems if isinstance(s, dict)] if isinstance(sems, list) else []


def llm_fix(rec: dict, method: str, registry: set[str] | None = None) -> dict:
    """แก้ข้อความหลัง OCR ตามวิธีที่เลือก"""
    if method == "none":
        return rec
    meta = rec.get("_meta")
    rec = {k: v for k, v in rec.items() if k != "_meta"}   # _meta ไม่ใช่ข้อมูลเอกสาร ห้ามส่งให้ LLM
    out = _llm_fix_body(rec, method, registry)
    if meta is not None:
        out["_meta"] = {**meta, "fix": method, "fix_ollama_version": ollama_version()}
    return out


def _llm_fix_body(rec: dict, method: str, registry: set[str] | None) -> dict:
    if method == "free":
        raw = ollama_generate(MODEL_TEXT,
                              FREE_FIX_PROMPT + json.dumps(rec, ensure_ascii=False),
                              fmt="json")
        out = parse_json_loose(raw)
        return out or rec
    if method == "constrained":
        raw = ollama_generate(MODEL_TEXT,
                              CONSTRAINED_FIX_PROMPT + json.dumps(rec, ensure_ascii=False),
                              fmt="json")
        out = parse_json_loose(raw) or rec
        # ขั้นบังคับด้วยโค้ด — ทำหลัง LLM เสมอ เพื่อกันโมเดลแอบแก้ฟิลด์ต้องห้าม
        out = apply_hard_constraints(out, registry)
        # ฟิลด์ต้องห้ามต้องเท่าเดิมจริง ๆ ไม่งั้นคืนค่าเดิมทับ
        out = restore_forbidden_fields(rec, out, registry)
        return out
    raise ValueError(f"ไม่รู้จักวิธีแก้: {method}")


def restore_forbidden_fields(orig: dict, fixed: dict,
                             registry: set[str] | None) -> dict:
    """
    ตรวจว่า LLM แอบแก้ฟิลด์ต้องห้ามหรือไม่ ถ้าแก้ ให้ดึงค่าเดิมกลับมา

    ยกเว้นกรณีเดียว: ค่าที่ผ่าน snap_* แล้วเปลี่ยน ถือว่าเป็นการแก้ด้วยกฎ ไม่ใช่การเดา
    """
    o_sems = _semesters(orig)
    f_sems = _semesters(fixed)
    for si, f_sem in enumerate(f_sems):
        if si >= len(o_sems):
            break
        o_courses = o_sems[si].get("subject") or []
        f_courses = f_sem.get("subject") or []
        for ci, fc in enumerate(f_courses):
            if ci >= len(o_courses) or not isinstance(fc, dict):
                break
            oc = o_courses[ci]
            fc["grade_earn"] = snap_grade(oc.get("grade_earn"))
            fc["subject_id"] = snap_code(oc.get("subject_id"), registry)
            fc["credit"] = snap_credits(oc.get("credit"))
    # student_id อยู่ส่วนหัว ก็เป็นฟิลด์ต้องห้ามเหมือนกัน
    o_head, f_head = orig.get("header_detail"), fixed.get("header_detail")
    if isinstance(o_head, dict) and isinstance(f_head, dict):
        f_head["student_id"] = o_head.get("student_id")
    return fixed


def build_registry(gt: dict) -> set[str]:
    """
    สร้างทะเบียนรหัสวิชาจากเฉลย — ใช้ในแล็บเท่านั้น

    ในระบบจริง ทะเบียนนี้มาจากฐานข้อมูลรายวิชาของสำนักทะเบียน
    ซึ่งเป็นข้อมูลสาธารณะและไม่ใช่ข้อมูลส่วนบุคคล จึงใช้ได้อย่างถูกต้อง
    """
    codes = set()
    for sem in _semesters(gt):
        for c in sem.get("subject") or []:
            code = c.get("subject_id")
            if code:
                codes.add(re.sub(r"\D", "", str(code)))
    return codes


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 5 — การวัดผล
# ═══════════════════════════════════════════════════════════════════════

def _item_key(parent: str, item: Any, i: int) -> str | None:
    """
    กุญแจที่เสถียรของสมาชิกใน list ภาคการศึกษา/รายวิชา  (None = ใช้ index ตามเดิม)

    ⚠️ ห้ามจับคู่ตาม index: ถ้าโมเดลอ่านตกไป 1 แถว ทุกแถวหลังจากนั้นจะเลื่อนหมด
    จึงใช้กุญแจเดียวกับ evaluate() ของ Lab 7A คือ "ปี/ภาค" และ "รหัสวิชา"
    """
    if not isinstance(item, dict):
        return None
    if parent.endswith("semesters"):
        return f"{norm(item.get('year'))}/{norm(item.get('sem_num'))}"
    if parent.endswith("subject"):
        return norm(item.get("subject_id")) or "?"
    return None


def flatten(rec: Any, prefix: str = "", keys_from: Any = None) -> dict[str, Any]:
    """
    แผ่ record ให้เป็น {เส้นทางของฟิลด์: ค่า}

    ตัวอย่าง: transcript_detail.semesters[2561/1].subject[13006006#0].grade_earn -> "c+"
    ทำแบบนี้เพื่อให้เทียบกับเฉลยได้ทีละฟิลด์ ไม่ใช่เทียบทั้งก้อน

    keys_from — record อ้างอิงที่มีโครงสร้างเดียวกัน (ใช้ตอน triage)
      ให้สมาชิกตำแหน่งที่ i ยืมกุญแจของ keys_from ตำแหน่งเดียวกัน
      เพราะถ้า LLM แก้รหัสวิชาผิด กุญแจจะเปลี่ยน แล้วการทำพังนั้นจะหายไปจากการนับ
    """
    out: dict[str, Any] = {}
    if isinstance(rec, dict):
        ref = keys_from if isinstance(keys_from, dict) else {}
        for k, v in rec.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k, ref.get(k)))
    elif isinstance(rec, list):
        ref = keys_from if isinstance(keys_from, list) else []
        seen: Counter = Counter()
        for i, v in enumerate(rec):
            src = ref[i] if i < len(ref) else v
            key = _item_key(prefix, src, i)
            if key is None:
                label = str(i)
            else:
                # กุญแจซ้ำได้ (เช่น ภาคที่อ่านปีไม่ออกทั้งคู่) จึงต่อท้ายลำดับที่พบ
                label = f"{key}#{seen[key]}"
                seen[key] += 1
            out.update(flatten(v, f"{prefix}[{label}]",
                               ref[i] if i < len(ref) else None))
    else:
        out[prefix] = rec
    return out


def norm(v: Any) -> str:
    """
    ปรับค่าให้เป็นรูปมาตรฐานก่อนเทียบ

    ทำไมต้อง normalize: OCR มักคืน "B +" หรือ " B+ " หรือ "b+"
    ถ้าไม่ปรับก่อน จะนับเป็นผิดทั้งที่เนื้อหาถูก ทำให้ประเมินโมเดลต่ำกว่าจริง
    แต่ต้องระวังไม่ normalize แรงเกินจนกลบความผิดที่มีความหมาย
    (เช่น ห้ามตัด "+" ทิ้ง เพราะ B กับ B+ คนละเกรด)
    """
    if v is None:
        return ""
    # ลบช่องว่างทั้งหมด เพราะ ground truth ของเราเก็บแบบไม่มีช่องว่าง
    # ("แคลคูลัส1", "bachelorofengineering") — ระดับเดียวกับ normalize "strict" ของ Lab 7
    s = unicodedata.normalize("NFC", str(v))
    s = re.sub(r"\s+", "", s)
    s = s.replace("\u200b", "").replace("\ufeff", "")
    # ตัวเลขไทยเป็นอารบิก
    s = s.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    return s.casefold()


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate — ระยะแก้ไขระดับตัวอักษร หารด้วยความยาวเฉลย"""
    ref, hyp = str(ref), str(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


@dataclass
class Triage:
    """
    ผลการจำแนกว่าการ "แก้" ของ LLM ให้ผลอย่างไรในแต่ละฟิลด์

    นี่คือหัวใจของ Lab 8A ทั้งแล็บ

    ตัวเลข accuracy เพียงตัวเดียวโกหกได้ง่ายมาก
    วิธีที่แก้ถูก 10 ช่อง แต่ทำพัง 8 ช่อง จะได้กำไรสุทธิ +2
    ดูเผิน ๆ เหมือนดีขึ้น แต่ใช้กับ transcript ไม่ได้เลย
    เพราะเราไม่มีทางรู้ว่า 8 ช่องที่พังคือช่องไหน — ต้องตรวจใหม่ทั้งใบอยู่ดี
    """
    fixed: int = 0             # เดิมผิด -> แก้แล้วถูก        (กำไร)
    corrupted: int = 0         # เดิมถูก -> แก้แล้วผิด        (ขาดทุน อันตรายที่สุด)
    kept_ok: int = 0           # เดิมถูก -> ยังถูก             (ไม่เปลี่ยน ดี)
    not_fixed: int = 0         # เดิมผิด -> ยังผิดเท่าเดิม     (ไม่เปลี่ยน เฉย ๆ)
    changed_wrong: int = 0     # เดิมผิด -> เปลี่ยนแต่ยังผิด    (เสียเวลาเปล่า)
    abstained_ok: int = 0      # เดิมถูก -> ตอบ null           (เสียของ แต่ปลอดภัย)
    abstained_bad: int = 0     # เดิมผิด -> ตอบ null           (ดีมาก ยอมรับว่าไม่รู้)
    critical_corrupted: int = 0  # corrupted ที่เกิดในฟิลด์สำคัญ
    examples: list = field(default_factory=list)

    @property
    def net(self) -> int:
        """กำไรสุทธิ — ตัวเลขที่คนมักดูตัวเดียวแล้วตัดสินใจผิด"""
        return self.fixed - self.corrupted

    @property
    def safety(self) -> float:
        """
        อัตราส่วนความปลอดภัย = แก้ถูก / (แก้ถูก + ทำพัง)

        1.00 = แก้แล้วไม่เคยทำพังเลย  ใช้กับงานที่ตรวจซ้ำไม่ได้
        0.50 = แก้ถูกเท่ากับทำพัง      ไม่มีประโยชน์
        ต่ำกว่า 0.90 ไม่ควรใช้กับ transcript
        """
        d = self.fixed + self.corrupted
        return self.fixed / d if d else 1.0


def triage(base: dict, fixed_rec: dict, gt: dict,
           max_examples: int = 12) -> Triage:
    """เทียบสามทาง: ก่อนแก้ / หลังแก้ / เฉลย แล้วจำแนกทุกฟิลด์"""
    # หลังแก้ต้องใช้กุญแจของ "ก่อนแก้" (จำนวนแถวเท่าเดิมตามกติกา)
    # ไม่งั้นถ้า LLM แก้รหัสวิชาผิด ทั้งแถวจะหลุดไปไม่ถูกนับว่าทำพัง
    fb, fg = flatten(base), flatten(gt)
    ff = flatten(fixed_rec, keys_from=base)
    t = Triage()
    for key, g in fg.items():
        b = fb.get(key)
        # ฟิลด์ที่หายไปจากผลหลังแก้ = LLM ลบทิ้ง นับเป็น null ไม่ใช่ "ไม่เปลี่ยน"
        # (เคยใช้ค่าเดิมแทน ทำให้ LLM ที่คืน JSON ว่างทั้งก้อนได้ kept_ok เต็ม)
        f = ff.get(key)
        gn, bn, fn = norm(g), norm(b), norm(f)
        b_ok, f_ok = (bn == gn), (fn == gn)
        is_null = (f is None or fn == "")
        is_crit = key.rsplit(".", 1)[-1] in CRITICAL_FIELDS

        if is_null and not (g is None or gn == ""):
            if b_ok:
                t.abstained_ok += 1
            else:
                t.abstained_bad += 1
        elif b_ok and f_ok:
            t.kept_ok += 1
        elif b_ok and not f_ok:
            t.corrupted += 1
            if is_crit:
                t.critical_corrupted += 1
            if len(t.examples) < max_examples:
                t.examples.append({"field": key, "kind": "CORRUPTED",
                                   "gt": g, "before": b, "after": f})
        elif (not b_ok) and f_ok:
            t.fixed += 1
            if len(t.examples) < max_examples:
                t.examples.append({"field": key, "kind": "FIXED",
                                   "gt": g, "before": b, "after": f})
        elif bn == fn:
            t.not_fixed += 1
        else:
            t.changed_wrong += 1
    return t


def field_accuracy(pred: dict, gt: dict) -> dict:
    """
    ความถูกต้องระดับฟิลด์ + CER เฉลี่ย + จำนวน hallucination

    hallucination = เฉลยว่างเปล่า แต่โมเดลใส่ค่ามาให้
    (แนวคิดเดียวกับคอลัมน์ Hall ใน Lab 7A)
    """
    # fp, fg = flatten(pred), flatten(gt)
    fp, fg = flatten({k: v for k, v in pred.items() if k != "_meta"}), flatten(gt)

    total = correct = hall = 0
    cer_sum = 0.0
    for key, g in fg.items():
        p = fp.get(key)
        gn, pn = norm(g), norm(p)
        total += 1
        if gn == pn:
            correct += 1
        cer_sum += cer(gn, pn)
        if gn == "" and pn != "":
            hall += 1
    # ฟิลด์ที่โมเดลสร้างเกินมาจากเฉลย ก็นับเป็น hallucination เช่นกัน
    hall += sum(1 for k, v in fp.items() if k not in fg and norm(v) != "")
    return {
        "fields": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "cer": cer_sum / total if total else 0.0,
        "hallucinated": hall,
    }


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 6 — คำสั่ง sweep : รันทุกระดับ noise × ทุกวิธีทำความสะอาด
# ═══════════════════════════════════════════════════════════════════════

# def cmd_sweep(args) -> None:
#     import cv2

#     noisy_dir = Path(args.input)
#     out_dir = Path(args.output)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

#     levels = args.levels.split(",") if args.levels else NOISE_LEVELS
#     methods = args.methods.split(",") if args.methods else CLEAN_METHODS

#     err_log_path = out_dir / "errors.log"
#     err_log = err_log_path.open("a", encoding="utf-8")

#     def log_error(context: str, e: Exception) -> None:
#         print(f"    ! {context}: {type(e).__name__} {e}")
#         err_log.write(f"\n{'=' * 68}\n{context}\n{'=' * 68}\n")
#         err_log.write(traceback.format_exc())
#         err_log.flush()

#     # รวมงานตามโมเดล: Typhoon อ่านทุกภาพให้เสร็จ แล้ว Qwen ค่อยจัด JSON ทั้งหมด
#     # ช่วยลดการโหลดโมเดลสลับไปมาบนการ์ดจอที่มี VRAM ไม่พอวางสองโมเดลพร้อมกัน
#     lab7 = _lab7_module()
#     lab7.assert_offline()
#     jobs = []
#     for level in levels:
#         lv_dir = noisy_dir / level
#         if not lv_dir.is_dir():
#             print(f"  ! ข้าม {level} (ไม่พบโฟลเดอร์)")
#             continue
#         pages = sorted(lv_dir.glob("page_*.png"))
#         for method in methods:
#             tag = f"{level}__{method}"
#             work = out_dir / tag
#             work.mkdir(parents=True, exist_ok=True)
#             t_ocr = time.time()

#             cleaned = []
#             for p in pages:
#                 img = cv2.imread(str(p))
#                 out_img = clean_image(img, method)
#                 q = work / p.name
#                 cv2.imwrite(str(q), out_img)
#                 cleaned.append(q)

#             markdown_pages = []
#             for q in cleaned:
#                 try:
#                     md = ocr_one_page(q, lab7)
#                 except Exception as e:
#                     # print(f"    ! {tag} {q.name}: {type(e).__name__} {e}")
#                     log_error(f"{tag} {q.name} (ocr_one_page)", e)
#                     continue
#                 (work / f"{q.stem}.md").write_text(md, encoding="utf-8")
#                 markdown_pages.append((q, md))

#             jobs.append({
#                 "level": level,
#                 "method": method,
#                 "tag": tag,
#                 "work": work,
#                 "markdown_pages": markdown_pages,
#                 "ocr_seconds": time.time() - t_ocr,
#             })

#     print("\n  OCR ครบแล้ว — เริ่มจัด Markdown เป็น JSON ต่อเนื่อง...")
#     rows = []
#     for job in jobs:
#         t_json = time.time()
#         level = job["level"]
#         method = job["method"]
#         tag = job["tag"]
#         work = job["work"]
#         merged = {
#             "header_detail": {},
#             "transcript_detail": {"semesters": []},
#             "footer_detail": {},
#         }
#         for q, md in job["markdown_pages"]:
#             try:
#                 rec = structure_one_page(md, lab7)
#             except Exception as e:
#                 # print(f"    ! {tag} {q.name}: {type(e).__name__} {e}")
#                 log_error(f"{tag} {q.name} (structure_one_page)", e)
#                 continue

#             if not merged["header_detail"] and rec.get("header_detail"):
#                 merged["header_detail"] = rec["header_detail"]
#             transcript = rec.get("transcript_detail") or {}
#             merged["transcript_detail"]["semesters"].extend(
#                 transcript.get("semesters") or [])
#             for key, value in transcript.items():
#                 if key != "semesters" and value is not None:
#                     merged["transcript_detail"][key] = value
#             if rec.get("footer_detail"):
#                 merged["footer_detail"] = rec["footer_detail"]

#         (work / "extracted.json").write_text(
#             json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
#         m = field_accuracy(merged, gt)
#         elapsed = job["ocr_seconds"] + (time.time() - t_json)
#         m.update({"level": level, "clean": method,
#                   "seconds": round(elapsed, 1)})
#         rows.append(m)
#         print(f"  [{tag:<24}] acc={m['accuracy']:.3f}  "
#               f"CER={m['cer']:.3f}  hall={m['hallucinated']}  "
#               f"({m['seconds']:.0f}s)")

#     # รวมกับผลของรอบก่อนหน้า แทนการเขียนทับ เพื่อให้รัน --levels/--methods
#     # เป็นบางส่วนได้เรื่อย ๆ โดยไม่เสียผลที่เคยรันไปแล้ว
#     sweep_path = out_dir / "sweep.json"
#     merged_rows = {}
#     if sweep_path.exists():
#         for r in json.loads(sweep_path.read_text(encoding="utf-8")):
#             merged_rows[(r["level"], r["clean"])] = r
#     for r in rows:
#         merged_rows[(r["level"], r["clean"])] = r
#     rows = list(merged_rows.values())

#     sweep_path.write_text(
#         json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
#     print_sweep_table(rows)

#     err_log.close()
#     if err_log_path.exists() and err_log_path.stat().st_size > 0:
#         print(f"\n  ! มี error เกิดขึ้นระหว่างรัน ดู traceback เต็มได้ที่ {err_log_path}")

def cmd_sweep(args) -> None:
    import cv2

    noisy_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    levels = args.levels.split(",") if args.levels else NOISE_LEVELS
    methods = args.methods.split(",") if args.methods else CLEAN_METHODS

    err_log_path = out_dir / "errors.log"
    err_log = err_log_path.open("a", encoding="utf-8")

    def log_error(context: str, e: Exception) -> None:
        print(f"    ! {context}: {type(e).__name__} {e}")
        err_log.write(f"\n{'=' * 68}\n{context}\n{'=' * 68}\n")
        err_log.write(traceback.format_exc())
        err_log.flush()

    lab7 = _lab7_module()
    lab7.assert_offline()
    doc = getattr(args, "doc", "") or doc_id_of(noisy_dir)
    info = doc_info(doc) if doc else {}
    rows = []
    for level in levels:
        lv_dir = noisy_dir / level
        if not lv_dir.is_dir():
            print(f"  ! ข้าม {level} (ไม่พบโฟลเดอร์)")
            continue
        pages = sorted(lv_dir.glob("page_*.png"))
        for method in methods:
            tag = f"{level}__{method}"
            work = out_dir / tag
            work.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            cleaned_bytes = []
            for p in pages:
                img = cv2.imread(str(p))
                out_img = clean_image(img, method)
                # out_img = clean_image(img, method, watermark_opacity=LEVEL_WATERMARK[level])
                q = work / p.name
                cv2.imwrite(str(q), out_img)
                cleaned_bytes.append(q.read_bytes())

            try:
                merged = lab7.pipeline_vlm(cleaned_bytes, save_md=work / "combined.md")
            except Exception as e:
                log_error(f"{tag} (pipeline_vlm)", e)
                continue
            merged["_meta"] = {**(merged.get("_meta") or {}), **run_meta(lab7, t0)}

            (work / "extracted.json").write_text(
                json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
            m = field_accuracy(merged, gt)
            m.update({**info, "level": level, "clean": method,
                      "seconds": round(time.time() - t0, 1),
                      "ollama_version": merged["_meta"]["ollama_version"]})
            rows.append(m)
            print(f"  [{tag:<24}] acc={m['accuracy']:.3f}  "
                  f"CER={m['cer']:.3f}  hall={m['hallucinated']}  "
                  f"({m['seconds']:.0f}s)")

    sweep_path = out_dir / "sweep.json"
    merged_rows = {}
    if sweep_path.exists():
        for r in json.loads(sweep_path.read_text(encoding="utf-8")):
            merged_rows[(r.get("doc_id", ""), r["level"], r["clean"])] = r
    for r in rows:
        merged_rows[(r.get("doc_id", ""), r["level"], r["clean"])] = r
    rows = list(merged_rows.values())

    sweep_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print_sweep_table(rows)

    err_log.close()
    if err_log_path.exists() and err_log_path.stat().st_size > 0:
        print(f"\n  ! มี error เกิดขึ้นระหว่างรัน ดู traceback เต็มได้ที่ {err_log_path}")


def print_sweep_table(rows: list[dict]) -> None:
    """ตารางเปรียบเทียบ ระดับ noise (แถว) × วิธีทำความสะอาด (คอลัมน์)"""
    if not rows:
        return
    methods = sorted({r["clean"] for r in rows}, key=CLEAN_METHODS.index)
    print()
    print("  ความถูกต้องระดับฟิลด์ (ยิ่งสูงยิ่งดี)")
    print("  " + "-" * 62)
    print("  {:<14}".format("ระดับ noise") + "".join(f"{m:>14}" for m in methods))
    print("  " + "-" * 62)
    for level in NOISE_LEVELS:
        sel = {r["clean"]: r for r in rows if r["level"] == level}
        if not sel:
            continue
        line = "  {:<14}".format(level)
        best = max((sel[m]["accuracy"] for m in methods if m in sel), default=0)
        for m in methods:
            if m in sel:
                v = sel[m]["accuracy"]
                mark = " *" if abs(v - best) < 1e-9 else "  "
                line += f"{v:>12.3f}{mark}"
            else:
                line += f"{'-':>14}"
        print(line)
    print("  " + "-" * 62)
    print("  * = ดีที่สุดของแถวนั้น")


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 7 — คำสั่ง fix : ให้ LLM แก้ แล้ว triage
# ═══════════════════════════════════════════════════════════════════════

def cmd_fix(args) -> None:
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    registry = build_registry(gt) if args.use_registry else None
    if registry:
        print(f"  ใช้ทะเบียนรหัสวิชา {len(registry)} รายการ")

    targets = sorted(p for p in in_dir.glob("*/extracted.json"))
    if not targets:
        print(f"  ! ไม่พบ extracted.json ใน {in_dir} — รัน sweep ก่อน")
        return

    methods = args.methods.split(",") if args.methods else ["free", "constrained"]
    doc = (getattr(args, "doc", "")
           or next((r.get("doc_id") for r in _load_json(in_dir / "sweep.json") or []
                    if r.get("doc_id")), "")
           or doc_id_of(in_dir.parent / "noisy"))
    info = doc_info(doc) if doc else {}
    report = []

    for src in targets:
        tag = src.parent.name
        base = json.loads(src.read_text(encoding="utf-8"))
        for method in methods:
            t0 = time.time()
            try:
                fixed_rec = llm_fix(base, method, registry)
            except Exception as e:
                print(f"  ! {tag}/{method}: {type(e).__name__} {e}")
                continue
            d = out_dir / f"{tag}__fix_{method}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "fixed.json").write_text(
                json.dumps(fixed_rec, ensure_ascii=False, indent=2), encoding="utf-8")

            t = triage(base, fixed_rec, gt)
            before = field_accuracy(base, gt)
            after = field_accuracy(fixed_rec, gt)
            row = {
                **info, "tag": tag, "fix": method,
                "ollama_version": ollama_version(),
                "acc_before": round(before["accuracy"], 4),
                "acc_after": round(after["accuracy"], 4),
                "fixed": t.fixed, "corrupted": t.corrupted,
                "critical_corrupted": t.critical_corrupted,
                "not_fixed": t.not_fixed, "changed_wrong": t.changed_wrong,
                "abstained_bad": t.abstained_bad, "abstained_ok": t.abstained_ok,
                "net": t.net, "safety": round(t.safety, 3),
                "seconds": round(time.time() - t0, 1),
            }
            report.append(row)
            (d / "triage.json").write_text(
                json.dumps({**row, "examples": t.examples},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{tag}/{method:<11}] "
                  f"acc {before['accuracy']:.3f} -> {after['accuracy']:.3f} | "
                  f"แก้ถูก {t.fixed:>3}  ทำพัง {t.corrupted:>3} "
                  f"(สำคัญ {t.critical_corrupted}) | safety {t.safety:.2f}")

    (out_dir / "fix_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_triage_table(report)


def print_triage_table(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    print("  ตาราง triage — ทำไม accuracy ตัวเดียวจึงไม่พอ")
    print("  " + "-" * 90)
    print("  {:<26}{:<12}{:>8}{:>8}{:>8}{:>9}{:>10}".format(
        "ชุดข้อมูล", "วิธีแก้", "แก้ถูก", "ทำพัง", "สุทธิ", "สำคัญพัง", "safety"))
    print("  " + "-" * 90)
    for r in rows:
        warn = "  <-- อันตราย" if r["critical_corrupted"] > 0 else ""
        print("  {:<26}{:<12}{:>8}{:>8}{:>8}{:>9}{:>10.2f}{}".format(
            r["tag"][:25], r["fix"], r["fixed"], r["corrupted"],
            r["net"], r["critical_corrupted"], r["safety"], warn))
    print("  " + "-" * 90)
    print("  สุทธิ = แก้ถูก - ทำพัง   ·   safety = แก้ถูก / (แก้ถูก + ทำพัง)")
    print("  ถ้ามีการทำพังในฟิลด์สำคัญแม้ช่องเดียว ห้ามนำวิธีนั้นไปใช้กับ transcript จริง")


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 8 — รายงานรวม
# ═══════════════════════════════════════════════════════════════════════

def cmd_report(args) -> None:
    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sweep = _load_json(root / "out" / "sweep.json") or _load_json(root / "sweep.json")
    fixrep = (_load_json(root / "fixed" / "fix_report.json")
              or _load_json(root / "fix_report.json"))

    lines = ["level,clean,accuracy,cer,hallucinated,seconds"]
    for r in sweep or []:
        lines.append("{level},{clean},{accuracy:.4f},{cer:.4f},"
                     "{hallucinated},{seconds}".format(**r))
    (out / "sweep.csv").write_text("\n".join(lines), encoding="utf-8")

    lines = ["tag,fix,acc_before,acc_after,fixed,corrupted,"
             "critical_corrupted,net,safety"]
    for r in fixrep or []:
        lines.append("{tag},{fix},{acc_before},{acc_after},{fixed},"
                     "{corrupted},{critical_corrupted},{net},{safety}".format(**r))
    (out / "triage.csv").write_text("\n".join(lines), encoding="utf-8")

    if sweep:
        print_sweep_table(sweep)
        plot_ascii(sweep)
        make_curve_png(sweep, out / "curve.png")
    if fixrep:
        print_triage_table(fixrep)
    make_compare_png(root / "noisy", out / "compare.png")
    print(f"\n  บันทึก CSV ที่ {out}/")


def cmd_report_all(args) -> None:
    """
    รวมผลทุกเอกสารใน work/<doc_id>/ ไว้ที่ work/summary/

    คอลัมน์ doc_id/group/lang ทำให้แยกดูได้ว่า noise กระทบ
    ตรี-ไทย / ตรี-อังกฤษ / บัณฑิต-ไทย / บัณฑิต-อังกฤษ ต่างกันอย่างไร
    """
    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    docs = sorted(d for d in root.iterdir()
                  if d.is_dir() and d.resolve() != out.resolve()
                  and any((d / sub).is_dir() for sub in ("noisy", "out", "fixed", "denoise_out")))

    def tagged(rows, d):
        # แถวจากรอบก่อนแยกโฟลเดอร์ไม่มี doc_id — เติมจากชื่อโฟลเดอร์
        info = doc_info(d.name)
        return [{**info, **{k: v for k, v in r.items() if v not in ("", None)}}
                for r in rows or []]

    sweep, fixrep, den = [], [], []
    for d in docs:
        sweep += tagged(_load_json(d / "out" / "sweep.json"), d)
        fixrep += tagged(_load_json(d / "fixed" / "fix_report.json"), d)
        den += tagged([_load_json(p) for p in sorted((d / "denoise_out").glob("*__metrics.json"))], d)

    def write_csv(name, rows, cols):
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r.get(c, "")) for c in cols))
        (out / name).write_text("\n".join(lines), encoding="utf-8")

    id_cols = ["doc_id", "group", "lang"]
    write_csv("sweep_all.csv", sweep, id_cols + ["level", "clean", "accuracy", "cer",
                                                 "hallucinated", "fields", "seconds",
                                                 "ollama_version"])
    write_csv("triage_all.csv", fixrep, id_cols + [
        "tag", "fix", "acc_before", "acc_after", "fixed", "corrupted",
        "critical_corrupted", "abstained_ok", "abstained_bad", "net", "safety",
        "ollama_version"])
    write_csv("denoise_all.csv", den, id_cols + ["input", "clean", "accuracy", "cer",
                                                 "hallucinated", "correct", "fields",
                                                 "ollama_version"])

    print(f"  พบ {len(docs)} เอกสาร: {', '.join(d.name for d in docs)}")
    if sweep:
        # ค่าเฉลี่ยต่อกลุ่ม group/lang — ใช้ตารางเดิมได้เลยโดยรวมแถวเป็นกลุ่ม
        groups = sorted({(r["group"], r["lang"]) for r in sweep})
        for g, lang in groups:
            sel = [r for r in sweep if (r["group"], r["lang"]) == (g, lang)]
            agg = {}
            for r in sel:
                agg.setdefault((r["level"], r["clean"]), []).append(r["accuracy"])
            rows = [{"level": lv, "clean": m, "accuracy": sum(v) / len(v)}
                    for (lv, m), v in agg.items()]
            n = len({r["doc_id"] for r in sel})
            print(f"\n  === {g}/{lang} (เฉลี่ย {n} เอกสาร) ===")
            print_sweep_table(rows)
    if den:
        print("\n  denoise (ภาพเดียว)")
        for r in den:
            print(f"  {r['doc_id']:<10}{r.get('group', ''):<10}{r.get('lang', ''):<4}"
                  f"{Path(r['input']).parent.name:<16}{r['clean']:<7}"
                  f"acc={r['accuracy']:.3f}  CER={r['cer']:.3f}  hall={r['hallucinated']}")
    print(f"\n  บันทึก CSV ที่ {out}/")


def plot_ascii(rows: list[dict], width: int = 40) -> None:
    """กราฟแท่งแบบตัวอักษร — หา 'จุดพัง' ได้โดยไม่ต้องเปิดโปรแกรมอื่น"""
    print()
    print("  ความถูกต้องเทียบกับระดับ noise (วิธี clean=none)")
    for level in NOISE_LEVELS:
        r = next((x for x in rows if x["level"] == level and x["clean"] == "none"), None)
        if not r:
            continue
        n = int(round(r["accuracy"] * width))
        print(f"  {level:<14}|{'#' * n}{'.' * (width - n)}| {r['accuracy']:.3f}")


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 9 — selftest : ตรวจว่าโค้ดวัดผลถูกก่อนเผาเวลา GPU
# ═══════════════════════════════════════════════════════════════════════

def cmd_selftest(args=None) -> bool:
    """
    ทดสอบตรรกะการวัดผลด้วยข้อมูลสมมติที่เรารู้คำตอบอยู่แล้ว

    ทำไมต้องมีขั้นนี้: การรัน OCR ครบทุกระดับใช้เวลาเป็นชั่วโมง
    ถ้าโค้ดวัดผลผิด จะรู้ตัวก็ต่อเมื่อเสียเวลาไปหมดแล้ว
    """
    print("=" * 68)
    print("  selftest — ตรวจตรรกะการวัดผล")
    print("=" * 68)
    passed = failed = 0

    def ck(name, got, want):
        nonlocal passed, failed
        if got == want:
            print(f"  [ ok ] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}: ได้ {got!r} ต้องการ {want!r}")
            failed += 1

    # 1) normalize
    ck("norm ตัดช่องว่าง", norm(" B+ "), "b+")
    ck("norm เลขไทย", norm("๓.๕๐"), "3.50")
    ck("norm ไม่กลืน + ทิ้ง", norm("B") == norm("B+"), False)

    # 2) CER
    ck("cer เหมือนกัน", cer("abc", "abc"), 0.0)
    ck("cer ผิดหนึ่งตัว", round(cer("abc", "abd"), 4), round(1 / 3, 4))

    # 3) snap_grade — ต้องแก้ที่แก้ได้ และคืน None เมื่อกำกวม
    ck("snap 8+ เป็น B+", snap_grade("8+"), "B+")
    ck("snap b เป็น B", snap_grade(" b "), "B")
    ck("snap ขยะคืน None", snap_grade("###"), None)

    # 4) snap_code
    reg = {"06026240", "06026241", "90130001"}
    ck("code ต่างหลักเดียวแก้ได้", snap_code("06026241", reg), "06026241")
    ck("code ต่างสองหลักคืน None", snap_code("06026999", reg), None)
    ck("code ที่มีอยู่ผ่าน", snap_code("90130001", reg), "90130001")

    # 5) snap_credits
    ck("credits ปกติ", snap_credits("3"), 3)
    ck("credits นอกช่วงคืน None", snap_credits("33"), None)

    # 6) triage — กรณีสำคัญที่สุด  (ใช้ schema จริงของ Lab 7A)
    def subj(sid, name, credit, grade):
        return {"subject_id": sid, "subject_name": name, "type": None,
                "credit": credit, "grade_earn": grade}

    gt = {"header_detail": {"student_id": "71010001"},
          "transcript_detail": {"semesters": [
              {"year": 2561, "sem_num": 1, "subject": [
                  subj("06026240", "ระบบอัจฉริยะ", 3, "b+"),
                  subj("06026241", "คอมพิวเตอร์วิทัศน์", 3, "a"),
              ]},
              {"year": 2561, "sem_num": 2, "subject": [
                  subj("06026242", "แคลคูลัส1", 3, "c"),
              ]},
          ]}}

    def sub_of(rec, si, ci):
        return rec["transcript_detail"]["semesters"][si]["subject"][ci]

    base = copy.deepcopy(gt)
    sub_of(base, 0, 0)["subject_name"] = "ระบบอัจฉรืยะ"   # ผิด (จะถูกแก้)
    fixed_rec = copy.deepcopy(gt)
    sub_of(fixed_rec, 0, 1)["grade_earn"] = "A-"          # เดิมถูก -> พัง
    t = triage(base, fixed_rec, gt)
    ck("triage นับ fixed", t.fixed, 1)
    ck("triage นับ corrupted", t.corrupted, 1)
    ck("triage สุทธิเป็นศูนย์", t.net, 0)
    ck("triage จับ critical (grade_earn)", t.critical_corrupted, 1)
    ck("triage safety = 0.5", t.safety, 0.5)

    # LLM แก้รหัสวิชาผิด — กุญแจเปลี่ยน แต่ต้องยังนับเป็นทำพัง
    bad_code = copy.deepcopy(gt)
    sub_of(bad_code, 1, 0)["subject_id"] = "06026243"
    t3 = triage(gt, bad_code, gt)
    ck("triage จับการแก้ subject_id ผิด", t3.critical_corrupted, 1)

    # 7) abstain นับแยกจริง
    ab = copy.deepcopy(gt)
    sub_of(ab, 0, 0)["subject_name"] = None
    t2 = triage(base, ab, gt)
    ck("abstain จากค่าที่เดิมผิด", t2.abstained_bad, 1)
    ck("abstain ไม่ถูกนับเป็น corrupted", t2.corrupted, 0)

    # LLM คืนโครงว่างทั้งก้อน — ต้องไม่ถูกนับว่า "ยังถูก"
    empty = {"header_detail": {}, "transcript_detail": {"semesters": []},
             "footer_detail": {}}
    t4 = triage(gt, empty, gt)
    # เหลือ kept_ok แค่ 3 ช่องที่เฉลยเป็น null อยู่แล้ว (type ของทั้ง 3 วิชา)
    ck("JSON ว่างไม่นับเป็น kept_ok", t4.kept_ok, 3)
    ck("JSON ว่างนับเป็น abstain", t4.abstained_ok, 1 + 3 * 4 + 2 * 2)

    # 7b) บั๊ก A — ช่องว่าง/ตัวพิมพ์ไม่ถือว่าผิด
    ck("norm ไม่สนช่องว่าง (ไทย)", norm("แคลคูลัส 1"), norm("แคลคูลัส1"))
    ck("norm ไม่สนช่องว่าง (อังกฤษ)",
       norm("Bachelor of Engineering"), "bachelorofengineering")

    # 7c) บั๊ก B — อ่านตกแถวแรก แถวหลังต้องไม่เลื่อนไปผิดทั้งหมด
    drop = copy.deepcopy(gt)
    del drop["transcript_detail"]["semesters"][0]["subject"][0]
    m_drop = field_accuracy(drop, gt)
    # type เป็น null ในเฉลย แถวที่หายจึงผิด 4 ฟิลด์ (null กับ ไม่มี นับว่าตรงกัน)
    ck("อ่านตก 1 วิชา ผิดแค่ 4 ฟิลด์ของวิชานั้น",
       m_drop["fields"] - m_drop["correct"], 4)
    drop_sem = copy.deepcopy(gt)
    del drop_sem["transcript_detail"]["semesters"][0]
    m_sem = field_accuracy(drop_sem, gt)
    ck("อ่านตกทั้งภาค ภาคถัดไปยังถูก",
       m_sem["fields"] - m_sem["correct"], 2 + 2 * 4)

    # 7d) บั๊ก C — hard constraint ทำงานกับ schema จริง
    noisy = copy.deepcopy(gt)
    sub_of(noisy, 0, 0)["grade_earn"] = "8+"
    sub_of(noisy, 0, 1)["credit"] = "3"
    sub_of(noisy, 1, 0)["subject_id"] = "07026242"   # ต่างจาก 06026242 หลักเดียว
    snapped = apply_hard_constraints(noisy, build_registry(gt))
    ck("constraint snap grade_earn 8+ -> B+", sub_of(snapped, 0, 0)["grade_earn"], "B+")
    ck("constraint snap credit เป็น int", sub_of(snapped, 0, 1)["credit"], 3)
    ck("constraint snap subject_id ด้วยทะเบียน",
       sub_of(snapped, 1, 0)["subject_id"], "06026242")
    ck("registry อ่านจาก schema จริง", len(build_registry(gt)), 3)
    sneaky = copy.deepcopy(gt)
    sneaky["header_detail"]["student_id"] = "71010002"
    sub_of(sneaky, 0, 0)["grade_earn"] = "A"
    restored = restore_forbidden_fields(gt, sneaky, None)
    ck("restore คืน grade_earn เดิม", norm(sub_of(restored, 0, 0)["grade_earn"]), "b+")
    ck("restore คืน student_id เดิม", restored["header_detail"]["student_id"], "71010001")

    # 8) hallucination
    gt_h = {"header": {"name_th": "สมชาย", "name_en": ""}}
    pred_h = {"header": {"name_th": "สมชาย", "name_en": "Somchai"}}
    ck("นับ hallucination", field_accuracy(pred_h, gt_h)["hallucinated"], 1)

    # 9) โครงสร้างภาพ (ไม่ต้องใช้ Ollama)
    try:
        import cv2
        import numpy as np
        blank = np.full((200, 400, 3), 255, dtype="uint8")
        cv2.putText(blank, "06026240 B+", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.line(blank, (350, 15), (350, 185), (0, 0, 0), 2)
        rot = apply_rotation(blank, 3.0)
        est = estimate_skew(cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY))
        ck("deskew หาเส้นตั้งและคืนมุมหมุนกลับ", abs(est + 3.0) < 1.0, True)
        for m in CLEAN_METHODS:
            out = clean_image(blank, m)
            assert out.shape[:2] == blank.shape[:2]
        ck("clean_image ทุกวิธีคืนภาพขนาดเดิม", True, True)
    except ImportError:
        print("  [skip] ไม่มี opencv จึงข้ามการทดสอบภาพ")

    print("=" * 68)
    print(f"  ผ่าน {passed} · ไม่ผ่าน {failed}")
    print("=" * 68)
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════
#  ส่วนที่ 10 — denoise : ทำความสะอาด + OCR + เทียบ gt ให้ภาพเดียว
# ═══════════════════════════════════════════════════════════════════════
#
#  ต่างจาก sweep ตรงที่ทำทีละภาพ ใช้ตอนอยากทดสอบไวว่า level/method คู่ไหน
#  ใช้ได้จริง โดยไม่ต้องรอให้ครบทุก combination ก่อน
# ═══════════════════════════════════════════════════════════════════════

def cmd_denoise(args) -> None:
    import cv2

    src = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    t_start = time.time()

    def step(msg: str) -> None:
        print(f"  [{time.time() - t_start:6.1f}s] {msg}")

    step(f"[1/3] อ่านภาพ {src} แล้วทำความสะอาดด้วยวิธี '{args.method}' ...")
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"❌ อ่านภาพไม่ได้: {src}")
    cleaned = clean_image(img, args.method)
    cleaned_path = out_dir / f"{src.stem}__{args.method}.png"
    cv2.imwrite(str(cleaned_path), cleaned)
    step(f"      บันทึกภาพที่สะอาดแล้ว: {cleaned_path}")

    # step("[2/3] เรียก Lab 7A OCR ภาพที่สะอาดแล้ว ...")
    # md, rec = extract_one_page(cleaned_path)
    # (out_dir / f"{src.stem}.md").write_text(md, encoding="utf-8")
    # pred_path = out_dir / f"{src.stem}__pred.json"
    # pred_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    # step(f"      บันทึกผลที่โมเดลอ่านได้: {pred_path}")
    step("[2/3] เรียก Lab 7A OCR ภาพที่สะอาดแล้ว ...")
    md_path = out_dir / f"{src.stem}.md"
    md, rec = extract_one_page(cleaned_path, save_md=md_path)
    pred_path = out_dir / f"{src.stem}__pred.json"
    pred_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    step(f"      บันทึกผลที่โมเดลอ่านได้: {pred_path}")

    step("[3/3] เทียบผลกับ ground truth ...")
    m = field_accuracy(rec, gt)
    step(f"      accuracy={m['accuracy']:.3f}  CER={m['cer']:.3f}  "
         f"hallucinated={m['hallucinated']}  ({m['correct']}/{m['fields']} ฟิลด์ถูก)")
    info = doc_info(args.doc) if getattr(args, "doc", "") else {}
    (out_dir / f"{src.stem}__{args.method}__metrics.json").write_text(
        json.dumps({**info, "input": str(src), "clean": args.method, **m,
                    "ollama_version": rec["_meta"]["ollama_version"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  เสร็จสิ้น รวมเวลา {time.time() - t_start:.1f} วินาที")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # vlm -> ocr_system -> src -> ราก
DATA_ROOT = PROJECT_ROOT / "data"   # data/input (ตรี) · data/input_G (บัณฑิต)
GT_ROOT = PROJECT_ROOT / "data"     # data/ground_truth · data/ground_truth_G
WORK_ROOT = PROJECT_ROOT / "work"


def find_pdf(doc_id: str) -> Path:
    hits = sorted(DATA_ROOT.glob(f"*/{doc_id}.pdf"))
    if not hits:
        raise SystemExit(f"❌ ไม่พบ PDF ของเอกสาร {doc_id} ใน {DATA_ROOT}/*/")
    return hits[0]


def find_gt(doc_id: str) -> Path:
    """เฉลยชื่อ Json_<id>_th.json หรือ Json_<id>_en.json อยู่ใน data/ground_truth หรือ data/ground_truth_G"""
    hits = sorted(GT_ROOT.glob(f"*/Json_{doc_id}_*.json"))
    if not hits:
        raise SystemExit(f"❌ ไม่พบเฉลยของเอกสาร {doc_id} ใน {GT_ROOT}/*/")
    return hits[0]


def doc_info(doc_id: str) -> dict:
    """ป้ายกำกับของเอกสาร: group (ตรี/บัณฑิต จากชื่อโฟลเดอร์เฉลย) และ lang (th/en)"""
    try:
        gt = find_gt(doc_id)
    except SystemExit:
        return {"doc_id": doc_id, "group": "", "lang": ""}
    return {"doc_id": doc_id,
            "group": "grad" if gt.parent.name.endswith("_G") else "undergrad",
            "lang": gt.stem.rsplit("_", 1)[-1]}


def doc_id_of(noisy_dir: Path) -> str:
    """เดา doc_id จาก manifest.json ของ noise (ใช้เมื่อไม่ได้ระบุ --doc)"""
    m = _load_json(noisy_dir / "manifest.json") or {}
    return Path(m.get("source", "")).stem


def resolve_doc_paths(args) -> None:
    """
    เติม -i / -o / -g ที่ไม่ได้ระบุ จาก --doc ตามโครง

        work/<doc_id>/noisy        ← noise
        work/<doc_id>/denoise_out  ← denoise
        work/<doc_id>/out          ← sweep
        work/<doc_id>/fixed        ← fix
        work/<doc_id>/report       ← report
        work/summary               ← report --all

    ถ้าระบุ -i / -o / -g เอง จะใช้ค่านั้นตามเดิม (คำสั่งแบบเอกสารเดียวใช้ได้เหมือนเก่า)
    """
    doc = getattr(args, "doc", None)
    work = Path(getattr(args, "work", None) or WORK_ROOT)
    cmd = args.cmd

    if cmd == "report" and args.all:
        args.input = args.input or str(work)
        args.output = args.output or str(work / "summary")
        return

    if doc:
        d = work / doc
        defaults = {
            "noise":   (lambda: find_pdf(doc), d / "noisy"),
            "denoise": (lambda: d / "noisy" / args.level / "page_01.png", d / "denoise_out"),
            "sweep":   (lambda: d / "noisy", d / "out"),
            "fix":     (lambda: d / "out", d / "fixed"),
            "report":  (lambda: d, d / "report"),
        }[cmd]
        if not args.input:
            args.input = str(defaults[0]())
        if not args.output:
            args.output = str(defaults[1])
        if hasattr(args, "gt") and not args.gt:
            args.gt = str(find_gt(doc))

    missing = [f for f in ("input", "output", "gt")
               if hasattr(args, f) and not getattr(args, f)]
    if missing:
        raise SystemExit(f"❌ ต้องระบุ {', '.join('--' + m for m in missing)} "
                         f"หรือใช้ --doc <doc_id> ให้เติมให้อัตโนมัติ")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Lab 8A — ทำเอกสารให้สะอาด และวัดว่า LLM แก้หรือทำพัง",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="ตรวจสภาพแวดล้อม")
    sub.add_parser("selftest", help="ทดสอบตรรกะการวัดผล (ไม่ต้องใช้ Ollama)")

    def doc_opts(p):
        p.add_argument("--doc", default="",
                       help="รหัสเอกสาร เช่น 71010001 — เติม -i/-o/-g ให้เป็น work/<doc>/... อัตโนมัติ")
        p.add_argument("--work", default=str(WORK_ROOT),
                       help="โฟลเดอร์รากของผลลัพธ์ (ค่าเริ่มต้น work/ ที่รากโปรเจกต์)")

    p = sub.add_parser("noise", help="สร้างชุดข้อมูล noise 4 ระดับ (L1-L4)")
    p.add_argument("-i", "--input", help="PDF หรือโฟลเดอร์ภาพต้นฉบับ")
    p.add_argument("-o", "--output")
    p.add_argument("--seed", type=int, default=42)
    doc_opts(p)

    p = sub.add_parser("denoise", help="ทำความสะอาด + OCR + เทียบ gt ให้ภาพเดียว")
    p.add_argument("-i", "--input",
                   help="ไฟล์ภาพ noise ไฟล์เดียว เช่น work/noisy/L3_tilted_copy/page_01.png")
    p.add_argument("-m", "--method", default="light", choices=CLEAN_METHODS,
                   help="วิธีทำความสะอาด (none/light/heavy)")
    p.add_argument("-g", "--gt", help="ไฟล์เฉลย JSON")
    p.add_argument("-o", "--output", help="โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น work/denoise_out)")
    p.add_argument("--level", default="L0_clean", choices=NOISE_LEVELS,
                   help="ใช้คู่กับ --doc เพื่อเลือกภาพ work/<doc>/noisy/<level>/page_01.png")
    doc_opts(p)

    p = sub.add_parser("sweep", help="รัน OCR ทุกระดับ x ทุกวิธีทำความสะอาด")
    p.add_argument("-i", "--input", help="โฟลเดอร์ที่ได้จาก noise")
    p.add_argument("-g", "--gt", help="ไฟล์เฉลย JSON")
    p.add_argument("-o", "--output")
    p.add_argument("--levels", default="", help="เช่น L0_clean,L2_watermark")
    p.add_argument("--methods", default="", help="เช่น none,light")
    doc_opts(p)

    p = sub.add_parser("fix", help="ให้ LLM แก้ข้อความ แล้ว triage")
    p.add_argument("-i", "--input", help="โฟลเดอร์ที่ได้จาก sweep")
    p.add_argument("-g", "--gt")
    p.add_argument("-o", "--output")
    p.add_argument("--methods", default="", help="เช่น free,constrained")
    p.add_argument("--use-registry", action="store_true",
                   help="ใช้ทะเบียนรหัสวิชาช่วยแก้รหัส")
    doc_opts(p)

    p = sub.add_parser("report", help="สรุปผลเป็นตารางและ CSV")
    p.add_argument("-i", "--input", help="โฟลเดอร์ work/ ของเอกสาร (หรือ work/ เมื่อใช้ --all)")
    p.add_argument("-o", "--output")
    p.add_argument("--all", action="store_true",
                   help="รวมผลทุกเอกสารใน work/*/ ไว้ที่ work/summary/")
    doc_opts(p)

    args = ap.parse_args()
    if args.cmd == "denoise" and not args.doc and not args.output:
        args.output = str(WORK_ROOT / "denoise_out")
    if args.cmd not in ("check", "selftest"):
        resolve_doc_paths(args)

    if args.cmd == "check":
        sys.exit(0 if check_environment() else 1)
    if args.cmd == "selftest":
        sys.exit(0 if cmd_selftest(args) else 1)
    if args.cmd == "noise":
        make_noise_dataset(Path(args.input), Path(args.output), args.seed)
    elif args.cmd == "denoise":
        cmd_denoise(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "fix":
        cmd_fix(args)
    elif args.cmd == "report":
        cmd_report_all(args) if args.all else cmd_report(args)



if __name__ == "__main__":
    main()
