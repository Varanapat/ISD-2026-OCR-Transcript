#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 lab7a_transcript.py
 Lab 7A — สกัดข้อมูลใบแสดงผลการศึกษา (Transcript) ด้วย LLM ที่รันบนเครื่องตัวเอง
================================================================================

 วิชา 06026240 การพัฒนาระบบอัจฉริยะ  |  ภาควิชาเทคโนโลยีสารสนเทศ สจล.

 --------------------------------------------------------------------------
 ⚠️  ข้อบังคับด้านความเป็นส่วนตัว — อ่านก่อนรัน
 --------------------------------------------------------------------------
  สคริปต์นี้จึงถูกออกแบบให้ "รันแบบออฟไลน์ 100%":
   - ไม่มีการเรียก API ภายนอกใด ๆ  ไม่มี API key  ไม่มีค่าใช้จ่าย
   - โมเดลถูกดาวน์โหลดมาเก็บไว้ในเครื่องก่อนหน้า แล้วรันผ่าน Ollama ที่ localhost
   - มีฟังก์ชัน assert_offline() ตรวจสอบซ้ำก่อนเริ่มทำงาน

 ห้ามแก้ OLLAMA_HOST ให้ชี้ไปเซิร์ฟเวอร์ภายนอกโดยเด็ดขาด
 ห้ามนำ transcript ของผู้อื่นมาใช้โดยไม่ได้รับความยินยอมเป็นลายลักษณ์อักษร

 หมายเหตุ: transcript ที่ใช้ในแล็บนี้เป็นข้อมูลจำลอง จึงไม่ต้องปิดทับส่วนหัว
 และ "ต้องไม่ปิด" เพราะส่วนหัวคือข้อมูลที่บอกว่าผลการเรียนเป็นของใคร
 ซึ่งเป็นส่วนที่ต้องนำไปวัดความแม่นยำด้วย

 --------------------------------------------------------------------------
 วิธีใช้
 --------------------------------------------------------------------------
   (รันจากรากโปรเจกต์ ocr_system/ โดยมี PYTHONPATH=src หรือ pip install -e .)

   # ตรวจว่าเครื่องพร้อม (ต้องทำก่อนเสมอ)
   python -m ocr_system.vlm.lab7a_transcript --check

   # รันครบทุก pipeline แล้วเทียบกับ ground truth (ผลลง outputs/lab7a/)
   python -m ocr_system.vlm.lab7a_transcript \
       --input data/input/71010001.pdf \
       --gt    data/ground_truth/Json_71010001_th.json \
       --pipeline all

   # รันเฉพาะ pipeline เดียว (ตอน debug จะเร็วกว่า)
   python -m ocr_system.vlm.lab7a_transcript -i data/input/x.pdf -g data/ground_truth/x.json -p vlm

   # ประเมินผลจากไฟล์ JSON ที่รันไว้แล้ว (ไม่ต้องเรียกโมเดลซ้ำ)
   python -m ocr_system.vlm.lab7a_transcript --eval-only outputs/lab7a/pred_vlm.json --gt data/ground_truth/x.json

 --------------------------------------------------------------------------
 สามเส้นทาง (pipeline) ที่เราจะเปรียบเทียบกัน
 --------------------------------------------------------------------------
   [A] baseline : OpenCV preprocess -> Tesseract (tha+eng) -> parse ด้วย regex
                  = วิธี Computer Vision ดั้งเดิม (อันนี้ทำในแลปก่อนหน้าแล้ว)

   [B] vlm      : Typhoon-OCR (VLM เฉพาะทางภาษาไทย) -> ได้ Markdown
                  -> ส่ง Markdown เข้า text LLM พร้อม JSON Schema -> ได้ JSON
                  = สองขั้น "อ่านให้ครบ" แล้วค่อย "จัดโครงสร้าง"

 คำถามที่แล็บนี้ต้องการคำตอบ:
   "แต่ละ attribute (รหัสวิชา / ชื่อวิชา / เกรด / หน่วยกิต / GPA)
    pipeline ไหนแม่นกว่ากัน และ เพราะอะไร"
================================================================================
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Union

# --- โมดูลกลาง lab7_metrics อยู่ในแพ็กเกจเดียวกัน (src/ocr_system/vlm) -------
#     รองรับทั้ง  python -m ocr_system.vlm.lab7a_transcript  และการรันไฟล์ตรง ๆ
try:
    from . import lab7_metrics as M
except ImportError:
    import lab7_metrics as M  # type: ignore[no-redef]  # noqa: E402

# รากของโปรเจกต์ ocr_system/  (vlm -> ocr_system -> src -> ราก)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ==============================================================================
#  ส่วนที่ 0 — ค่าตั้งต้น (แก้ตรงนี้ได้ถ้าเครื่องคุณสเปกต่างจากที่แล็บกำหนด)
# ==============================================================================

# Ollama รันเป็น service อยู่บนเครื่องเรา  ค่า default คือ 127.0.0.1:11434
# 127.0.0.1 = loopback address = "ตัวเครื่องเอง" ข้อมูลไม่ออกจาก network card เลย
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# --- โมเดลที่ใช้ ---------------------------------------------------------
# ขั้นที่ 1 (อ่านภาพ -> ข้อความ): Typhoon-OCR เป็น VLM ที่ fine-tune มาสำหรับ
#   เอกสารไทยโดยเฉพาะ  ⚠️ สำคัญ: มันเป็น "task-specific model"
#   ใช้ได้เฉพาะกับ prompt ของมันเอง สั่งให้พ่น JSON ตาม schema เราไม่ได้
#   จึงต้องมีขั้นที่ 2 เสมอ
MODEL_OCR = os.getenv("LAB7_MODEL_OCR", "scb10x/typhoon-ocr1.5-3b")

# ขั้นที่ 2 (ข้อความ -> JSON): text LLM ที่รองรับ structured output
#   ถ้าเครื่องแรม >= 16GB แนะนำ qwen3:8b  ถ้าน้อยกว่านั้นใช้ qwen3:4b
MODEL_TEXT = os.getenv("LAB7_MODEL_TEXT", "qwen3:4b")

# ความละเอียดตอนแปลง PDF เป็นภาพ
# 150 DPI เพียงพอสำหรับตัวหนังสือขนาด 10-12pt
# 300 DPI ชัดกว่าแต่ภาพใหญ่ขึ้น 4 เท่า -> VLM ช้าลงมากและอาจเกิน context
DPI = int(os.getenv("LAB7_DPI", "150"))

REQUEST_TIMEOUT = 900       # วินาที — CPU-only อาจใช้เวลา 5-10 นาทีต่อหน้า

# ข้าม pipeline baseline (Tesseract) ทั้งหมด
#     export LAB7_SKIP_BASELINE=1
# ใช้เมื่อไม่ได้ติดตั้ง Tesseract หรือไม่ต้องการเสียเวลารัน baseline
SKIP_BASELINE = os.getenv("LAB7_SKIP_BASELINE", "").strip() in ("1", "true", "yes")

# ==============================================================================
#  ส่วนที่ 1 — ตรวจความพร้อมของเครื่อง และ ยืนยันว่าออฟไลน์
# ==============================================================================

def _need(mod: str, pipname: str = "") -> Any:
    """import แบบมีข้อความบอกวิธีติดตั้งเมื่อไม่เจอ"""
    try:
        return __import__(mod)
    except ImportError:
        raise SystemExit(
            f"\n❌ ไม่พบไลบรารี '{mod}'\n"
            f"   ติดตั้งด้วย:  pip install {pipname or mod}\n"
        )


def assert_offline() -> None:
    """
    ตรวจว่า OLLAMA_HOST ชี้ไปที่เครื่องตัวเองจริง

    ทำไมต้องตรวจ? เพราะถ้าใครเผลอ export OLLAMA_HOST=https://some-cloud.com
    ข้อมูล transcript จะถูกส่งออกไปโดยที่เราไม่รู้ตัว
    การตรวจแบบนี้เรียกว่า "fail closed" — ถ้าไม่แน่ใจ ให้หยุด ไม่ใช่ปล่อยผ่าน
    """
    allowed = ("127.0.0.1", "localhost", "0.0.0.0", "::1")
    host = OLLAMA_HOST.replace("http://", "").replace("https://", "").split(":")[0]
    if host not in allowed:
        raise SystemExit(
            f"\n❌ หยุดทำงานเพื่อความปลอดภัยของข้อมูล\n"
            f"   OLLAMA_HOST = {OLLAMA_HOST}\n"
            f"   ซึ่งไม่ใช่เครื่องภายใน  ข้อมูล transcript ห้ามออกนอกเครื่อง\n"
            f"   แก้โดย:  unset OLLAMA_HOST\n"
        )
    print(f"✓ ยืนยันโหมดออฟไลน์: {OLLAMA_HOST} (loopback เท่านั้น)")


def check_environment() -> bool:
    """เช็กทุกอย่างที่ต้องใช้ แล้วรายงานเป็นรายการ — รันด้วย --check"""
    ok = True
    print("\n" + "=" * 70)
    print("  ตรวจความพร้อมของเครื่อง")
    print("=" * 70)

    # --- 1) Ollama ติดตั้งหรือยัง ---
    exe = shutil.which("ollama")
    if exe:
        try:
            v = subprocess.run(["ollama", "--version"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
            print(f"  ✓ พบ Ollama: {v}")
        except Exception:
            print("  ✓ พบ Ollama (อ่านเวอร์ชันไม่ได้)")
    else:
        print("  ✗ ไม่พบคำสั่ง ollama  --> ดูขั้นตอนติดตั้งในเอกสารแล็บ ส่วนที่ 2")
        ok = False

    # --- 2) service รันอยู่ไหม ---
    try:
        requests = _need("requests")
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        installed = [m["name"] for m in r.json().get("models", [])]
        print(f"  ✓ Ollama service ทำงานที่ {OLLAMA_HOST}")
        print(f"    โมเดลที่มีในเครื่อง ({len(installed)}):")
        for m in installed:
            print(f"      - {m}")

        # --- 3) โมเดลที่แล็บต้องใช้ครบไหม ---
        for tag, role in [(MODEL_OCR, "อ่านภาพ"), (MODEL_TEXT, "จัด JSON")]:
            # ollama เติม ":latest" ให้อัตโนมัติ จึงต้องเทียบแบบตัดท้าย
            hit = any(i == tag or i.startswith(tag + ":") or i.split(":")[0] == tag
                      for i in installed)
            mark = "✓" if hit else "✗"
            print(f"  {mark} [{role}] {tag}" + ("" if hit else f"   --> ollama pull {tag}"))
            if not hit:
                ok = False
    except Exception as e:
        print(f"  ✗ ต่อ Ollama ไม่ได้: {e}")
        print("    --> เปิด terminal อีกหน้าต่างแล้วสั่ง:  ollama serve")
        ok = False

    # --- 4) ไลบรารี Python ---
    for mod, pip in [("fitz", "pymupdf"), ("PIL", "pillow"), ("requests", "requests")]:
        try:
            __import__(mod)
            print(f"  ✓ python: {mod}")
        except ImportError:
            print(f"  ✗ python: {mod}   --> pip install {pip}")
            ok = False

    # --- 5) ของที่ไม่จำเป็น — ขาดได้ ไม่ทำให้ --check ตก ---
    #
    # ⚠️ สังเกตว่าส่วนนี้ "ไม่มี ok = False" เลย
    #    เพราะถ้าบอกผู้ใช้ว่า "ไม่มีก็รันได้" แล้วยังทำให้ผลตรวจตก
    #    ก็เท่ากับหลอกผู้ใช้ ผลตรวจจะเชื่อถือไม่ได้
    #    หลักการ: สิ่งที่ทำให้ "ไม่พร้อม" ต้องเป็นสิ่งที่ขาดแล้วรันไม่ได้จริงเท่านั้น
    print("\n  ส่วนเสริม (ขาดได้ ไม่ทำให้ --check ตก):")

    # Tesseract ใช้กับ pipeline baseline อย่างเดียว
    # ปกติติดตั้งไว้แล้วตั้งแต่ Lab 5-6  ถ้าไม่ต้องการใช้ ให้ตั้ง
    #     export LAB7_SKIP_BASELINE=1
    # แล้วสคริปต์จะข้าม pipeline นี้ไปเลย ไม่ตรวจและไม่รัน
    if SKIP_BASELINE:
        print("  ○ tesseract — ข้ามตามค่า LAB7_SKIP_BASELINE=1")
    else:
        has_exe = shutil.which("tesseract") is not None
        try:
            __import__("pytesseract")
            has_lib = True
        except ImportError:
            has_lib = False

        if has_exe and has_lib:
            print("  ✓ tesseract (จาก Lab 5-6) — ใช้กับ pipeline baseline")
        else:
            miss = []
            if not has_lib:
                miss.append("pip install pytesseract")
            if not has_exe:
                miss.append("ติดตั้งตัว engine (ดูเอกสาร Lab 5)")
            print(f"  ✗ tesseract — {' + '.join(miss)}")
            print("      pipeline baseline จะถูกข้ามไป (pipeline vlm ยังใช้ได้ตามปกติ)")
            print("      ถ้าไม่ต้องการใช้ baseline เลย:  export LAB7_SKIP_BASELINE=1")

    try:
        __import__("pythainlp")
        print("  ✓ pythainlp (ตัดคำไทยสำหรับ WER)")
    except ImportError:
        print("  ✗ pythainlp   --> pip install pythainlp")
        print("      ถ้าไม่มี ค่า WER ของข้อความไทยจะไม่มีความหมาย")

    print("=" * 70)
    print("  พร้อมใช้งาน ✓" if ok else "  ยังไม่พร้อม — แก้ตามรายการ ✗ ด้านบน")
    print("=" * 70 + "\n")
    return ok


# ==============================================================================
#  ส่วนที่ 2 — เตรียมภาพจาก input
# ==============================================================================


def load_pages(path: str) -> list[bytes]:
    """
    รับได้ทั้ง PDF และไฟล์ภาพ  คืนค่าเป็น list ของ PNG bytes (หน้าละ 1 ตัว)

    ทำไมใช้ PyMuPDF (fitz) แทน pdf2image?
      pdf2image ต้องติดตั้ง poppler เพิ่มในระดับระบบปฏิบัติการ
      ซึ่งบน Windows ยุ่งยากมาก  ส่วน PyMuPDF เป็น pure pip install
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"❌ ไม่พบไฟล์: {path}")

    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        print(f"  อ่านไฟล์ภาพ: {p.name}")
        return [p.read_bytes()]

    if p.suffix.lower() != ".pdf":
        raise SystemExit(f"❌ ไม่รองรับนามสกุล {p.suffix}")

    fitz = _need("fitz", "pymupdf")
    doc = fitz.open(str(p))
    pages: list[bytes] = []

    # zoom = DPI / 72  เพราะ PDF ใช้หน่วย point (1/72 นิ้ว) เป็นค่าเริ่มต้น
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
        print(f"  แปลงหน้า {i + 1}/{len(doc)}  ({pix.width}x{pix.height} px @ {DPI} DPI)")
    doc.close()
    return pages


# ── หมายเหตุ: ทำไมแล็บนี้ไม่มีฟังก์ชันปิดทับข้อมูลส่วนบุคคล ────────────
#
#  transcript ที่ใช้ในแล็บนี้เป็น "ข้อมูลจำลอง" ทั้งหมด ชื่อและรหัสถูกสมมติขึ้น
#  จึงไม่จำเป็นต้องปิดทับ  และที่สำคัญกว่านั้นคือ "ต้องไม่ปิด" เพราะ
#
#      ส่วนหัวคือข้อมูลที่บอกว่า "ผลการเรียนนี้เป็นของใคร"
#      ถ้าปิดทิ้ง เอกสารที่สกัดได้จะไร้ประโยชน์ทันที
#      และเราจะวัดความแม่นยำของ 14 ฟิลด์ในส่วนหัวไม่ได้เลย
#
#  ⚠️ เมื่อนำระบบไปใช้กับ transcript ของจริง ต้องคุ้มครองข้อมูลด้วยวิธีอื่น
#     แทนการปิดทับ ซึ่งได้ผลดีกว่าและไม่ทำลายข้อมูล:
#       - เก็บผลลัพธ์ไว้ในเครื่องเท่านั้น (สคริปต์นี้บังคับด้วย assert_offline)
#       - ควบคุมสิทธิ์การเข้าถึงโฟลเดอร์ output
#       - ลบข้อมูลทันทีที่ใช้งานเสร็จ ตามหลัก data minimisation ของ PDPA
#       - ไม่พิมพ์ชื่อหรือรหัสนักศึกษาลง log


# ==============================================================================
#  ส่วนที่ 3 — JSON SCHEMA
# ==============================================================================
#
#  Schema นี้ "ต้องตรงกับ ground truth เป๊ะ ๆ" ทุก key
#  เพราะเราจะเอา output ไปเทียบกันตรง ๆ  ถ้าตั้งชื่อ key ต่างกันแม้ตัวเดียว
#  ตัวประเมินจะมองว่าโมเดลอ่านไม่ได้ ทั้งที่จริงอ่านได้
#
#  Ollama รองรับ "structured outputs": ส่ง JSON Schema เข้าไปใน field `format`
#  แล้วมันจะบังคับ decoder ให้พ่นเฉพาะ token ที่ตรง schema เท่านั้น
#  --> JSON พังไม่ได้ในเชิงไวยากรณ์ (แต่ "เนื้อหา" ยังผิดได้อยู่ดี!)
#
#  นี่คือความแตกต่างสำคัญที่ต้องเข้าใจ:
#      constrained decoding รับประกัน "รูปร่าง"  ไม่ได้รับประกัน "ความจริง"
# ==============================================================================

_S = {"type": "string"}
_SN = {"type": ["string", "null"]}          # string หรือ null ก็ได้
_IN = {"type": ["integer", "null"]}

TRANSCRIPT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "header_detail": {
            "type": "object",
            "properties": {
                "uni_name": _SN,
                "uni_address": _SN,
                "student_id": _SN,
                "faculty_name": _SN,
                "prename": _SN,          # คำนำหน้า: นาย / นางสาว / นาง
                "name": _SN,             # ชื่อ-สกุล
                "date_of_birth": _SN,    # รูปแบบ YYYY-MM-DD
                "admis_date": _SN,       # วันเข้าศึกษา
                "grad_date": _SN,        # วันสำเร็จการศึกษา; ถ้ายังไม่จบใช้ 0000-00-00
                "grad_reason": _SN,
                "degree": _SN,           # เช่น วิทยาศาสตรบัณฑิต
                "major": _SN,
                "program": _SN,
                "honor": _IN,            # 0 = ไม่ได้เกียรตินิยม, 1/2 = อันดับ
            },
            "required": ["student_id", "prename", "name", "faculty_name",
             "uni_name", "uni_address", "date_of_birth", "admis_date",
             "grad_date", "grad_reason", "degree", "program", "honor"],
        },
        "transcript_detail": {
            "type": "object",
            "properties": {
                "semesters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "year": _IN,          # ปีการศึกษา พ.ศ. เช่น 2561
                            "sem_num": _IN,       # 1, 2, หรือ 3 (ภาคฤดูร้อน)
                            # ⚠️ ชื่อสองตัวนี้ชวนสับสนมาก อ่านคำอธิบายใน prompt
                            "GPS": _SN,           # = เกรดเฉลี่ยเฉพาะภาคนี้
                            "GPA": _SN,           # = เกรดเฉลี่ยสะสม (cumulative)
                            "pass_reason": _SN,
                            "subject": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject_id": _SN,
                                        "subject_name": _SN,
                                        "type": _SN,
                                        "credit": _IN,
                                        "grade_earn": _SN,
                                    },
                                    "required": ["subject_id", "subject_name",
                                                 "credit", "grade_earn"],
                                },
                            },
                        },
                        "required":  ["year", "sem_num", "subject", "GPA", "GPS", "pass_reason"],
                    },
                },
                "master_comprehensive": _SN,
                "master_thesis": _SN,
                "master_qualify": _SN,
                "total_credits_earned": _IN,
                "cumulative_gpa": _SN,
            },
            "required": ["semesters", "cumulative_gpa", "total_credits_earned"],
        },
        "footer_detail": {
            "type": "object",
            "properties": {
                "updated_at": _SN,
                "by": {
                    "type": "object",
                    "properties": {
                        "by_signature": _SN,
                        "by_position": _SN,
                        "by_reg": _SN,
                    },
                },
            },
            "required" : ["updated_at", "by"],
        },
    },
    "required": ["header_detail", "transcript_detail"],
}


# ==============================================================================
#  ส่วนที่ 4 — PROMPT
# ==============================================================================
#
#  หลักการเขียน prompt สำหรับงานถอดข้อมูลจากเอกสาร (มี 4 ข้อ):
#
#   1. "ถอดตามที่เห็น ห้ามเดา"  — LLM ถูกฝึกมาให้ตอบให้ครบเสมอ
#      ถ้าไม่สั่งชัด ๆ มันจะเติมช่องว่างด้วยสิ่งที่ "น่าจะใช่"
#      สำหรับ transcript การเดาเกรดผิด 1 ตัว = ข้อมูลเสียหายทั้งใบ
#
#   2. "ยอมรับว่าไม่รู้ได้"  — ให้ทางออกกับโมเดล (ตอบ null)
#      ถ้าไม่ให้ทางออก มันจะ "ถูกบังคับ" ให้เดา
#
#   3. "ห้ามคำนวณเอง"  — ให้ลอก GPA ที่พิมพ์อยู่มาเลย
#      เพราะเราจะเอา GPA ที่พิมพ์ ไปตรวจกับ GPA ที่เราคำนวณจากตาราง
#      ถ้าปล่อยให้โมเดลคำนวณ มันจะตรงกันเสมอ = ตรวจอะไรไม่ได้
#
#   4. "อ่านทีละแถว ห้ามข้าม"  — failure mode อันดับหนึ่งของตารางคือ "ตกแถว"
# ==============================================================================

SYSTEM_PROMPT = """You are a precise document transcription system for Thai university transcripts.
You transcribe exactly what is printed. You never infer, correct, complete, or beautify.
When something is unreadable you output null. You are penalised heavily for guessing."""

# EXTRACT_PROMPT = """ต่อไปนี้คือข้อความที่ถอดจากใบแสดงผลการศึกษา (transcript) ของสถาบันในประเทศไทย
# จงแปลงเป็น JSON ตาม schema ที่กำหนดเท่านั้น

# === กติกาสำคัญ (ผิดข้อใดข้อหนึ่งถือว่าใช้งานไม่ได้) ===

# [1] ห้ามเดาเกรด
#     grade_earn ต้องเป็นหนึ่งใน: A B+ B C+ C D+ D F W S U I P T
#     ถ้าช่องใดเบลอ อ่านไม่ออก หรือกำกวม ให้ใส่ null — ห้ามเดาเป็นค่าที่ดูสมเหตุสมผล

# [2] อ่านตารางทีละแถว จากบนลงล่าง
#     ห้ามข้ามแถว ห้ามรวมสองแถวเป็นแถวเดียว ห้ามสลับลำดับ
#     ถ้าหน้ากระดาษแบ่งเป็นสองคอลัมน์ ให้อ่านคอลัมน์ซ้ายให้จบก่อนแล้วค่อยขึ้นคอลัมน์ขวา
#     รวมวิชาที่เทียบโอน วิชาที่ถอน (W) และวิชาที่ลงซ้ำ ทุกครั้งที่ปรากฏ

# [3] ⚠️ ความหมายของ GPA และ GPS ในเอกสารนี้ (อ่านให้ดี ชื่อมันสลับกับที่คนทั่วไปเข้าใจ)
#     "GPS" = เกรดเฉลี่ยของภาคการศึกษานั้นภาคเดียว  (semester GPA)
#     "GPA" = เกรดเฉลี่ยสะสมนับตั้งแต่ภาคแรกจนถึงภาคนั้น (cumulative GPA)
#     ในเอกสารมักพิมพ์ว่า "ผลการศึกษาประจำภาค" -> GPS
#                      และ "ผลการศึกษาสะสม"    -> GPA

# [4] ห้ามคำนวณ GPA หรือ GPS เอง — ให้คัดลอกตัวเลขที่พิมพ์อยู่ในเอกสารเท่านั้น
#     งานนี้คือการ "ถอดข้อความ" ไม่ใช่การคำนวณ
#     ถ้าช่องใดไม่มีพิมพ์ไว้ ให้ใส่ null ห้ามคำนวณขึ้นมาเติม

# [5] รูปแบบข้อมูล
#     - credit  : จำนวนเต็ม เช่น 3  (ไม่ใช่ "3(3-0-6)")
#     - year    : ปีการศึกษา พ.ศ. เช่น 2561
#     - sem_num : 1, 2, หรือ 3 (3 = ภาคฤดูร้อน)
#     - วันที่  : YYYY-MM-DD  ถ้ายังไม่สำเร็จการศึกษาให้ grad_date = "0000-00-00"
#     - แปลงเลขไทย ๐๑๒๓๔๕๖๗๘๙ เป็นเลขอารบิก
#     - honor   : 0 ถ้าไม่ได้เกียรตินิยม, 1 = อันดับหนึ่ง, 2 = อันดับสอง

# [6] ช่องที่ไม่ปรากฏในเอกสาร ให้ใส่ null — ห้ามแต่งขึ้นมา

# === ข้อความจากเอกสาร ===
# {document_text}

# === สิ้นสุดข้อความ ===
# ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบายใด ๆ ก่อนหรือหลัง และ/no_think"""

EXTRACT_PROMPT = """ต่อไปนี้คือข้อความที่ถอดจากใบแสดงผลการศึกษา (transcript) ของสถาบันในประเทศไทย
จงแปลงเป็น JSON ตาม schema ที่กำหนดเท่านั้น

=== กติกาสำคัญ (ผิดข้อใดข้อหนึ่งถือว่าใช้งานไม่ได้) ===

[1] ห้ามเดาเกรด
    grade_earn ต้องเป็นหนึ่งใน: A B+ B C+ C D+ D F W S U I P T
    ถ้าช่องใดเบลอ อ่านไม่ออก หรือกำกวม ให้ใส่ null — ห้ามเดาเป็นค่าที่ดูสมเหตุสมผล

[2] อ่านตารางทีละแถว จากบนลงล่าง
    ห้ามข้ามแถว ห้ามรวมสองแถวเป็นแถวเดียว ห้ามสลับลำดับ
    ถ้าหน้ากระดาษแบ่งเป็นสองคอลัมน์ ให้อ่านคอลัมน์ซ้ายให้จบก่อนแล้วค่อยขึ้นคอลัมน์ขวา
    รวมวิชาที่เทียบโอน วิชาที่ถอน (W) และวิชาที่ลงซ้ำ ทุกครั้งที่ปรากฏ

[3] คัดลอกค่าต่อไปนี้ตรงจากบรรทัดที่มี label กำกับในเอกสาร:
    "GPS: <ข้อความ>"            -> ใส่ใน field GPS ของภาคนั้น
    "GPA: <ข้อความ>"            -> ใส่ใน field GPA ของภาคนั้น
    "pass_reason: <ข้อความ>"    -> ใส่ใน field pass_reason ของภาคนั้น
                                   ภาคนั้นไม่มีรายวิชา (subject = []) และ GPS/GPA = null
    "cumulative_gpa: <ข้อความ>"  -> ใส่ใน field cumulative_gpa

[4] รูปแบบข้อมูล
    - credit  : จำนวนเต็ม เช่น 3  (ไม่ใช่ "3(3-0-6)")
    - type    : ถ้าตารางวิชามีคอลัมน์ Type ให้ลอกค่าในคอลัมน์นั้นใส่ field type ตามที่พิมพ์
      ถ้าตารางไม่มีคอลัมน์ Type ให้ใส่ null
    - year    : ตัวเลขปีที่พิมพ์ในหัวของภาคนั้น ลอกตามที่พิมพ์ ห้ามบวกหรือลบ 543
      (เอกสารไทยพิมพ์ พ.ศ. เอกสารอังกฤษพิมพ์ ค.ศ. ให้ลอกตามนั้น โปรแกรมจะแปลงให้เอง)
      ถ้าหัวภาคไม่มีปี ให้ใส่ null ห้ามเดา
    - sem_num : 1, 2, หรือ 3 (3 = ภาคฤดูร้อน)
      ภาษาอังกฤษ: "1st Semester" = 1, "2nd Semester" = 2, "Summer" = 3
    - วันที่ (date_of_birth, admis_date, grad_date, updated_at) : รูปแบบ YYYY-MM-DD
      YYYY = ปีตามที่พิมพ์ในเอกสาร ห้ามบวกหรือลบ 543 (โปรแกรมจะแปลงให้เอง)
      แปลงชื่อเดือนเป็นเลข 2 หลัก เช่น มกราคม/January = 01, กันยายน/September = 09
      ถ้ายังไม่สำเร็จการศึกษา หรือเอกสารระบุ N/A ให้กำหนดค่าดังนี้:
      • grad_date = "0000-00-00"
      • grad_reason : ดูข้อความในช่องวันสำเร็จการศึกษา
        - ถ้ามีแต่ "N/A" ไม่มีข้อความอื่น -> null
        - ถ้ามีเหตุผลต่อท้าย -> ลอกข้อความทั้งช่องรวมคำว่า N/A ตามที่พิมพ์
          (เช่น ช่องที่เขียนว่า N/A แล้วตามด้วยเหตุผลในวงเล็บ ให้เก็บทั้งสองส่วน)
    - แปลงเลขไทย ๐๑๒๓๔๕๖๗๘๙ เป็นเลขอารบิก
    - honor   : 0 ถ้าไม่ได้เกียรตินิยม, 1 = อันดับหนึ่ง, 2 = อันดับสอง

[5] uni_name / uni_address
    บรรทัดแรกสุดของเอกสารไม่มี label กำกับ แต่แยกได้จากคำว่า "เลขที่" ที่ขึ้นต้นที่อยู่เสมอ:
    ข้อความก่อนคำว่า "เลขที่" ตัวแรก = uni_name
    ตั้งแต่คำว่า "เลขที่" เป็นต้นไปจนจบบรรทัด = uni_address

[6] pass_reason
    ใส่ค่าได้ก็ต่อเมื่อเอกสารมีข้อความระบุสถานะผ่าน/ไม่ผ่านของภาคนั้นตรง ๆ เท่านั้น
    ห้ามอนุมานจากเกรดที่เห็น (เช่น เห็นเกรด F เยอะแล้วสรุปเองว่า "ไม่ผ่าน")
    ถ้าเอกสารไม่มีข้อความแบบนี้ระบุไว้ ให้ใส่ null เสมอ

[7] footer_detail.by — 3 บรรทัดสุดท้ายก่อนจบเอกสารมีความหมายต่างกัน แยกตามตำแหน่ง:
    บรรทัดที่อยู่ในวงเล็บ (ชื่อคน)     -> by_signature
    บรรทัดถัดมา (ตำแหน่ง)            -> by_position
    บรรทัดสุดท้าย (หน่วยงาน/สำนัก)    -> by_reg
    ทั้งสามค่าไม่ใช่ field เดียวกัน ห้ามใส่ค่าเดียวกันซ้ำในหลาย field

[8] ช่องที่ไม่ปรากฏในเอกสาร ให้ใส่ null — ห้ามแต่งขึ้นมา

[9] ค้นหาค่าเกรดเฉลี่ยประจำภาคการศึกษา (GPS / GPA ในภาคนั้น) และเกรดเฉลี่ยสะสม (cumulative_gpa / GPAX):
    - สำหรับ GPS / GPA ประจำภาค: ให้มองหาตัวเลขที่เป็นเกรดเฉลี่ยประจำภาคนั้นๆ ซึ่งอาจระบุด้วยคำว่า "GPS", "GPA", "Semester GPA", "คะแนนเฉลี่ยประจำภาคการศึกษา", "เฉลี่ยประจำภาค" หรือตัวเลขสรุปท้ายตารางของภาคนั้น
    - สำหรับ cumulative_gpa: ให้มองหาเกรดเฉลี่ยสะสมรวมของทั้งเอกสาร ซึ่งอาจระบุด้วยคำว่า "cumulative_gpa", "Cumulative GPA", "GPAX", "คะแนนเฉลี่ยสะสม", "คะแนนเฉลี่ย" หรือเกรดสะสมที่อยู่ท้ายภาคการศึกษาสุดท้าย

[10] การสกัดค่า GPS, GPA และ cumulative_gpa (ต้องสกัดให้ครบทุกภาคการศึกษา):
    ก) field "GPS" (เกรดเฉลี่ยประจำภาค):
       - รูปแบบปริญญาตรี: ลอกจากบรรทัด "GPS: <ตัวเลข>" หรือ "GPS:<ตัวเลข>" (เช่น "GPS: 3.83" ให้ดึง "3.83")
       - รูปแบบบัณฑิตศึกษา / ข้อความยาว: ลอกจาก "คะแนนเฉลี่ยประจำภาคการศึกษา : <ตัวเลข>" หรือ "Semester GPA: <ตัวเลข>"
    ข) field "GPA" (เกรดเฉลี่ยสะสมประจำภาค):
       - รูปแบบปริญญาตรี: ลอกจากบรรทัด "GPA: <ตัวเลข>" หรือ "GPA:<ตัวเลข>" (เช่น "GPA: 3.83" ให้ดึง "3.83")
       - รูปแบบบัณฑิตศึกษา / ข้อความยาว: ลอกจาก "คะแนนเฉลี่ย : <ตัวเลข>" หรือ "Cumulative GPA: <ตัวเลข>"
    ค) field "cumulative_gpa" (เกรดเฉลี่ยสะสมรวมทั้งเอกสาร):
       - ลอกจากบรรทัด "cumulative_gpa: <ตัวเลข>", "Cumulative GPA: <ตัวเลข>" หรือ "คะแนนเฉลี่ยสะสม" ท้ายเอกสาร
    ง) กฎเพิ่มเติม:
       - หากภาคใดระบุตัวเลขเป็น "0.00" หรือ "0,00" ให้ใส่ "0.00" ห้ามใส่ null
       - หากพิมพ์เป็นเครื่องหมาย "-" ให้ใส่ "-"
       - ห้ามคำนวณเองเด็ดขาด ให้ลอกตัวเลขตามที่ปรากฏในเอกสารเท่านั้น

[11] การอ่านตารางและป้ายกำกับ
    - อ่านตารางทีละแถว จากบนลงล่าง หากพบคำว่า "[ดิบ]" อยู่หน้าบรรทัด ให้ข้ามคำว่า "[ดิบ]" แล้วอ่านข้อความข้างในตามปกติ
    - รวมวิชาที่เทียบโอน วิชาที่ถอน (W) และวิชาที่ลงซ้ำ ทุกครั้งที่ปรากฏ


=== ข้อความจากเอกสาร ===
{document_text}

=== สิ้นสุดข้อความ ===
ตอบเป็น JSON เท่านั้น ห้ามมีคำอธิบายใด ๆ ก่อนหรือหลัง และ"""


# Typhoon-OCR เป็นโมเดลเฉพาะทาง ใช้ได้เฉพาะกับ prompt ของมันเอง
# ถ้าเปลี่ยน prompt ผลจะแย่ลงมาก — นี่คือลักษณะของ task-specific model
# TYPHOON_PROMPT = ("Below is an image of a document page. "
#                   "Extract all text content and structure into markdown format. "
#                   "Preserve tables using markdown table syntax.")
# TYPHOON_PROMPT = (
#     "Transcribe the entire text from this Thai transcript image into clean Markdown. "
#     "1. Keep the exact section headers (e.g., student details, semester blocks, signatures). "
#     "2. Group courses by academic year and semester clearly. "
#     "3. Represent all course grades and GPA/GPS summaries as explicit Markdown tables. "
#     "4. Do not summarize or skip any course or number."
# )

TYPHOON_PROMPT = (
    "Transcribe the entire text from this Thai transcript image into clean Markdown format.\n"
    "Rules:\n"
    "1. Keep all exact text, labels, and section headers as printed without summarizing or omitting anything.\n"
    "2. Group courses strictly by academic year and semester blocks.\n"
    "3. Use standard Markdown pipe tables (| ... |) for courses. DO NOT use HTML table tags (<table>, <tr>, <td>).\n"
    "4. In the course tables, strictly separate [Course Code], [Course Name], [Credit], and [Grade] into distinct individual columns: | Code | Name | Credit | Grade |.\n"
    "5. Keep GPA, GPS, and total credits clearly separated below each corresponding semester table, "
    "each on its own line using the exact format: \"GPS: <value>\" and \"GPA: <value>\" "
    "(use \"GPS\" for that semester's average, \"GPA\" for cumulative average, matching the labels printed in the document).\n"
    "6. Do not duplicate grade characters or merge adjacent text cells."
)


# ==============================================================================
#  ส่วนที่ 5 — เรียกใช้ Ollama
# ==============================================================================


def ollama_chat(model: str, messages: list[dict], *, fmt: dict | None = None,
                images: list[bytes] | None = None, temperature: float = 0.0,
                retries: int = 2, think: bool | None = None) -> str:
    """
    เรียก Ollama ผ่าน HTTP API ที่ localhost

    Parameters
    ----------
    fmt    : JSON Schema สำหรับ structured output (ส่ง None ถ้าอยากได้ข้อความธรรมดา)
    images : list ของ PNG bytes — จะถูกแปลงเป็น base64 แนบใน message สุดท้าย

    ทำไม temperature = 0?
      temperature ควบคุมความสุ่มในการเลือก token ถัดไป
      งานถอดเอกสารต้องการ "ผลเดิมทุกครั้งที่รัน" (reproducibility)
      ถ้าตั้งสูงกว่า 0 รันสองครั้งจะได้ผลไม่เหมือนกัน --> เปรียบเทียบโมเดลไม่ได้
    """
    requests = _need("requests")

    if images:
        # Ollama รับภาพผ่าน field "images" เป็น list ของ base64 string
        messages = [dict(m) for m in messages]
        messages[-1]["images"] = [base64.b64encode(im).decode() for im in images]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 16384,        # context window; ตารางยาว ๆ ต้องการเยอะ
            "num_predict": 8192,     # จำกัดความยาว output กัน loop ไม่รู้จบ
        },
    }
    if fmt is not None:
        payload["format"] = fmt
    if think is not None:
        payload['think'] = think
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload,
                              timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            print(f"      ({model} ใช้เวลา {time.time() - t0:.1f} วิ, "
                  f"ได้ {len(content):,} ตัวอักษร)")
            if not content.strip():
                raise ValueError("โมเดลตอบกลับมาว่าง")
            return content
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"      ⚠ ลองใหม่ครั้งที่ {attempt + 1}: {e}")
                time.sleep(3)
    raise RuntimeError(f"เรียก {model} ไม่สำเร็จ: {last_err}")


def parse_json(text: str) -> dict:
    """
    ดึง JSON ออกจากข้อความที่โมเดลตอบมา

    ถึงจะใช้ structured output แล้ว ก็ยังควรมีตัวนี้ไว้ เพราะ:
      - บางโมเดล (โดยเฉพาะพวก reasoning) แทรก <think>...</think> มาข้างหน้า
      - บางครั้งห่อด้วย markdown code fence
    ใช้ raw_decode แทนการหา '}' ตัวสุดท้าย เพราะมันตัดพอดีวงเล็บที่สมดุลกัน
    """
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)        # ตัด reasoning ทิ้ง
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.MULTILINE)

    starts = [p for p in (t.find("{"), t.find("[")) if p != -1]
    if not starts:
        raise ValueError(f"ไม่พบ JSON ในคำตอบ:\n{text[:400]}")
    obj, _ = json.JSONDecoder().raw_decode(t[min(starts):])
    return obj


# ==============================================================================
#  ส่วนที่ 6 — PIPELINE A : Tesseract (baseline จาก Lab 5-6)
# ==============================================================================


def pipeline_baseline(pages: list[bytes]) -> dict:
    """
    เส้นทางดั้งเดิม: ปรับภาพด้วย OpenCV -> Tesseract -> แกะด้วย regex

    เราใส่ pipeline นี้ไว้เพื่อเป็น "เส้นฐาน" (baseline)
    ถ้าไม่มีเส้นฐาน เราจะบอกไม่ได้ว่า LLM ดีขึ้นแค่ไหน หรือดีขึ้นจริงหรือเปล่า
    การมี baseline ที่อ่อนแอเกินไปก็ทำให้ผลดูดีเกินจริง — ต้องระวัง
    """
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        import cv2
    except ImportError as e:
        print(f"  ⚠ ข้าม baseline: {e}")
        return {}

    full_text = ""
    for i, png in enumerate(pages):
        img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))

        # --- ปรับภาพก่อนส่งเข้า OCR (ทบทวนจาก Lab 5) ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Otsu หาค่าขีดแบ่งอัตโนมัติจาก histogram — เหมาะกับเอกสารที่แสงสม่ำเสมอ
        # ถ้าภาพถ่ายมีเงา ให้เปลี่ยนไปใช้ adaptiveThreshold แทน
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- OCR ---
        # --psm 6 = สมมติว่าเป็นบล็อกข้อความเดียวที่จัดเรียงสม่ำเสมอ
        #           เหมาะกับตาราง มากกว่า psm 3 (auto) ที่มักตัดคอลัมน์ผิด
        # lang="tha+eng" = ต้องติดตั้ง tha.traineddata ด้วย
        txt = pytesseract.image_to_string(bw, lang="tha+eng", config="--psm 6")
        full_text += f"\n=== หน้า {i + 1} ===\n{txt}"
        print(f"      Tesseract หน้า {i + 1}: {len(txt):,} ตัวอักษร")

    return _rule_based_parse(full_text)


def _rule_based_parse(text: str) -> dict:
    """
    แกะข้อมูลจากข้อความดิบด้วย regex

    ⚠️ อ่านโค้ดส่วนนี้ให้ดี แล้วถามตัวเองว่า:
       "ถ้าเอกสารเปลี่ยนรูปแบบนิดเดียว โค้ดนี้จะพังไหม"
       คำตอบคือ "พัง" — และนั่นคือเหตุผลที่เราหันมาใช้ LLM
       LLM ไม่ต้องการ regex ใหม่ทุกครั้งที่ฟอร์มเปลี่ยน

       แต่ข้อดีของ regex คือ "ทำนายผลได้" — ผิดแบบเดิมทุกครั้ง
       ในขณะที่ LLM อาจผิดคนละแบบในแต่ละครั้ง ซึ่งดีบักยากกว่ามาก
    """
    out: dict = {
        "header_detail": {},
        "transcript_detail": {"semesters": []},
        "footer_detail": {},
    }

    # --- รหัสนักศึกษา: ตัวเลข 8 หลักที่อยู่หลังคำว่า "รหัส" ---
    m = re.search(r"รหัส\D{0,20}(\d{8})", text)
    if m:
        out["header_detail"]["student_id"] = m.group(1)

    # --- คำนำหน้า + ชื่อ ---
    m = re.search(r"(นางสาว|นาย|นาง)\s*([^\n]{2,60})", text)
    if m:
        out["header_detail"]["prename"] = m.group(1)
        out["header_detail"]["name"] = m.group(2).strip()

    m = re.search(r"(คณะ[^\n]{2,40})", text)
    if m:
        out["header_detail"]["faculty_name"] = m.group(1).strip()

    # --- แถววิชา ---
    # รูปแบบที่คาดหวัง:  <รหัส 8 หลัก> <ชื่อวิชา> <หน่วยกิต> <เกรด>
    # regex นี้ "เปราะบางมาก": ถ้าชื่อวิชามีตัวเลขอยู่ข้างใน หรือ
    # Tesseract อ่านช่องว่างหาย ก็จะจับไม่ได้ทันที
    row_re = re.compile(
        r"(\d{8})\s+(.{3,80}?)\s+(\d)\s+([ABCDF][+]?|[WSUIPT])\b",
        re.IGNORECASE,
    )

    # --- หัวข้อภาคการศึกษา ---
    sem_re = re.compile(r"(?:ภาค|ภาคเรียน|semester)\D{0,12}([123])\D{0,20}(25\d{2})")

    current = {"year": None, "sem_num": None, "GPA": None, "GPS": None,
               "pass_reason": None, "subject": []}

    for line in text.splitlines():
        sm = sem_re.search(line)
        if sm:
            if current["subject"]:
                out["transcript_detail"]["semesters"].append(current)
            current = {"year": int(sm.group(2)), "sem_num": int(sm.group(1)),
                       "GPA": None, "GPS": None, "pass_reason": None, "subject": []}
            continue

        for rm in row_re.finditer(line):
            current["subject"].append({
                "subject_id": rm.group(1),
                "subject_name": rm.group(2).strip(),
                "type": None,
                "credit": int(rm.group(3)),
                "grade_earn": rm.group(4).lower(),
            })

    if current["subject"]:
        out["transcript_detail"]["semesters"].append(current)

    n = sum(len(s["subject"]) for s in out["transcript_detail"]["semesters"])
    print(f"      regex แกะได้ {len(out['transcript_detail']['semesters'])} ภาค, "
          f"{n} วิชา")
    return out


# ==============================================================================
#  ส่วนที่ 7 — PIPELINE B : Typhoon-OCR -> text LLM
# ==============================================================================

GRADE_SET = {"A", "B+", "B", "C+", "C", "D+", "D", "F", "W", "S", "U", "I", "P", "T"}

def _extract_cells(row_html: str) -> list[str]:
    """ดึงข้อความในแต่ละ <td>/<th> ของ 1 แถว (<tr>...</tr>)
    ทนต่อ tag ที่ผิดปกติ เช่น <th> ที่ไม่ปิด, colspan/rowspan attribute"""
    cells = re.findall(r'<t[dh][^>]*>(.*?)(?=<t[dh]|</tr>|$)', row_html, flags=re.S)
    return [re.sub(r'<[^>]+>', '', c).strip() for c in cells]


CODE_START = re.compile(r'^\d{6,}')
SEM_SUMMARY_RE = r'(?:คะแนนเฉลี่ยประจำภาคการศึกษา|GPS|Semester GPA)\s*:\s*([\d.,N/A-]+).*?(?:คะแนนเฉลี่ย|GPA|Cumulative GPA)\s*:\s*([\d.,N/A-]+)'
TITLE_RE = r'ภาคการศึกษาที่\s*(\d)\s*ปีการศึกษา\s*(\d{4})'

# ท้ายชื่อวิชาที่มี [ประเภท] หน่วยกิต เกรด ติดมา เช่น "SEMINAR 1 Nc 1 S", "แคลคูลัส 1 3 D+"
# (Typhoon บางครั้งรวมทุกคอลัมน์ไว้ใน cell ชื่อ แล้วปล่อย cell หน่วยกิต/เกรดว่าง)
TRAILING_RE = re.compile(
    r'^(?P<name>.*?\S)\s+(?:(?P<type>[A-Za-z]{1,3}(?:\s+[A-Za-z]{1,3})*)\s+)?'
    r'(?P<credit>\d{1,2})\s+(?P<grade>[A-Za-z]\+?)$')


def _is_grade(text: str) -> bool:
    return text.strip().upper() in GRADE_SET


# ภาค "รักษาสภาพ" / "Maintain" — ภาคที่ไม่ได้ลงเรียน แค่รักษาสถานภาพนักศึกษาไว้
# เฉลยเก็บภาคแบบนี้เป็น pass_reason = "maintain" โดยไม่มีรายวิชา และ GPS/GPA เป็น null
MAINTAIN_RE = re.compile(r'^\s*(?:\*{0,2})\s*(?:รักษาสภาพ|maintain)\s*(?:\*{0,2})\s*$', re.I)
MAINTAIN_LINE = "pass_reason: maintain"


def _is_maintain(text: str) -> bool:
    return bool(MAINTAIN_RE.match(text or ""))


def _first_type(text: str) -> str:
    """
    ประเภทวิชาต้องมีค่าเดียว (เช่น Cr) — ถ้า OCR อ่านมาหลายตัว ("Cr Nc") เก็บตัวแรก

    เอกสารมีคอลัมน์ประเภทช่องเดียวต่อวิชา ค่าที่เกินมาจึงเป็นของคอลัมน์อื่นที่หลุดมา
    """
    return text.split()[0] if text.strip() else ""


def _is_type(text: str) -> bool:
    """ประเภทวิชา เช่น Cr / Nc — ตัวอักษรล้วน 1-3 ตัว ที่ไม่ใช่เกรด"""
    t = text.strip()
    return bool(re.fullmatch(r'[A-Za-z]{1,3}', t)) and not _is_grade(t)


def _split_trailing(name: str) -> tuple[str, str, str, str] | None:
    """แยก (ชื่อ, ประเภท, หน่วยกิต, เกรด) ออกจากท้ายชื่อวิชา — None ถ้าท้ายชื่อไม่ใช่รูปนี้"""
    m = TRAILING_RE.match(name.strip())
    if not m or not _is_grade(m.group("grade")):
        return None
    return m.group("name"), m.group("type") or "", m.group("credit"), m.group("grade")


def _extract_cells_span(row_html: str) -> list[tuple[str, int]]:
    """เหมือน _extract_cells แต่คืน (ข้อความ, คอลัมน์เริ่มต้น) โดยนับ colspan ด้วย"""
    out, col = [], 0
    for attrs, body in re.findall(r'<t[dh]([^>]*)>(.*?)(?=<t[dh]|</tr>|$)', row_html, flags=re.S):
        span_m = re.search(r'colspan\s*=\s*"?(\d+)', attrs)
        out.append((re.sub(r'<[^>]+>', '', body).strip(), col))
        col += int(span_m.group(1)) if span_m else 1
    return out


def _reading_order(rows: list[str]) -> list[list[str]]:
    """
    คืน cell ของทุกแถวตาม "ลำดับการอ่าน"

    transcript หลายแบบจัดหน้าเป็นสองคอลัมน์ (ภาคหนึ่งอยู่ซ้าย อีกภาคอยู่ขวา)
    Typhoon จึงพ่นวิชาสองภาคมาในแถว <tr> เดียวกัน ถ้าอ่านทีละแถว วิชาของสองภาค
    จะปนกัน — ฟังก์ชันนี้แยกซ้าย/ขวา แล้วเรียง "คอลัมน์ซ้ายทั้งหมด แล้วค่อยคอลัมน์ขวา"

    หาเส้นแบ่งจากแถวที่มีรหัสวิชาสองตัว (คอลัมน์ที่รหัสตัวที่สองเริ่ม นับ colspan)
    แยกเฉพาะแถวที่ส่วนขวามีความหมายในตัวเอง (มีรหัสวิชา/ชื่อภาค/บรรทัดสรุปภาค)
    ส่วนแถวอื่น เช่น "หน่วยกิตรวม : | 36" ที่ค่าอยู่ฝั่งขวาของ label จะไม่ถูกผ่า
    ถ้าตารางไม่ใช่สองคอลัมน์ คืนลำดับเดิมทุกแถว
    """
    spans = [_extract_cells_span(r) for r in rows]
    splits = []
    for cells in spans:
        starts = [col for text, col in cells if CODE_START.match(text)]
        if len(starts) >= 2:
            splits.append(starts[1])
    if not splits:
        return [[t for t, _ in cells] for cells in spans]
    split = max(set(splits), key=splits.count)

    left, right = [], []
    for cells in spans:
        lpart = [t for t, col in cells if col < split]
        rpart = [t for t, col in cells if col >= split]
        rtext = " ".join(rpart)
        meaningful = (any(CODE_START.match(t) for t in rpart)
                      or re.search(TITLE_RE, rtext) or re.search(SEM_SUMMARY_RE, rtext))
        if meaningful:
            if any(lpart):
                left.append(lpart)
            right.append(rpart)
        else:
            left.append([t for t, _ in cells])
    return left + right


RAW_ROW_TAG = "[ดิบ]"


def _raw_row(cells: list[str]) -> str:
    """แถวที่ parse ไม่ได้ — ส่งข้อความเดิมต่อให้ LLM ขั้น 2 อ่านเอง พร้อมป้ายกำกับ"""
    return f"{RAW_ROW_TAG} " + " | ".join(c for c in cells if c)


def _has_leftover(text: str, consumed: list[str]) -> bool:
    """หลังตัดส่วนที่ regex ใช้ไปแล้ว ยังเหลือตัวอักษร/ตัวเลขอยู่ไหม (ถ้าเหลือ = มีข้อมูลที่จะหาย)"""
    for c in consumed:
        text = text.replace(c, " ", 1)
    return bool(re.search(r'[0-9A-Za-z\u0E00-\u0E7F]', text))


def normalize_typhoon_table(raw_markdown: str) -> str:
    """
    Typhoon-OCR มักพ่นตารางวิชาออกมาเป็น <table> ก้อนเดียวยาว ไม่มี newline,
    โครงสร้าง tag ไม่สมบูรณ์ (colspan/rowspan, <th> ไม่ปิด) โดยไม่สนใจคำสั่งใน
    TYPHOON_PROMPT เลย (ยืนยันแล้วหลายรอบว่าสั่งไม่ได้จริง)

    ฟังก์ชันนี้ parse ก้อน HTML นั้นด้วย regex อย่าง deterministic แทนที่จะพึ่งให้
    LLM ขั้น 2 (qwen3) ต้องมาตีความโครงสร้างเอง แล้วคืนข้อความที่จัดรูปแบบชัดเจน
    แยกตามภาคการศึกษา พร้อม label "GPS:"/"GPA:" ตรงตัว — ตัดความกำกวมเรื่อง
    "ตัวเลขนี้ผูกกับภาคไหน" ที่เป็นสาเหตุให้ GPA/GPS สกัดไม่ได้มาหลายรอบ

    ข้อความนอก <table>...</table> (ส่วน header/footer ที่เป็นร้อยแก้ว) จะถูก
    คงไว้เหมือนเดิมทุกตัวอักษร ไม่แตะ — เพราะส่วนนั้น extract ได้แม่นอยู่แล้ว

    Safety net: แถวในตารางที่ไม่เข้ารูปแบบที่รู้จัก (ภาษาอื่น, layout อื่น,
    รหัสแยก cell ฯลฯ) จะไม่ถูกทิ้งเงียบ ๆ แต่ส่งต่อเป็น "[ดิบ] <ข้อความเดิม>"
    ตามตำแหน่งเดิม ให้ LLM ขั้น 2 อ่านเอง — parse ได้ก็จัดให้ parse ไม่ได้ก็ไม่ทำหาย
    """
    table_match = re.search(r'<table>.*?</table>', raw_markdown, flags=re.S)
    if table_match is None:
        return raw_markdown  # ไม่เจอตาราง — ไม่มีอะไรต้อง normalize

    table_html = table_match.group(0)
    rows = re.findall(r'<tr>.*?</tr>', table_html, flags=re.S)
    ordered = _reading_order(rows)   # ตารางสองคอลัมน์ -> ซ้ายทั้งหมดก่อน แล้วค่อยขวา

    # ดึง (เลขภาค, ปีการศึกษา) ทุกคู่ ตามลำดับการอ่าน (ไม่ใช่ลำดับ HTML)
    title_re = TITLE_RE
    titles = re.findall(title_re, "\n".join(" ".join(c) for c in ordered))
    # ขอบเขตของภาค: ถ้าตารางมีแถวชื่อภาค ให้ "แถวชื่อภาค" เป็นตัวเริ่มภาคใหม่
    # (แม่นกว่าบรรทัดสรุปภาค เพราะ OCR ของหน้าสองคอลัมน์มักวางบรรทัดสรุปภาค
    #  ก่อนวิชาสุดท้ายของภาคนั้น) ถ้าไม่มีชื่อภาคเลย ค่อยใช้บรรทัดสรุปภาคแบ่งแทน
    split_on_titles = bool(titles)

    # บล็อกของแต่ละภาค — เก็บ (ชนิด, ข้อความ) ตามลำดับที่เจอ
    #   "course" = แถววิชาที่ parse ได้   "raw" = แถวที่ parse ไม่ได้ (ส่งต่อแบบดิบ)
    #   "title"/"gps" = ชื่อภาค/บรรทัดสรุปภาค ณ ตำแหน่งเดิม (ใช้เมื่อบล็อกไม่มีแถววิชาที่ parse ได้)
    semesters: list[list[tuple[str, str]]] = [[]]
    gps_gpa_lines: list[str] = []
    trailing_lines: list[str] = []

    for cells in ordered:
        if not any(cells):
            continue
        joined = " ".join(cells)

        # 1) บรรทัดสรุปท้ายภาค -> ตัวคั่นภาค + บรรทัด GPS/GPA ระบุชัด
        #    ข้ามถ้าแถวนี้ขึ้นต้นด้วยรหัสวิชา เพราะเป็น "แถววิชาที่มีบรรทัดสรุปติดมา"
        #    (บรรทัดสรุปจริงไม่เคยขึ้นต้นด้วยรหัสวิชา) ให้ข้อ 6 จัดการแยกวิชา/สรุปออกจากกัน
        m = None if CODE_START.match(cells[0]) else re.search(SEM_SUMMARY_RE, joined)
        if m:
            gps, gpa = m.group(1), m.group(2)
            gps_gpa_lines.append(f"GPS: {gps}\nGPA: {gpa}")
            semesters[-1].append(("gps", gps_gpa_lines[-1]))
            if _has_leftover(joined, [m.group(0)]):
                semesters[-1].append(("raw", _raw_row(cells)))
            if not split_on_titles:
                semesters.append([])  # เริ่มเก็บวิชาของภาคถัดไป
            continue

        # 2) เกรดเฉลี่ยสะสม (ระดับทั้งเอกสาร ไม่ใช่ระดับภาค)
        m = re.search(r'คะแนนเฉลี่ยสะสม\s*:\s*([\d.]+)', joined)
        if m:
            trailing_lines.append(f"cumulative_gpa: {m.group(1)}")
            if _has_leftover(joined, [m.group(0)]):
                trailing_lines.append(_raw_row(cells))
            continue

        # 3) หน่วยกิตรวม
        m = re.search(r'จำนวนหน่วยกิตที่สอบได้ทั้งหมด\s*:\s*(\d+)', joined)
        if m:
            trailing_lines.append(f"total_credits_earned: {m.group(1)}")
            if _has_leftover(joined, [m.group(0)]):
                trailing_lines.append(_raw_row(cells))
            continue

        # 4) บรรทัดปิดท้าย — ข้าม
        if "สิ้นสุดการแสดงผลการศึกษา" in joined:
            continue

        # 5) แถวที่ไม่ได้ขึ้นต้นด้วยรหัสวิชา (หัวตาราง/ชื่อภาค/ข้อความอื่น)
        #    ชื่อภาคที่จับด้วย title_re ไปแล้วไม่ต้องส่งซ้ำ แต่ถ้ามีข้อความอื่นปน
        #    หรือเป็นรูปแบบที่ไม่รู้จัก ให้ส่งต่อแบบดิบ แทนการทิ้งเงียบ ๆ
        if not re.match(r'^\d{6,}', cells[0]):
            # แถว "รักษาสภาพ"/"Maintain" (เซลล์อื่นในแถวเป็นเศษคอลัมน์ที่ OCR พามา)
            if any(_is_maintain(c) for c in cells):
                # แต่ละแถวรักษาสภาพ = หนึ่งภาค จึงเริ่มบล็อกใหม่ทั้งเมื่อบล็อกเดิม
                # มีวิชา และเมื่อบล็อกเดิมเป็นภาครักษาสภาพอยู่แล้ว
                if any(k in ("course", "pass") for k, _ in semesters[-1]):
                    semesters.append([])
                semesters[-1].append(("pass", MAINTAIN_LINE))
                continue
            consumed = [t.group(0) for t in re.finditer(title_re, joined)]
            if _has_leftover(joined, consumed):
                # อะไรก็ตามที่มาหลังภาครักษาสภาพ = เนื้อหาของภาคถัดไป
                # (เช่นหัวภาคภาษาอังกฤษที่เรายังไม่ได้ parse เป็น title)
                if any(k == "pass" for k, _ in semesters[-1]):
                    semesters.append([])
                semesters[-1].append(("raw", _raw_row(cells)))
            elif consumed:
                if split_on_titles and any(k == "course" for k, _ in semesters[-1]):
                    semesters.append([])  # แถวชื่อภาค = เริ่มภาคใหม่
                semesters[-1].append(("title", _raw_row(cells)))
            continue

        # 6) แถววิชา — รองรับทั้ง "รหัส ชื่อ" ใน cell เดียว และ "รหัส" | "ชื่อ" แยก cell
        code_m = re.match(r'^(\d{6,})\s+(.*)$', cells[0])
        if code_m:
            code, name, rest = code_m.group(1), code_m.group(2).strip(), cells[1:]
        elif re.fullmatch(r'\d{6,}', cells[0]) and len(cells) > 1 and cells[1]:
            code, name, rest = cells[0], cells[1], cells[2:]
        else:
            # รูปแถวที่ไม่คาดไว้ — ส่งต่อแบบดิบ
            semesters[-1].append(("raw", _raw_row(cells)))
            continue

        # บรรทัดสรุปภาคที่ OCR ยัดมาในเซลล์ชื่อวิชา (เช่น
        # "CHEMISTRY LABORATORY GPS : 0.00 GPA : -") — ดึงออกมาเป็นบรรทัดของภาค
        # ไม่งั้นทั้งวิชาและ GPS/GPA จะเพี้ยนไปด้วยกัน
        inline_gps = re.search(SEM_SUMMARY_RE, name)
        if inline_gps:
            name = name[:inline_gps.start()].strip()

        credit = next((c for c in rest if c.isdigit()), "")
        grade = next((c for c in rest if _is_grade(c)), "")
        ctype = _first_type(next((c for c in rest if _is_type(c)), ""))

        # ท้ายชื่อมี [ประเภท] หน่วยกิต เกรด ติดมา: ใช้เติมช่องที่ว่าง
        # หรือตัดทิ้งถ้าเป็นค่าซ้ำกับคอลัมน์ที่แยกมาแล้ว
        tail = _split_trailing(name)
        if tail:
            t_name, t_type, t_credit, t_grade = tail
            # ข้อมูลท้ายชื่อคือคอลัมน์ที่ OCR ยุบเข้ามา — ใช้เติม "เฉพาะช่องที่ว่าง"
            # ส่วนช่องที่แยกมาเป็นเซลล์จริงแล้วเชื่อถือมากกว่า จึงไม่ทับ
            # (เช่น "THESIS Cr Nc 9 I" ที่มีเซลล์เกรด S อยู่แล้ว -> ได้ชื่อ THESIS,
            #  ประเภท Cr, หน่วยกิต 9 และคงเกรด S ไว้)
            name = t_name
            credit = credit or t_credit
            grade = grade or t_grade
            ctype = ctype or _first_type(t_type)

        # ภาครักษาสภาพถูกพิมพ์เป็นแถววิชา (มีรหัสด้วย) — เฉลยไม่นับเป็นรายวิชา
        if _is_maintain(name):
            if any(k == "course" for k, _ in semesters[-1]):
                semesters.append([])       # ภาครักษาสภาพเป็นคนละภาคกับภาคที่มีวิชา
            if not any(k == "pass" for k, _ in semesters[-1]):
                semesters[-1].append(("pass", MAINTAIN_LINE))
            continue

        if any(k == "pass" for k, _ in semesters[-1]):
            semesters.append([])   # มีวิชาต่อจากภาครักษาสภาพ = ขึ้นภาคใหม่

        # (1) แถวซ้ำ: OCR พ่นวิชาเดิมซ้ำในภาคเดียวกัน — เก็บแถวแรกพอ
        #     (วิชาเดียวกันที่ลงซ้ำคนละภาคยังเก็บครบ เพราะเทียบเฉพาะในบล็อกเดียวกัน)
        entry = (code, name, ctype, credit, grade)
        last_course = next((e for k, e in reversed(semesters[-1]) if k == "course"), None)
        if entry != last_course:
            semesters[-1].append(("course", entry))

        if inline_gps:
            gps_gpa_lines.append(f"GPS: {inline_gps.group(1)}\nGPA: {inline_gps.group(2)}")
            semesters[-1].append(("gps", gps_gpa_lines[-1]))
            if not split_on_titles:
                semesters.append([])   # บรรทัดสรุปภาค = จบภาคนั้น

        # เซลล์ที่เหลือมีรหัสวิชาอีกตัว = มีวิชาอื่นอยู่ในแถวเดียวกัน (เช่น ตาราง
        # สองคอลัมน์) — ส่งส่วนนั้นต่อแบบดิบ แทนการทิ้งไปพร้อมคอลัมน์ noise
        extra = next((i for i, c in enumerate(cells[1:], 1) if re.match(r'^\d{6,}', c)), None)
        if extra is not None:
            semesters[-1].append(("raw", _raw_row(cells[extra:])))

    # ประกอบข้อความสะอาด จับคู่บล็อกวิชากับ GPS/GPA ที่ตามมาตามลำดับในเอกสาร
    # (นับเฉพาะบล็อกที่มีแถววิชาที่ parse ได้ ให้การจับคู่ title เหมือนเดิม
    #  บล็อกที่มีแต่แถวดิบจะพิมพ์ทุกอย่างตามตำแหน่งเดิม รวมชื่อภาคและ GPS/GPA
    #  เพื่อให้ LLM ยังเห็นขอบเขตของภาค และไม่กิน title ของบล็อกที่ parse ได้)
    # คอลัมน์ Type ใส่เฉพาะเอกสารที่มีประเภทวิชา (เช่น ใบบัณฑิตศึกษา Cr/Nc)
    has_type = any(kind == "course" and entry[2]
                   for block in semesters for kind, entry in block)

    def fmt(kind, entry):
        if kind != "course":
            return entry
        code, name, ctype, credit, grade = entry
        return (f"{code} | {name} | {ctype} | {credit} | {grade}" if has_type
                else f"{code} | {name} | {credit} | {grade}")

    # ชื่อภาคและ GPS/GPA ของแต่ละบล็อก: ใช้ตัวที่ "อยู่ในบล็อกนั้นจริง" ก่อน
    # ถ้าบล็อกไม่มี ค่อยใช้ตัวถัดไปตามลำดับ (การจับคู่ตามลำดับอย่างเดียวจะเพี้ยน
    # ทันทีที่ OCR เรียงแถวผิด เช่น วิชาสุดท้ายของภาคหลุดไปอยู่หลังบรรทัดสรุปภาค)
    out = []
    i = 0          # นับบล็อกที่มีวิชา (ใช้ตั้งชื่อ "บล็อกที่ n" เมื่อหาชื่อภาคไม่ได้)
    t_next = 0     # ตำแหน่งชื่อภาคถัดไปใน titles ที่ยังไม่ถูกใช้
    g_next = 0     # ตำแหน่งบรรทัด GPS/GPA ถัดไปที่ยังไม่ถูกใช้
    for block in semesters:
        has_course = any(kind == "course" for kind, _ in block)
        has_pass = any(kind == "pass" for kind, _ in block)
        if not has_course and not has_pass:
            out.extend(text for _, text in block)
            continue

        own_titles = [m.groups() for kind, entry in block if kind == "title"
                      for m in [re.search(title_re, entry)] if m]
        if own_titles and own_titles[0] in titles[t_next:]:
            title = own_titles[0]
            t_next = titles.index(title, t_next) + 1
        elif t_next < len(titles):
            title = titles[t_next]
            t_next += 1
        else:
            title = None
        if title:
            out.append(f"[ภาคการศึกษาที่ {title[0]} ปีการศึกษา {title[1]}]")
        else:
            out.append(f"[ภาคการศึกษา บล็อกที่ {i + 1}]")

        if has_course:
            out.append("Code | Name | Type | Credit | Grade" if has_type
                       else "Code | Name | Credit | Grade")
            out.extend(fmt(kind, entry) for kind, entry in block if kind in ("course", "raw"))
        else:
            out.extend(text for kind, text in block if kind == "raw")
        if has_pass:
            out.append(MAINTAIN_LINE)

        own_gps = next((entry for kind, entry in block if kind == "gps"), None)
        # ภาครักษาสภาพไม่มีเกรดเฉลี่ยของตัวเอง (เฉลยเป็น null) — ใช้คิวของ GPS ต่อไป
        # ตามลำดับ แต่ไม่พิมพ์ออกมา เพื่อไม่ให้ภาคถัดไปหยิบบรรทัดผิด
        if own_gps is not None and own_gps in gps_gpa_lines[g_next:]:
            if has_course:
                out.append(own_gps)
            g_next = gps_gpa_lines.index(own_gps, g_next) + 1
        elif g_next < len(gps_gpa_lines):
            if has_course:
                out.append(gps_gpa_lines[g_next])
                g_next += 1
        out.append("")
        i += 1

    out.extend(trailing_lines)

    clean_table_text = "\n".join(out)
    return raw_markdown[:table_match.start()] + clean_table_text + raw_markdown[table_match.end():]

# ในไฟล์ src/ocr_system/lab7a_transcript.py

def structure_markdown(markdown_text: Union[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(markdown_text, dict):
        markdown_text = (
            markdown_text.get("markdown")
            or markdown_text.get("text")
            or str(markdown_text)
        )

    if not markdown_text or not isinstance(markdown_text, str) or not markdown_text.strip():
        return {
            "header_detail": {},
            "transcript_detail": {"semesters": []},
            "footer_detail": {},
        }

    # Print ดูข้อความ Markdown ใน Terminal เพื่อ Debug
    print("=== DEBUG: MARKDOWN INPUT ===")
    print(markdown_text[:500])  # แสดง 500 ตัวอักษรแรก
    print("=============================")

    prompt = f"""คุณเป็นระบบสกัดข้อมูลจากใบแสดงผลการเรียน (Transcript) 
จงอ่านข้อความ Markdown ด้านล่างนี้ แล้วแปลงให้อยู่ในรูปแบบ JSON Object ให้ถูกต้องที่สุด 

**ข้อกำหนด:**
1. คืนค่าเฉพาะ JSON Object เท่านั้น ห้ามใส่คำอธิบายเพิ่มเติม
2. โครงสร้าง JSON ต้องประกอบด้วย 3 Key หลัก: "header_detail", "transcript_detail", และ "footer_detail"

**ตัวอย่างรูปแบบ JSON:**
{{
  "header_detail": {{
    "student_id": "65010001",
    "name": "นายสมชาย ใจดี"
  }},
  "transcript_detail": {{
    "semesters": [
      {{
        "year": 2565,
        "sem_num": 1,
        "subject": [
          {{
            "subject_id": "01001001",
            "subject_name": "COMPUTER PROGRAMMING",
            "credit": 3,
            "grade_earn": "A"
          }}
        ]
      }}
    ]
  }},
  "footer_detail": {{
    "gpa": 3.50
  }}
}}

**ข้อความ Markdown ที่ต้องสกัด:**
{markdown_text}
"""

    try:
        response = ollama_chat(
            model=MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            fmt={"type": "object"},  # หรือส่ง schema/dict ตามที่นิยามไว้
        )

        if isinstance(response, dict):
            raw_json = response.get("message", {}).get("content", "{}")
        else:
            raw_json = str(response)

        print("=== DEBUG: LLM RAW OUTPUT ===")
        print(raw_json)
        print("=============================")

        data = json.loads(raw_json)

        # หาก postprocess_record ทำงานผิดพลาด ให้ใช้ data ตรงๆ สำรองไว้
        processed_data = postprocess_record(data)
        if not processed_data or not isinstance(processed_data, dict):
            return data

        return processed_data

    except Exception as e:
        print(f"[Error] structure_markdown failed: {e}")
        return {
            "header_detail": {},
            "transcript_detail": {"semesters": []},
            "footer_detail": {},
        }

# ==============================================================================
#  แปลงปี พ.ศ./ค.ศ. ด้วยโค้ด (ไม่ให้ LLM คำนวณเอง)
# ==============================================================================
#
#  เอกสารไทยพิมพ์ปีเป็น พ.ศ.  เอกสารอังกฤษพิมพ์ปีเป็น ค.ศ.
#  แต่ ground truth เก็บ:  year ของภาค = พ.ศ.   วันที่ทุกช่อง = ค.ศ.
#
#  เคยให้ LLM (qwen3:4b) บวก/ลบ 543 เองผ่าน prompt แล้วพบว่า
#    - อ่านตัวเลขปีได้ถูก แต่ไม่ยอมบวก 543
#    - ลอกตัวเลขจากตัวอย่างใน prompt ไปใช้แทนค่าจริง (เช่น 1999, 2553)
#  จึงแบ่งหน้าที่ใหม่: LLM "ลอกปีตามที่พิมพ์" ส่วนโค้ดนี้ "ทำเลขคณิต"
# ==============================================================================

THAI_RATIO_THRESHOLD = 0.3     # สัดส่วนตัวอักษรไทยขั้นต่ำที่ถือว่าเป็นเอกสารภาษาไทย
BE_OFFSET = 543                # พ.ศ. = ค.ศ. + 543
BE_MIN = 2400                  # ตัวกันแปลงซ้ำ: ปีตั้งแต่ 2400 ขึ้นไปถือว่าเป็น พ.ศ. แล้ว
DATE_FIELDS = (("header_detail", "date_of_birth"), ("header_detail", "admis_date"),
               ("header_detail", "grad_date"), ("footer_detail", "updated_at"))

def detect_language(ocr_text: str) -> str:
    """
    เดาภาษาของเอกสารจากสัดส่วนตัวอักษรไทย คืนค่า "th" หรือ "en"

    ⚠️ ต้องส่งข้อความที่ Typhoon อ่านได้จริงเท่านั้น ห้ามส่งข้อความหลัง
       normalize_typhoon_table() หรือที่มี "=== หน้า n ===" เพราะโค้ดเราแทรกคำไทย
       ("[ดิบ]", "[ภาคการศึกษา บล็อกที่ n]") ลงไปในเอกสารอังกฤษด้วย
    เอกสารไทยมีชื่อวิชาภาษาอังกฤษปนได้ แต่ตัวอักษรส่วนใหญ่ยังเป็นไทย
    """
    thai = len(re.findall(r'[฀-๿]', ocr_text))
    latin = len(re.findall(r'[A-Za-z]', ocr_text))
    total = thai + latin
    return "th" if total and thai / total >= THAI_RATIO_THRESHOLD else "en"


def convert_years(data: dict, lang: str) -> dict:
    """
    แปลงปีให้ตรงกับรูปแบบของ ground truth ตามภาษาของเอกสาร

        เอกสารอังกฤษ:  year ของภาค + 543      วันที่ใช้ตามเดิม (ค.ศ. อยู่แล้ว)
        เอกสารไทย   :  year ของภาคใช้ตามเดิม   วันที่ - 543

    ตัวกันแปลงซ้ำ: แปลงเฉพาะค่าที่ยังอยู่ในระบบเดิม
      (year < 2400 = ยังเป็น ค.ศ.,  ปีของวันที่ >= 2400 = ยังเป็น พ.ศ.)
      ถ้า LLM แปลงมาให้แล้ว จะไม่ถูกบวก/ลบซ้ำ
    ไม่แตะ "0000-00-00" (ยังไม่สำเร็จการศึกษา) และไม่แตะ grad_reason
    """
    if not isinstance(data, dict):
        return data

    if lang == "en":
        td = data.get("transcript_detail")
        sems = td.get("semesters") if isinstance(td, dict) else None
        for sem in sems if isinstance(sems, list) else []:
            if not isinstance(sem, dict) or sem.get("year") is None:
                continue
            try:
                y = int(str(sem["year"]).strip())
            except ValueError:
                continue                      # อ่านเป็นตัวเลขไม่ได้ — ปล่อยไว้ตามเดิม
            sem["year"] = y + BE_OFFSET if y < BE_MIN else y

    if lang == "th":
        for section, key in DATE_FIELDS:
            part = data.get(section)
            if not isinstance(part, dict):
                continue
            m = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', str(part.get(key) or "").strip())
            if not m or m.group(1) == "0000":
                continue
            y = int(m.group(1))
            if y >= BE_MIN:
                part[key] = f"{y - BE_OFFSET:04d}-{m.group(2)}-{m.group(3)}"

    return data


def enforce_maintain(data: dict) -> dict:
    """
    ภาค "รักษาสภาพ" (maintain): เฉลยเก็บเป็น pass_reason = "maintain"
    โดยไม่มีรายวิชา และ GPS/GPA เป็น null — บังคับด้วยโค้ด ไม่ต้องหวังให้ LLM ทำตาม

    ครอบสองทาง: ภาคที่ normalizer ทำเครื่องหมายไว้แล้ว และภาคที่ LLM ยังใส่
    "รักษาสภาพ"/"Maintain" มาเป็นรายวิชา
    """
    if not isinstance(data, dict):
        return data
    td = data.get("transcript_detail")
    sems = td.get("semesters") if isinstance(td, dict) else None
    for sem in sems if isinstance(sems, list) else []:
        if not isinstance(sem, dict):
            continue
        subjects = [x for x in (sem.get("subject") or []) if isinstance(x, dict)]
        maintain_rows = [x for x in subjects if _is_maintain(str(x.get("subject_name") or ""))]
        flagged = _is_maintain(str(sem.get("pass_reason") or "")) or bool(maintain_rows)
        if not flagged:
            continue
        real_rows = [x for x in subjects if x not in maintain_rows]
        if real_rows:
            # ภาคนี้มีวิชาจริง แปลว่า flag มาผิดภาค (OCR วางแถวคาบเกี่ยว)
            # เก็บวิชาจริงกับเกรดเฉลี่ยไว้ ถอดเฉพาะร่องรอยของภาครักษาสภาพ
            sem["subject"] = real_rows
            if _is_maintain(str(sem.get("pass_reason") or "")):
                sem["pass_reason"] = None
            continue
        sem["pass_reason"] = "maintain"
        sem["subject"] = []
        sem["GPS"] = None
        sem["GPA"] = None
    return data


# คำนำหน้าชื่อที่ระบบทะเบียนใช้ — ชุดปิด ใช้ตัดชื่อที่ OCR พิมพ์ติดกับคำนำหน้า
PRENAMES = ("นางสาว", "นาง", "นาย", "mrs.", "miss", "mr.", "ms.")
NUMERIC_RE = re.compile(r'^[\d.,]+$')


def _strip_spaces(value: Any) -> str:
    """ลดรูปแบบเดียวกับตอนวัดผล (ตัดช่องว่าง + ตัวพิมพ์เล็ก) ไว้เทียบค่าในโค้ด"""
    return M.normalize(value, "strict")


def postprocess_record(data: dict) -> dict:
    """
    เก็บกวาดค่าที่ LLM มักพลาดแบบเดิมซ้ำ ๆ ด้วยกฎที่ตายตัว

    ทำในโค้ดเพราะเป็นกฎที่ไม่ต้องตีความ และ prompt สั่งแล้วไม่นิ่ง
    ทุกข้อเป็นการ "ถอดค่าที่ผิดออก" ไม่ใช่การเดาค่าใหม่
    """
    if not isinstance(data, dict):
        return data
    head = data.get("header_detail")
    td = data.get("transcript_detail")

    if isinstance(head, dict):
        # 1) คำนำหน้าติดกับชื่อ เช่น prename="นายบัณฑิตคอ3" -> prename="นาย", name="บัณฑิตคอ3..."
        raw_pre = str(head.get("prename") or "").strip()
        flat = _strip_spaces(raw_pre)
        for title in PRENAMES:
            if flat.startswith(title) and flat != title:
                extra = raw_pre[len(raw_pre) - (len(flat) - len(title)):].strip()
                head["prename"] = raw_pre[:len(raw_pre) - len(extra)].strip()
                name = str(head.get("name") or "").strip()
                if extra and not _strip_spaces(name).startswith(_strip_spaces(extra)):
                    head["name"] = f"{extra} {name}".strip()
                break

        # 2) grad_reason ที่เป็น "N/A" ล้วน เฉลยเก็บเป็น null
        #    รวมถึงกรณีคำว่า "maintain"/"รักษาสภาพ" หลุดมาจากภาคเรียน
        #    ไม่ใช่เหตุผลจบการศึกษา
        if (_strip_spaces(head.get("grad_reason")) in ("n/a", "na", "-")
                or _is_maintain(str(head.get("grad_reason") or ""))):
            head["grad_reason"] = None

        # 3) major ซ้ำกับ program = โมเดลลอกมาใส่ (เอกสารชุดนี้ไม่มีช่องสาขาวิชาแยก)
        if head.get("major") is not None and _strip_spaces(head.get("major")) == _strip_spaces(head.get("program")):
            head["major"] = None

    # 4) master_* เก็บข้อความผลสอบ (ผ่าน/pass) — ค่าที่เป็นตัวเลขคือหยิบเกรด
    #    เฉลี่ยมาใส่ผิดช่อง ค่าที่เป็น "maintain"/"รักษาสภาพ" คือหลุดมาจาก
    #    ภาคเรียนรักษาสภาพ ไม่ใช่ผลสอบจริง
    if isinstance(td, dict):
        subject_names = {
            _strip_spaces(x.get("subject_name"))
            for sem in (td.get("semesters") or []) if isinstance(sem, dict)
            for x in (sem.get("subject") or []) if isinstance(x, dict)
        } - {""}
        for key in ("master_comprehensive", "master_thesis", "master_qualify"):
            value = td.get(key)
            if value is None:
                continue
            text = str(value).strip()
            # ค่าที่ตรงกับชื่อวิชาในตาราง = ลอกมาจากรายวิชา ไม่ใช่ผลสอบของหลักสูตร
            if (NUMERIC_RE.match(text) or _is_maintain(text)
                    or _strip_spaces(text) in subject_names):
                td[key] = None
    return data


def pipeline_vlm(pages: list[bytes], save_md: Path | None = None) -> dict:
    """
    สองขั้น:
      ขั้น 1  Typhoon-OCR อ่านภาพ -> Markdown ที่รักษาโครงสร้างตาราง
      ขั้น 2  text LLM อ่าน Markdown -> JSON ตาม schema

    ทำไมต้องแยกสองขั้น แทนที่จะให้ VLM ตัวเดียวทำจบ?
      เพราะ Typhoon-OCR เป็น task-specific model  มันเก่งเรื่อง "อ่านภาษาไทย"
      มาก แต่รับ prompt แบบอื่นไม่ได้ สั่งให้พ่น JSON schema ของเราไม่ได้
      การแยกงานทำให้แต่ละโมเดลทำสิ่งที่ตัวเองถนัด
      = หลักการ "separation of concerns" ในการออกแบบระบบ

    ข้อเสียที่ต้องรู้: error สะสมสองชั้น
      ถ้าขั้น 1 อ่าน "B+" เป็น "8+" ขั้น 2 ก็จะไม่มีทางแก้ให้ถูกได้
      เรียกว่า error propagation — เป็นจุดอ่อนของ pipeline แบบต่อกันเป็นทอด
    """
    md_parts: list[str] = []
    ocr_parts: list[str] = []     # ข้อความจาก Typhoon ล้วน ๆ ไว้ตรวจภาษา (ไม่มี marker ที่เราแทรก)
    for i, png in enumerate(pages):
        print(f"    [ขั้น 1/2] Typhoon-OCR อ่านหน้า {i + 1}/{len(pages)}...")
        md = ollama_chat(
            MODEL_OCR,
            [{"role": "user", "content": TYPHOON_PROMPT}],
            images=[png],
            temperature=0,   # เอกสารทางการแนะนำ 0-0.1 สำหรับโมเดล OCR
        )
        md_parts.append(f"\n=== หน้า {i + 1} ===\n{md}")
        ocr_parts.append(md)

    document_text = "\n".join(md_parts)

    # เก็บ Markdown กลางทางไว้ดูด้วย — สำคัญมากตอน debug
    # ถ้า JSON ออกมาผิด เราต้องรู้ว่าผิดที่ขั้นไหน
    normalized_document_text = normalize_typhoon_table(document_text)
    if save_md:
        save_md.write_text(normalized_document_text, encoding="utf-8")
        print(f"    บันทึก Markdown กลางทาง: {save_md}")

    print(f"    [ขั้น 2/2] {MODEL_TEXT} จัดรูปเป็น JSON...")
    raw = ollama_chat(
        MODEL_TEXT,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACT_PROMPT.format(document_text=normalized_document_text)},
        ],
        fmt=TRANSCRIPT_SCHEMA,
        think=False,
    )
    print("=== RAW ===")
    print(raw)
    print("=== END RAW ===")
    data = parse_json(raw)

    # แปลงปี พ.ศ./ค.ศ. ด้วยโค้ด ตามภาษาของเอกสาร (LLM ลอกปีตามที่พิมพ์มา)
    lang = detect_language("\n".join(ocr_parts))
    print(f"    ภาษาเอกสาร: {lang} -> แปลงปีด้วยโค้ด")
    data = convert_years(data, lang)
    data = enforce_maintain(data)
    data = postprocess_record(data)
    if isinstance(data, dict):
        data["_meta"] = {**(data.get("_meta") or {}), "detected_lang": lang}
    return data



# ==============================================================================
#  ส่วนที่ 9 — ตรวจสอบความสอดคล้องภายใน (self-verification)
# ==============================================================================
#
#  กลไกนี้ตรวจว่า JSON ที่โมเดลพ่นออกมา "สมเหตุสมผลในตัวเอง" หรือไม่
#  โดยไม่ต้องใช้ ground truth เลย ซึ่งเป็นสถานการณ์จริงเมื่อนำระบบไปใช้งาน
#  กับ transcript ใบใหม่ที่ไม่เคยมีใครกรอกเฉลยไว้
#
#  ────────────────────────────────────────────────────────────────────────
#  ⚠️ ทำไมเราถึง "ไม่" ใช้การคำนวณ GPA มาตรวจ  (บทเรียนสำคัญของแล็บนี้)
#  ────────────────────────────────────────────────────────────────────────
#  ตอนออกแบบครั้งแรก เราเคยคิดจะใช้กฎว่า
#       Σ(ค่าระดับคะแนน × หน่วยกิต) ÷ Σ(หน่วยกิต)  ต้องเท่ากับ GPA ที่พิมพ์ไว้
#  ฟังดูสวยงามมาก เพราะเหมือน transcript มี "เฉลย" ซ่อนอยู่ในตัวเอง
#
#  แต่พอเอาไปใช้กับเอกสารจริง กฎนี้แจ้งเตือนผิดพลาด (false alarm) บ่อยมาก
#  เพราะระเบียบการคิดเกรดเฉลี่ยจริงซับซ้อนกว่าสูตรข้างบนหลายชั้น:
#
#    1. การปัดเศษ — บางระบบปัดขึ้น บางระบบตัดทิ้ง (truncate)
#       2.335 อาจกลายเป็น 2.34 หรือ 2.33 ก็ได้ ต่างกันที่ทศนิยมตำแหน่งที่ 2
#    2. วิชาลงซ้ำ (retake) — บางระเบียบให้เกรดใหม่แทนเกรดเดิม
#       บางระเบียบให้นับทั้งสองครั้ง  เราไม่รู้จากตัวเอกสารว่าใช้แบบไหน
#    3. วิชาเทียบโอน (T) และวิชาที่ให้เกรด S/U — ไม่นำมาคิด GPA
#       แต่บางใบพิมพ์หน่วยกิตไว้ในคอลัมน์เดียวกัน แยกไม่ออกจากตัวเอกสาร
#    4. วิชาที่ถอน (W) กลางภาค บางใบยังแสดงหน่วยกิตไว้
#
#  ────────────────────────────────────────────────────────────────────────
#  หลักการที่ได้จากเรื่องนี้ (สำคัญกว่าตัวกฎเอง)
#  ────────────────────────────────────────────────────────────────────────
#      "กฎตรวจสอบที่แจ้งเตือนผิดบ่อย แย่กว่าการไม่มีกฎเลย"
#
#  เพราะเมื่อระบบเตือนบ่อยจนคนเลิกเชื่อ พอมันเตือนถูกจริง ๆ ก็จะไม่มีใครสนใจ
#  ปรากฏการณ์นี้เรียกว่า alarm fatigue (ความล้าจากการเตือน)
#  พบได้ในระบบเฝ้าระวังผู้ป่วยในโรงพยาบาล และระบบแจ้งเตือนความปลอดภัย
#
#  เราจึงเลือกใช้เฉพาะ "กฎที่ผิดไม่ได้" คือกฎเชิงโครงสร้างและเชิงรูปแบบ
#  ซึ่งไม่ขึ้นกับการตีความระเบียบใด ๆ
#
#  GPA และ GPS ยังคงถูก "ถอดออกมา" ตามที่พิมพ์ไว้ในเอกสาร
#  แต่จะถูกนำไปวัดความถูกต้องในส่วนที่ 10 (เทียบกับ ground truth) เท่านั้น
#  ไม่นำมาคำนวณย้อนกลับ
# ==============================================================================

# เกรดที่ปรากฏได้ในใบแสดงผลการศึกษาของสถาบันในประเทศไทย
VALID_GRADES = {
    "a", "b+", "b", "c+", "c", "d+", "d", "f",      # เกรดที่คิดคะแนน
    "w",        # ถอนรายวิชา (withdrawn)
    "s", "u",   # ผ่าน / ไม่ผ่าน (satisfactory / unsatisfactory)
    "i",        # ยังไม่สมบูรณ์ (incomplete)
    "p",        # ผ่าน (pass)
    "t",        # เทียบโอน (transfer)
    "ผ", "ม",   # ผ่าน / ไม่ผ่าน แบบภาษาไทย
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def verify_internal(data: dict) -> dict:
    """
    ตรวจความสอดคล้องภายในด้วยกฎเชิงโครงสร้าง 7 ข้อ

    ทุกกฎในนี้ถูกเลือกด้วยเกณฑ์เดียว:
        "ถ้ากฎนี้แจ้งเตือน แปลว่ามีอะไรผิดจริงแน่นอน ไม่ใช่แค่ตีความต่างกัน"

    ถ้าคิดกฎใหม่ได้ ให้ถามตัวเองด้วยเกณฑ์นี้ก่อนเสมอ
    """
    issues: list[str] = []
    td = data.get("transcript_detail") or {}
    sems = td.get("semesters") or []
    hd = data.get("header_detail") or {}

    # ── กฎ 1: รหัสนักศึกษาและวันที่ต้องถูกรูปแบบ ──────────────────────
    sid = M.normalize(hd.get("student_id"), "strict")
    if sid and not re.fullmatch(r"\d{8}", sid):
        issues.append(f"รหัสนักศึกษาผิดรูปแบบ: {hd.get('student_id')!r} "
                      f"(ต้องเป็นตัวเลข 8 หลัก)")
    for f in ("date_of_birth", "admis_date", "grad_date"):
        v = hd.get(f)
        if v and not DATE_RE.match(str(v)):
            issues.append(f"{f} ผิดรูปแบบ: {v!r} (ต้องเป็น YYYY-MM-DD)")

    # ── กฎ 2: ปี/ภาค ต้องอยู่ในช่วงที่เป็นไปได้ ───────────────────────
    seen_terms: set[tuple] = set()
    for s in sems:
        y, n = s.get("year"), s.get("sem_num")
        if not isinstance(y, int) or not (2500 <= y <= 2600):
            issues.append(f"ปีการศึกษาผิดปกติ: {y!r} (คาดว่าเป็น พ.ศ. เช่น 2561)")
        if n not in (1, 2, 3):
            issues.append(f"ภาคการศึกษาผิดปกติ: {n!r} (ต้องเป็น 1, 2 หรือ 3)")

        # ── กฎ 3: ห้ามมีภาคการศึกษาซ้ำ ──────────────────────────────
        # ถ้าซ้ำ มักแปลว่าโมเดลอ่านหน้าเดิมสองรอบ หรือแบ่งภาคผิด
        key = (y, n)
        if key in seen_terms:
            issues.append(f"ภาค {y}/{n} ปรากฏซ้ำมากกว่าหนึ่งครั้ง "
                          f"--> อาจอ่านหน้าเดิมซ้ำ หรือแบ่งภาคผิด")
        seen_terms.add(key)

    # ── กฎ 4: ภาคการศึกษาต้องเรียงจากเก่าไปใหม่ ───────────────────────
    # transcript พิมพ์เรียงตามเวลาเสมอ ถ้าไม่เรียง แปลว่าโมเดลสลับลำดับ
    order = [(s.get("year"), s.get("sem_num")) for s in sems
             if isinstance(s.get("year"), int) and isinstance(s.get("sem_num"), int)]
    if order != sorted(order):
        issues.append("ลำดับภาคการศึกษาไม่เรียงจากเก่าไปใหม่ "
                      "--> โมเดลอาจสลับลำดับ หรืออ่านข้ามหน้า")

    # ── ตรวจระดับรายวิชา ─────────────────────────────────────────────
    n_subj_total = 0
    n_grade_null = 0

    for s in sems:
        tag = f"{s.get('year')}/{s.get('sem_num')}"
        subs = s.get("subject") or []
        n_subj_total += len(subs)

        # ── กฎ 5: จำนวนวิชาต่อภาคต้องสมเหตุสมผล ────────────────────
        # ระเบียบทั่วไปลงได้ 9-22 หน่วยกิต ~ 3-8 วิชา
        # เผื่อกรณีภาคฤดูร้อนและกรณีพิเศษ จึงตั้งช่วงกว้างไว้ที่ 1-12
        if len(subs) == 0:
            issues.append(f"ภาค {tag} ไม่มีรายวิชาเลย --> น่าจะอ่านตกทั้งตาราง")
        elif len(subs) > 12:
            issues.append(f"ภาค {tag} มีถึง {len(subs)} วิชา "
                          f"--> น่าจะรวมสองภาคเข้าด้วยกัน หรือมีวิชาซ้ำ")

        seen_codes: set[str] = set()
        term_credits = 0

        for sub in subs:
            code = M.normalize(sub.get("subject_id"), "strict")
            g = M.normalize(sub.get("grade_earn"), "strict")
            c = sub.get("credit")

            # ── กฎ 6: รูปแบบของแต่ละช่อง ────────────────────────────
            if not re.fullmatch(r"\d{8}", code):
                issues.append(f"[{tag}] รหัสวิชาผิดรูปแบบ: {sub.get('subject_id')!r}")

            if not g:
                # ตอบ null = โมเดลยอมรับว่าอ่านไม่ออก ซึ่งเป็นพฤติกรรมที่เราต้องการ
                # ไม่ถือเป็นข้อผิดพลาด แต่ต้องนับไว้รายงาน
                n_grade_null += 1
            elif g not in VALID_GRADES:
                issues.append(f"[{tag}] เกรดไม่ถูกต้อง {sub.get('grade_earn')!r} "
                              f"ที่วิชา {sub.get('subject_id')}")

            if not isinstance(c, int):
                issues.append(f"[{tag}] หน่วยกิตไม่ใช่จำนวนเต็ม: {c!r} "
                              f"ที่วิชา {sub.get('subject_id')}")
            elif not (1 <= c <= 9):
                issues.append(f"[{tag}] หน่วยกิตผิดปกติ: {c} "
                              f"ที่วิชา {sub.get('subject_id')} (คาดว่า 1-9)")
            else:
                term_credits += c

            # ── กฎ 7: ห้ามมีรหัสวิชาซ้ำภายในภาคเดียวกัน ─────────────
            # ลงวิชาเดิมซ้ำในภาคเดียวกันเป็นไปไม่ได้ตามระเบียบ
            # (ลงซ้ำต่างภาคได้ ซึ่งเราไม่ตรวจ)
            if code and code in seen_codes:
                issues.append(f"[{tag}] รหัสวิชา {code} ปรากฏซ้ำในภาคเดียวกัน "
                              f"--> อาจอ่านแถวเดิมสองครั้ง")
            seen_codes.add(code)

        # หน่วยกิตรวมต่อภาค — ตั้งช่วงกว้างเพราะภาคฤดูร้อนลงน้อยได้
        if term_credits > 30:
            issues.append(f"ภาค {tag} มีหน่วยกิตรวม {term_credits} "
                          f"--> น่าจะรวมสองภาคเข้าด้วยกัน")

    return {
        "ok": len(issues) == 0,
        "n_semesters": len(sems),
        "n_subjects": n_subj_total,
        "n_grade_null": n_grade_null,       # โมเดลยอมรับว่าอ่านไม่ออกกี่ช่อง
        "issues": issues,
    }


# ==============================================================================
#  ส่วนที่ 10 — ประเมินผลเทียบกับ GROUND TRUTH
# ==============================================================================
#
#  รายการ attribute ที่เราจะวัดแยกกัน  แต่ละตัวมีลักษณะไม่เหมือนกัน:
#
#  ⭐ หลักการ: วัด "ทุกฟิลด์ในเอกสาร" ไม่เลือกวัดเฉพาะบางส่วน
#
#     เหตุผลคือถ้าเราเลือกวัดเฉพาะฟิลด์ที่โมเดลทำได้ดี ตัวเลขจะสวยเกินจริง
#     และเราจะไม่รู้เลยว่าฟิลด์ที่เหลือพังหรือเปล่า
#     ในไฟล์ ground truth มี 14 ฟิลด์ในส่วนหัว + 5 ฟิลด์สรุป + 4 ฟิลด์ท้าย
#     บวกกับฟิลด์รายภาคและรายวิชา  ทั้งหมดต้องถูกวัด
#
#    student_id    ตัวเลขล้วน   -> ผิดตัวเดียวก็ใช้ไม่ได้  วัดด้วย exact match เป็นหลัก
#    name          ไทย          -> วัด CER  (WER ไม่มีความหมาย เพราะ GT ลบช่องว่างหมด)
#    subject_id    ตัวเลขล้วน   -> เป็น "กุญแจ" ที่ใช้จับคู่แถวด้วย
#    subject_name  ไทย+อังกฤษ  -> วัด CER
#    credit        จำนวนเต็ม    -> exact match
#    grade_earn    หมวดหมู่     -> exact match  ⭐ ตัวชี้วัดที่สำคัญที่สุด
#    GPA / GPS     ตัวเลขทศนิยม -> exact match
#
#  ⚠️ ทำไม transcript ถึงไม่วัด WER ของ subject_name?
#     เพราะ ground truth ที่ให้มาถูก "ลบช่องว่างออกหมด" แล้ว
#     ("programmingformanagement")  พอลบช่องว่าง การตัดคำก็ไร้ความหมาย
#     --> เราจึงปิด WER ไว้สำหรับกลุ่ม A และเปิดสำหรับกลุ่ม B (หลักสูตร)
#         ซึ่ง ground truth ยังคงช่องว่างไว้
#     นี่เป็นบทเรียนว่า "รูปแบบของ ground truth กำหนดว่าวัดอะไรได้บ้าง"
# ==============================================================================


def evaluate(pred: dict, gt: dict) -> tuple[dict, dict]:
    """เทียบ prediction กับ ground truth แล้วคืน (stats, สรุปการจับคู่แถว)"""

    S = M.FieldStat   # ย่อชื่อให้อ่านง่าย
    # ---- 13 กลุ่ม ครอบคลุมทุกฟิลด์ในเอกสาร ----
    stats: dict[str, M.FieldStat] = {
        # ── ส่วนหัว 14 ฟิลด์ ──
        "student_id":   S("รหัสนักศึกษา ⭐"),
        "person":       S("คำนำหน้า/ชื่อ ⭐"),
        "institution":  S("สถาบัน/ที่อยู่"),
        "faculty":      S("คณะ/หลักสูตร"),
        "dates":        S("วันที่"),
        "status":       S("สถานะ/เกียรตินิยม"),
        # ── รายภาคการศึกษา ──
        "sem_meta":     S("ปี/ภาค"),
        "gpa":          S("GPA/GPS"),
        "sem_misc":     S("หมายเหตุรายภาค"),
        # ── รายวิชา ──
        "subject_id":   S("รหัสวิชา"),
        "subject_name": S("ชื่อวิชา"),
        "subject_type": S("ประเภทวิชา"),
        "credit":       S("หน่วยกิต"),
        "grade":        S("เกรด ⭐"),
        # ── สรุปและท้ายเอกสาร ──
        "summary":      S("สรุปผลการศึกษา"),
        "footer":       S("ผู้รับรองท้ายเอกสาร"),
    }

    gh = gt.get("header_detail") or {}
    ph = pred.get("header_detail") or {}

    # ---------- ส่วนหัว: ครบทั้ง 14 ฟิลด์ ----------
    # ⭐ ส่วนหัวคือข้อมูลที่บอกว่า "ผลการเรียนนี้เป็นของใคร"
    #    ถ้าอ่านรหัสนักศึกษาผิดตัวเดียว ผลการเรียนทั้งใบจะไปผูกกับคนผิด
    #    จึงเป็นส่วนที่สำคัญไม่แพ้ตารางเกรด และต้องวัดให้ครบทุกฟิลด์
    stats["student_id"].add(gh.get("student_id"), ph.get("student_id"),
                            "student_id", track_wer=False)
    for f in ("prename", "name"):
        stats["person"].add(gh.get(f), ph.get(f), f, track_wer=False)

    # ชื่อและที่อยู่สถาบัน — ข้อความไทยยาวที่สุดในเอกสาร
    # เป็นตัววัดความสามารถอ่านภาษาไทยที่ดีมาก เพราะมีทั้งคำยาวและตัวเลขปน
    for f in ("uni_name", "uni_address"):
        stats["institution"].add(gh.get(f), ph.get(f), f, track_wer=False)

    for f in ("faculty_name", "degree", "program", "major"):
        stats["faculty"].add(gh.get(f), ph.get(f), f, track_wer=False)
    for f in ("date_of_birth", "admis_date", "grad_date"):
        stats["dates"].add(gh.get(f), ph.get(f), f, track_wer=False)

    # สถานะการสำเร็จการศึกษา
    #   grad_reason เป็นข้อความอิสระ เช่น "n/a(พ้นสภาพ1/2562)"
    #   honor เป็นตัวเลข 0/1/2  --> โมเดลมักเดาเป็น 0 เสมอ ต้องจับให้ได้
    for f in ("grad_reason", "honor"):
        stats["status"].add(gh.get(f), ph.get(f), f, track_wer=False)

    # ---------- ภาคการศึกษา ----------
    gsems = (gt.get("transcript_detail") or {}).get("semesters") or []
    psems = (pred.get("transcript_detail") or {}).get("semesters") or []

    # จับคู่ภาคการศึกษาด้วยกุญแจ "ปี/ภาค"
    # ⚠️ ห้ามจับคู่ตาม index! ถ้าโมเดลอ่านตกไป 1 ภาค ทุกภาคหลังจะเลื่อนหมด
    sem_align = M.align_by_key(
        gsems, psems,
        key_fn=lambda s: f"{s.get('year')}/{s.get('sem_num')}",
    )

    subj_align_total = M.AlignResult()

    for g_sem, p_sem in sem_align.matched:
        stats["sem_meta"].add(g_sem.get("year"), p_sem.get("year"),
                              "year", track_wer=False)
        stats["sem_meta"].add(g_sem.get("sem_num"), p_sem.get("sem_num"),
                              "sem_num", track_wer=False)
        tag = f"{g_sem.get('year')}/{g_sem.get('sem_num')}"
        stats["gpa"].add(g_sem.get("GPA"), p_sem.get("GPA"),
                         f"GPA@{tag}", track_wer=False)
        stats["gpa"].add(g_sem.get("GPS"), p_sem.get("GPS"),
                         f"GPS@{tag}", track_wer=False)
        # pass_reason มักเป็น null  ถ้าโมเดลเติมค่าขึ้นมา = hallucination
        stats["sem_misc"].add(g_sem.get("pass_reason"), p_sem.get("pass_reason"),
                              f"pass_reason@{tag}", track_wer=False)

        # ---------- รายวิชาในภาคนั้น ----------
        sa = M.align_by_key(
            g_sem.get("subject") or [], p_sem.get("subject") or [],
            key_fn=lambda s: M.normalize(s.get("subject_id"), "strict"),
        )
        subj_align_total.matched.extend(sa.matched)
        subj_align_total.missed.extend(sa.missed)
        subj_align_total.spurious.extend(sa.spurious)

        for g_sub, p_sub in sa.matched:
            key = f"{tag} {g_sub.get('subject_id')}"
            stats["subject_id"].add(g_sub.get("subject_id"),
                                    p_sub.get("subject_id"), key, track_wer=False)
            stats["subject_name"].add(g_sub.get("subject_name"),
                                      p_sub.get("subject_name"), key, track_wer=False)
            stats["subject_type"].add(g_sub.get("type"), p_sub.get("type"),
                                      key, track_wer=False)
            stats["credit"].add(g_sub.get("credit"), p_sub.get("credit"),
                                key, track_wer=False)
            stats["grade"].add(g_sub.get("grade_earn"), p_sub.get("grade_earn"),
                               key, track_wer=False)

        # วิชาที่โมเดลอ่านตก: นับเป็น deletion เต็มจำนวนตัวอักษร
        # ถ้าไม่นับ ผลจะดูดีเกินจริง (โมเดลที่อ่านแค่ 3 วิชาแล้วถูกหมด
        # จะได้ CER = 0 ทั้งที่ตกไป 20 วิชา!)
        for g_sub in sa.missed:
            k = f"{tag} {g_sub.get('subject_id')} [ตกแถว]"
            stats["subject_id"].add(g_sub.get("subject_id"), "", k, track_wer=False)
            stats["subject_name"].add(g_sub.get("subject_name"), "", k, track_wer=False)
            stats["subject_type"].add(g_sub.get("type"), "", k, track_wer=False)
            stats["credit"].add(g_sub.get("credit"), "", k, track_wer=False)
            stats["grade"].add(g_sub.get("grade_earn"), "", k, track_wer=False)

    # ภาคที่โมเดลอ่านตกทั้งภาค
    for g_sem in sem_align.missed:
        tag = f"{g_sem.get('year')}/{g_sem.get('sem_num')} [ตกทั้งภาค]"
        for g_sub in g_sem.get("subject") or []:
            k = f"{tag} {g_sub.get('subject_id')}"
            stats["subject_id"].add(g_sub.get("subject_id"), "", k, track_wer=False)
            stats["subject_name"].add(g_sub.get("subject_name"), "", k, track_wer=False)
            stats["subject_type"].add(g_sub.get("type"), "", k, track_wer=False)
            stats["credit"].add(g_sub.get("credit"), "", k, track_wer=False)
            stats["grade"].add(g_sub.get("grade_earn"), "", k, track_wer=False)
            subj_align_total.missed.append(g_sub)

    # ---------- สรุปผลการศึกษา (5 ฟิลด์) ----------
    gtd = gt.get("transcript_detail") or {}
    ptd = pred.get("transcript_detail") or {}
    # master_* เป็น null ในใบปริญญาตรี  ถ้าโมเดลเติมค่า = hallucination
    for f in ("total_credits_earned", "cumulative_gpa",
              "master_comprehensive", "master_thesis", "master_qualify"):
        stats["summary"].add(gtd.get(f), ptd.get(f), f, track_wer=False)

    # ---------- ท้ายเอกสาร (4 ฟิลด์) ----------
    # ⚠️ ส่วนนี้อยู่ล่างสุดของหน้า ซึ่งเป็นตำแหน่งที่มักถูกอ่านตกมากที่สุด
    #    ทั้งจาก OCR (ขอบกระดาษ) และจาก LLM (ชน num_predict แล้วตัดท้ายทิ้ง)
    #    เป็นตัวชี้วัดที่ดีว่าโมเดลอ่าน "จนจบเอกสาร" จริงหรือไม่
    gf = gt.get("footer_detail") or {}
    pf = pred.get("footer_detail") or {}
    stats["footer"].add(gf.get("updated_at"), pf.get("updated_at"),
                        "updated_at", track_wer=False)
    gby = gf.get("by") or {}
    pby = pf.get("by") or {}
    for f in ("by_signature", "by_position", "by_reg"):
        stats["footer"].add(gby.get(f), pby.get(f), f, track_wer=False)

    align_summary = {
        "semester": {
            "matched": len(sem_align.matched),
            "missed": len(sem_align.missed),
            "spurious": len(sem_align.spurious),
            "precision": round(sem_align.precision, 4),
            "recall": round(sem_align.recall, 4),
            "f1": round(sem_align.f1, 4),
        },
        "subject": {
            "matched": len(subj_align_total.matched),
            "missed": len(subj_align_total.missed),
            "spurious": len(subj_align_total.spurious),
            "precision": round(subj_align_total.precision, 4),
            "recall": round(subj_align_total.recall, 4),
            "f1": round(subj_align_total.f1, 4),
        },
    }
    return stats, align_summary


# ==============================================================================
#  ส่วนที่ 11 — MAIN
# ==============================================================================


def run_pipeline(name: str, pages: list[bytes], outdir: Path) -> dict | None:
    """เรียก pipeline ตามชื่อ พร้อมจับ error ไม่ให้ทั้งโปรแกรมล้ม"""
    print(f"\n{'─' * 70}")
    print(f"  PIPELINE: {name}")
    print(f"{'─' * 70}")
    t0 = time.time()
    try:
        if name == "baseline":
            data = pipeline_baseline(pages)
        elif name == "vlm":
            data = pipeline_vlm(pages, save_md=outdir / "intermediate_vlm.md")
        else:
            raise ValueError(f"ไม่รู้จัก pipeline: {name}")
    except Exception as e:
        print(f"  ❌ {name} ล้มเหลว: {e}")
        return None

    if not data:
        return None

    data["_meta"] = {
        "pipeline": name,
        "elapsed_sec": round(time.time() - t0, 1),
        "models": {"ocr": MODEL_OCR, "text": MODEL_TEXT},
        "dpi": DPI,
    }
    path = outdir / f"pred_{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ บันทึก {path}  (ใช้เวลารวม {data['_meta']['elapsed_sec']} วิ)")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Lab 7A — สกัดข้อมูล Transcript ด้วย LLM ที่รันบนเครื่อง",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-i", "--input", help="ไฟล์ transcript (.pdf/.png/.jpg)")
    ap.add_argument("-g", "--gt", help="ไฟล์ ground truth (.json)")
    ap.add_argument("-o", "--out", default=str(PROJECT_ROOT / "outputs" / "lab7a"),
                    help="โฟลเดอร์ผลลัพธ์ (ค่าเริ่มต้น outputs/lab7a ที่รากโปรเจกต์)")
    ap.add_argument("-p", "--pipeline", default="all",
                    choices=["all", "baseline", "vlm"])
    ap.add_argument("--check", action="store_true", help="ตรวจความพร้อมของเครื่องแล้วออก")
    ap.add_argument("--eval-only", metavar="PRED_JSON",
                    help="ข้ามการเรียกโมเดล ประเมินจากไฟล์ JSON ที่มีอยู่แล้ว")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check_environment() else 1)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # โหมดประเมินอย่างเดียว
    # ------------------------------------------------------------------
    if args.eval_only:
        if not args.gt:
            raise SystemExit("❌ --eval-only ต้องระบุ --gt ด้วย")
        pred = json.loads(Path(args.eval_only).read_text(encoding="utf-8"))
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        stats, align = evaluate(pred, gt)
        M.print_table(stats, f"ผลประเมิน: {Path(args.eval_only).name}")
        print(f"\n  การจับคู่รายวิชา: matched={align['subject']['matched']} "
              f"missed={align['subject']['missed']} "
              f"spurious={align['subject']['spurious']}  "
              f"(F1={align['subject']['f1']:.3f})")
        M.print_errors(stats)
        return

    if not args.input:
        raise SystemExit("❌ ต้องระบุ --input  (ดูตัวอย่าง: --help)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Lab 7A — Transcript OCR ด้วย LLM ที่รันบนเครื่องตัวเอง")
    print("=" * 70)
    assert_offline()

    print(f"\nเตรียมภาพจาก: {args.input}")
    pages = load_pages(args.input)

    if args.pipeline == "all":
        names = ["vlm"] if SKIP_BASELINE else ["baseline", "vlm"]
        if SKIP_BASELINE:
            print("\n(ข้าม pipeline baseline ตามค่า LAB7_SKIP_BASELINE=1)")
    else:
        names = [args.pipeline]

    results: dict[str, dict] = {}
    for n in names:
        r = run_pipeline(n, pages, outdir)
        if r:
            results[n] = r

    # ------------------------------------------------------------------
    # ตรวจความสอดคล้องภายใน (ไม่ต้องใช้เฉลย)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ตรวจความสอดคล้องภายใน (กฎเชิงโครงสร้าง — ไม่ใช้ ground truth)")
    print("=" * 70)
    for n, data in results.items():
        v = verify_internal(data)
        icon = "✓" if v["ok"] else "✗"
        print(f"\n  {icon} {n}: {v['n_semesters']} ภาค, {v['n_subjects']} วิชา, "
              f"อ่านเกรดไม่ออก {v['n_grade_null']} ช่อง")
        for msg in v["issues"][:6]:
            print(f"      • {msg}")
        if len(v["issues"]) > 6:
            print(f"      ... และอีก {len(v['issues']) - 6} รายการ")

    # ------------------------------------------------------------------
    # เทียบกับ ground truth
    # ------------------------------------------------------------------
    if not args.gt:
        print("\n(ไม่ได้ระบุ --gt จึงข้ามการเทียบกับเฉลย)")
        return

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    all_csv_rows = outdir / "comparison.csv"
    combined: dict[str, Any] = {}

    for n, data in results.items():
        stats, align = evaluate(data, gt)
        M.print_table(stats, f"PIPELINE = {n}")
        print(f"  การจับคู่รายวิชา: "
              f"อ่านเจอ {align['subject']['matched']} / "
              f"ตก {align['subject']['missed']} / "
              f"แต่งเกิน {align['subject']['spurious']}   "
              f"P={align['subject']['precision']:.3f} "
              f"R={align['subject']['recall']:.3f} "
              f"F1={align['subject']['f1']:.3f}")
        M.print_errors(stats, limit=2)

        d = M.stats_to_dict(stats)
        d["alignment"] = align
        d["internal_check"] = verify_internal(data)
        combined[n] = d

        # เขียน CSV แบบต่อท้าย เพื่อเอาไปทำกราฟใน Excel
        mode = "a" if all_csv_rows.exists() and n != names[0] else "w"
        tmp = outdir / f"_tmp_{n}.csv"
        M.save_csv(stats, str(tmp), extra={"pipeline": n})
        with open(all_csv_rows, mode, encoding="utf-8-sig") as fout:
            lines = tmp.read_text(encoding="utf-8-sig").splitlines()
            fout.write("\n".join(lines if mode == "w" else lines[1:]) + "\n")
        tmp.unlink()

    (outdir / "evaluation.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ เสร็จสิ้น")
    print(f"  ตารางเปรียบเทียบ (เปิดใน Excel): {all_csv_rows}")
    print(f"  ผลละเอียด: {outdir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
