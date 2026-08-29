#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 lab7a_transcript.py
 Lab 7A — สกัดข้อมูลใบแสดงผลการศึกษา (Transcript) ด้วย LLM ที่รันบนเครื่องตัวเอง
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
from typing import Any

# --- เพิ่ม path ของโฟลเดอร์แม่ เพื่อ import โมดูลกลาง lab7_metrics ---------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lab7_metrics as M  # noqa: E402


# ==============================================================================
#  ส่วนที่ 0 — ค่าตั้งต้น
# ==============================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

MODEL_OCR = os.getenv("LAB7_MODEL_OCR", "scb10x/typhoon-ocr1.5-3b")
MODEL_TEXT = os.getenv("LAB7_MODEL_TEXT", "qwen3:4b")

DPI = int(os.getenv("LAB7_DPI", "150"))
REQUEST_TIMEOUT = 900       # วินาที

SKIP_BASELINE = os.getenv("LAB7_SKIP_BASELINE", "").strip() in ("1", "true", "yes")

# ==============================================================================
#  ส่วนที่ 1 — ตรวจความพร้อมของเครื่อง และ ยืนยันว่าออฟไลน์
# ==============================================================================

def _need(mod: str, pipname: str = "") -> Any:
    try:
        return __import__(mod)
    except ImportError:
        raise SystemExit(
            f"\n❌ ไม่พบไลบรารี '{mod}'\n"
            f"   ติดตั้งด้วย:  pip install {pipname or mod}\n"
        )


def assert_offline() -> None:
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
    ok = True
    print("\n" + "=" * 70)
    print("  ตรวจความพร้อมของเครื่อง")
    print("=" * 70)

    exe = shutil.which("ollama")
    if exe:
        try:
            v = subprocess.run(["ollama", "--version"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
            print(f"  ✓ พบ Ollama: {v}")
        except Exception:
            print("  ✓ พบ Ollama (อ่านเวอร์ชันไม่ได้)")
    else:
        print("  ✗ ไม่พบคำสั่ง ollama")
        ok = False

    try:
        requests = _need("requests")
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        installed = [m["name"] for m in r.json().get("models", [])]
        print(f"  ✓ Ollama service ทำงานที่ {OLLAMA_HOST}")
        print(f"    โมเดลที่มีในเครื่อง ({len(installed)}):")
        for m in installed:
            print(f"      - {m}")

        for tag, role in [(MODEL_OCR, "อ่านภาพ"), (MODEL_TEXT, "จัด JSON")]:
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

    for mod, pip in [("fitz", "pymupdf"), ("PIL", "pillow"), ("requests", "requests")]:
        try:
            __import__(mod)
            print(f"  ✓ python: {mod}")
        except ImportError:
            print(f"  ✗ python: {mod}   --> pip install {pip}")
            ok = False

    print("\n  ส่วนเสริม (ขาดได้ ไม่ทำให้ --check ตก):")

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
                miss.append("ติดตั้งตัว engine")
            print(f"  ✗ tesseract — {' + '.join(miss)}")

    try:
        __import__("pythainlp")
        print("  ✓ pythainlp (ตัดคำไทยสำหรับ WER)")
    except ImportError:
        print("  ✗ pythainlp   --> pip install pythainlp")

    print("=" * 70)
    print("  พร้อมใช้งาน ✓" if ok else "  ยังไม่พร้อม — แก้ตามรายการ ✗ ด้านบน")
    print("=" * 70 + "\n")
    return ok


# ==============================================================================
#  ส่วนที่ 2 — เตรียมภาพจาก input
# ==============================================================================

def load_pages(path: str) -> list[bytes]:
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

    mat = fitz.Matrix(DPI / 72, DPI / 72)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
        print(f"  แปลงหน้า {i + 1}/{len(doc)}  ({pix.width}x{pix.height} px @ {DPI} DPI)")
    doc.close()
    return pages


# ==============================================================================
#  ส่วนที่ 3 — JSON SCHEMA
# ==============================================================================

TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "header_detail": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "prename": {"type": "string"},
                "name": {"type": "string"},
                "uni_name": {"type": "string"},
                "uni_address": {"type": "string"},
                "faculty_name": {"type": "string"},
                "degree": {"type": "string"},
                "program": {"type": "string"},
                "major": {"type": "string"},
                "date_of_birth": {"type": "string"},
                "admis_date": {"type": "string"},
                "grad_date": {"type": "string"},
                "grad_reason": {"type": "string"},
                "honor": {"type": "string"}
            }
        },
        "transcript_detail": {
            "type": "object",
            "properties": {
                "semesters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "year": {"type": "integer"},
                            "sem_num": {"type": "integer"},
                            "GPA": {"type": "number"},
                            "GPS": {"type": "number"},
                            "pass_reason": {"type": ["string", "null"]},
                            "subject": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject_id": {"type": "string"},
                                        "subject_name": {"type": "string"},
                                        "type": {"type": ["string", "null"]},
                                        "credit": {"type": "number"},
                                        "grade_earn": {"type": "string"}
                                    },
                                    "required": ["subject_id", "subject_name", "credit", "grade_earn"]
                                }
                            }
                        },
                        "required": ["year", "sem_num", "subject"]
                    }
                },
                "total_credits_earned": {"type": "number"},
                "cumulative_gpa": {"type": "number"},
                "master_comprehensive": {"type": ["string", "null"]},
                "master_thesis": {"type": ["string", "null"]},
                "master_qualify": {"type": ["string", "null"]}
            }
        },
        "footer_detail": {
            "type": "object",
            "properties": {
                "updated_at": {"type": "string"},
                "by": {
                    "type": "object",
                    "properties": {
                        "by_signature": {"type": "string"},
                        "by_position": {"type": "string"},
                        "by_reg": {"type": "string"}
                    }
                }
            }
        }
    },
    "required": ["header_detail", "transcript_detail", "footer_detail"]
}


# ==============================================================================
#  ส่วนที่ 4 — PROMPT
# ==============================================================================

SYSTEM_PROMPT = """You are a precise document transcription system for Thai university transcripts.
You transcribe exactly what is printed into JSON format adhering strictly to the provided JSON Schema.
Extract all header_detail fields (student_id, prename, name, uni_name, dates, etc.) and footer_detail without skipping."""

EXTRACT_PROMPT = """ต่อไปนี้คือข้อความที่ถอดจากใบแสดงผลการศึกษา (transcript) ของสถาบันในประเทศไทย
จงแปลงเป็น JSON ตาม schema ที่กำหนดอย่างเคร่งครัด

=== กติกาสำคัญ ===

[1] เกรด (grade_earn): ต้องเป็นตัวอักษรมาตรฐานเท่านั้น เช่น A, B+, B, C+, C, D+, D, F, W, S, U
    *** ห้ามใส่เครื่องหมายลบ หรือขีดต่อท้าย เช่น F- หรือ C- ให้แปลงเป็น F หรือ C เท่านั้น ***

[2] ข้อมูลส่วนหัว (header_detail): ถอดมาให้ครบทุกช่อง ได้แก่
    - student_id: รหัสประจำตัวนักศึกษา (ตัวเลข 8 หลัก)
    - prename: คำนำหน้านาม (เช่น นาย, นางสาว, นายแพทย์)
    - name: ชื่อ-นามสกุล
    - date_of_birth, admis_date, grad_date: หากมีวันที่ ให้แปลงเป็นรูปแบบ YYYY-MM-DD
    - grad_reason: เหตุผลการสำเร็จการศึกษา หรือสถานะ (เช่น N/A (พ้นสภาพ 1 / 2562))

[3] ชื่อ Key บังคับในระดับภาคการศึกษาและรายวิชา:
    - year : ปีการศึกษา พ.ศ. (ตัวเลข เช่น 2561)
    - sem_num : ภาคเรียน (ตัวเลข 1, 2, หรือ 3)
    - subject : รายการวิชา (Array ของ Object)
    - subject_id : รหัสวิชา (ตัวเลข 8 หลัก)
    - subject_name : ชื่อวิชา
    - credit : หน่วยกิต (ตัวเลข)
    - grade_earn : เกรดที่ได้

[4] อ่านตารางทีละแถว จากบนลงล่าง และคัดลอกตัวเลข GPA/GPS ตามที่พิมพ์ไว้ ห้ามคำนวณเอง

=== ตัวอย่างโครงสร้าง JSON ที่ต้องการ ===
{{
  "header_detail": {{
    "student_id": "71010001",
    "prename": "นาย",
    "name": "สมชาย ใจดี"
  }},
  "transcript_detail": {{
    "semesters": [
      {{
        "year": 2561,
        "sem_num": 1,
        "GPA": 3.50,
        "GPS": 3.50,
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
  }}
}}

=== ข้อความจากเอกสาร ===
{document_text}

=== สิ้นสุดข้อความ ===
ตอบเป็น JSON เท่านั้น"""

def clean_transcript_json(obj: Any) -> Any:
    """ทำความสะอาดข้อมูลเกรดและรูปแบบวันที่หลังสกัดจาก LLM"""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k == "grade_earn" and isinstance(v, str):
                cleaned[k] = re.sub(r"-", "", v).strip()
            else:
                cleaned[k] = clean_transcript_json(v)
        return cleaned
    elif isinstance(obj, list):
        return [clean_transcript_json(item) for item in obj]
    return obj


TYPHOON_PROMPT = (
    "Extract all text content and structure into plain markdown format. "
    "Do NOT use HTML tags, Tailwind classes, or <table> tags. "
    "Use standard markdown text layout."
)


# ==============================================================================
#  ส่วนที่ 5 — เรียกใช้ Ollama
# ==============================================================================

def ollama_chat(model: str, messages: list[dict], *, fmt: dict | None = None,
                images: list[bytes] | None = None, temperature: float = 0.0,
                retries: int = 2) -> str:
    requests = _need("requests")

    if images:
        messages = [dict(m) for m in messages]
        messages[-1]["images"] = [base64.b64encode(im).decode() for im in images]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
    }
    if fmt is not None:
        payload["format"] = fmt

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
    if not text:
        return {}
    
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL)
    t = re.sub(r"```(?:json)?", "", t, flags=re.IGNORECASE)
    t = t.strip("` \n\r\t")

    parsed = {}
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            pass

    if not parsed:
        try:
            starts = [p for p in (t.find("{"), t.find("[")) if p != -1]
            if starts:
                obj, _ = json.JSONDecoder().raw_decode(t[min(starts):])
                if isinstance(obj, dict):
                    parsed = obj
        except Exception:
            pass

    return clean_transcript_json(parsed)


# ==============================================================================
#  ส่วนที่ 6 — PIPELINE A : Tesseract (baseline จาก Lab 5-6)
# ==============================================================================

def pipeline_baseline(pages: list[bytes]) -> dict:
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        gray_resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        bw = cv2.adaptiveThreshold(
            gray_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        txt = pytesseract.image_to_string(bw, lang="tha+eng", config="--psm 6")
        full_text += f"\n=== หน้า {i + 1} ===\n{txt}"
        print(f"      Tesseract หน้า {i + 1}: {len(txt):,} ตัวอักษร")

    return _rule_based_parse(full_text)


def _rule_based_parse(text: str) -> dict:
    out: dict = {
        "header_detail": {},
        "transcript_detail": {"semesters": []},
        "footer_detail": {},
    }

    m = re.search(r"รหัส\D{0,20}(\d{8})", text)
    if m:
        out["header_detail"]["student_id"] = m.group(1)

    m = re.search(r"(คณะ[^\n]{2,40})", text)
    if m:
        out["header_detail"]["faculty_name"] = m.group(1).strip()

    row_re = re.compile(
        r"(\d{8})\s*(.+?)\s*(\d)\s*([ABCDF][+]?|[WSUIPTผมีต])\b",
        re.IGNORECASE,
    )
    
    sem_re = re.compile(r"(?:ภาค|ภๅค|ภค|semester|sem)\D{0,15}([123])\D{0,20}(25\d{2})", re.IGNORECASE)

    current = None

    for line in text.splitlines():
        cleaned_line = re.sub(r"[|—\-_]+", " ", line).strip()
        if not cleaned_line:
            continue

        sm = sem_re.search(cleaned_line)
        if sm:
            if current and current.get("subject"):
                out["transcript_detail"]["semesters"].append(current)
            
            current = {
                "year": int(sm.group(2)),
                "sem_num": int(sm.group(1)),
                "GPA": None,
                "GPS": None,
                "subject": []
            }
            continue

        for rm in row_re.finditer(cleaned_line):
            if current is not None:
                current["subject"].append({
                    "subject_id": rm.group(1),
                    "subject_name": rm.group(2).strip(),
                    "type": None,
                    "credit": int(rm.group(3)),
                    "grade_earn": rm.group(4).upper(),
                })

    if current and current.get("subject"):
        out["transcript_detail"]["semesters"].append(current)

    return out


# ==============================================================================
#  ส่วนที่ 7 — PIPELINE B : Typhoon-OCR -> text LLM
# ==============================================================================

def pipeline_vlm(pages: list[bytes], save_md: Path | None = None) -> dict:
    md_parts: list[str] = []
    for i, png in enumerate(pages):
        print(f"    [ขั้น 1/2] Typhoon-OCR อ่านหน้า {i + 1}/{len(pages)}...")
        md = ollama_chat(
            MODEL_OCR,
            [{"role": "user", "content": TYPHOON_PROMPT}],
            images=[png],
            temperature=0.1,
        )
        md_parts.append(f"\n=== หน้า {i + 1} ===\n{md}")

    document_text = "\n".join(md_parts)

    if save_md:
        save_md.write_text(document_text, encoding="utf-8")
        print(f"    บันทึก Markdown กลางทาง: {save_md}")

    print(f"    [ขั้น 2/2] {MODEL_TEXT} จัดรูปเป็น JSON...")
    raw = ollama_chat(
        MODEL_TEXT,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACT_PROMPT.format(document_text=document_text[:12000])},
        ],
        fmt=TRANSCRIPT_SCHEMA,
    )
    return parse_json(raw)


# ==============================================================================
#  ส่วนที่ 9 — ตรวจสอบความสอดคล้องภายใน (self-verification)
# ==============================================================================

VALID_GRADES = {
    "a", "b+", "b", "c+", "c", "d+", "d", "f",
    "w", "s", "u", "i", "p", "t", "ผ", "ม",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def verify_internal(data: dict) -> dict:
    issues: list[str] = []
    td = data.get("transcript_detail") or {}
    sems = td.get("semesters") or []
    hd = data.get("header_detail") or {}

    sid = M.normalize(hd.get("student_id"), "strict")
    if sid and not re.fullmatch(r"\d{8}", sid):
        issues.append(f"รหัสนักศึกษาผิดรูปแบบ: {hd.get('student_id')!r} (ต้องเป็นตัวเลข 8 หลัก)")
    for f in ("date_of_birth", "admis_date", "grad_date"):
        v = hd.get(f)
        if v and not DATE_RE.match(str(v)):
            issues.append(f"{f} ผิดรูปแบบ: {v!r} (ต้องเป็น YYYY-MM-DD)")

    seen_terms: set[tuple] = set()
    for s in sems:
        y, n = s.get("year"), s.get("sem_num")
        if not isinstance(y, int) or not (2500 <= y <= 2600):
            issues.append(f"ปีการศึกษาผิดปกติ: {y!r} (คาดว่าเป็น พ.ศ. เช่น 2561)")
        if n not in (1, 2, 3):
            issues.append(f"ภาคการศึกษาผิดปกติ: {n!r} (ต้องเป็น 1, 2 หรือ 3)")

        key = (y, n)
        if key in seen_terms:
            issues.append(f"ภาค {y}/{n} ปรากฏซ้ำมากกว่าหนึ่งครั้ง")
        seen_terms.add(key)

    order = [(s.get("year"), s.get("sem_num")) for s in sems
             if isinstance(s.get("year"), int) and isinstance(s.get("sem_num"), int)]
    if order != sorted(order):
        issues.append("ลำดับภาคการศึกษาไม่เรียงจากเก่าไปใหม่")

    n_subj_total = 0
    n_grade_null = 0

    for s in sems:
        tag = f"{s.get('year')}/{s.get('sem_num')}"
        subs = s.get("subject") or []
        n_subj_total += len(subs)

        if len(subs) == 0:
            issues.append(f"ภาค {tag} ไม่มีรายวิชาเลย")
        elif len(subs) > 12:
            issues.append(f"ภาค {tag} มีถึง {len(subs)} วิชา")

        seen_codes: set[str] = set()
        term_credits = 0

        for sub in subs:
            code = M.normalize(sub.get("subject_id"), "strict")
            g = M.normalize(sub.get("grade_earn"), "strict")
            c = sub.get("credit")

            if not re.fullmatch(r"\d{8}", code):
                issues.append(f"[{tag}] รหัสวิชาผิดรูปแบบ: {sub.get('subject_id')!r}")

            if not g:
                n_grade_null += 1
            elif g not in VALID_GRADES:
                issues.append(f"[{tag}] เกรดไม่ถูกต้อง {sub.get('grade_earn')!r} ที่วิชา {sub.get('subject_id')}")

            if not isinstance(c, int):
                issues.append(f"[{tag}] หน่วยกิตไม่ใช่จำนวนเต็ม: {c!r} ที่วิชา {sub.get('subject_id')}")
            elif not (1 <= c <= 9):
                issues.append(f"[{tag}] หน่วยกิตผิดปกติ: {c} ที่วิชา {sub.get('subject_id')}")
            else:
                term_credits += c

            if code and code in seen_codes:
                issues.append(f"[{tag}] รหัสวิชา {code} ปรากฏซ้ำในภาคเดียวกัน")
            seen_codes.add(code)

        if term_credits > 30:
            issues.append(f"ภาค {tag} มีหน่วยกิตรวม {term_credits}")

    return {
        "ok": len(issues) == 0,
        "n_semesters": len(sems),
        "n_subjects": n_subj_total,
        "n_grade_null": n_grade_null,
        "issues": issues,
    }


# ==============================================================================
#  ส่วนที่ 10 — ประเมินผลเทียบกับ GROUND TRUTH
# ==============================================================================

def evaluate(pred: dict, gt: dict) -> tuple[dict, dict]:
    S = M.FieldStat
    stats: dict[str, M.FieldStat] = {
        "student_id":   S("รหัสนักศึกษา ⭐"),
        "person":       S("คำนำหน้า/ชื่อ ⭐"),
        "institution":  S("สถาบัน/ที่อยู่"),
        "faculty":      S("คณะ/หลักสูตร"),
        "dates":        S("วันที่"),
        "status":       S("สถานะ/เกียรตินิยม"),
        "sem_meta":     S("ปี/ภาค"),
        "gpa":          S("GPA/GPS"),
        "sem_misc":     S("หมายเหตุรายภาค"),
        "subject_id":   S("รหัสวิชา"),
        "subject_name": S("ชื่อวิชา"),
        "subject_type": S("ประเภทวิชา"),
        "credit":       S("หน่วยกิต"),
        "grade":        S("เกรด ⭐"),
        "summary":      S("สรุปผลการศึกษา"),
        "footer":       S("ผู้รับรองท้ายเอกสาร"),
    }

    gh = gt.get("header_detail") or {}
    ph = pred.get("header_detail") or {}

    stats["student_id"].add(gh.get("student_id"), ph.get("student_id"), "student_id")
    for f in ("prename", "name"):
        stats["person"].add(gh.get(f), ph.get(f), f, track_wer=False)

    for f in ("uni_name", "uni_address"):
        stats["institution"].add(gh.get(f), ph.get(f), f, track_wer=False)

    for f in ("faculty_name", "degree", "program", "major"):
        stats["faculty"].add(gh.get(f), ph.get(f), f, track_wer=False)
    for f in ("date_of_birth", "admis_date", "grad_date"):
        stats["dates"].add(gh.get(f), ph.get(f), f, track_wer=False)

    for f in ("grad_reason", "honor"):
        stats["status"].add(gh.get(f), ph.get(f), f, track_wer=False)

    gsems = (gt.get("transcript_detail") or {}).get("semesters") or []
    psems = (pred.get("transcript_detail") or {}).get("semesters") or []

    sem_align = M.align_by_key(
        gsems, psems,
        key_fn=lambda s: f"{s.get('year')}/{s.get('sem_num')}",
    )

    subj_align_total = M.AlignResult()

    for g_sem, p_sem in sem_align.matched:
        stats["sem_meta"].add(g_sem.get("year"), p_sem.get("year"), "year", track_wer=False)
        stats["sem_meta"].add(g_sem.get("sem_num"), p_sem.get("sem_num"), "sem_num", track_wer=False)
        tag = f"{g_sem.get('year')}/{g_sem.get('sem_num')}"
        stats["gpa"].add(g_sem.get("GPA"), p_sem.get("GPA"), f"GPA@{tag}", track_wer=False)
        stats["gpa"].add(g_sem.get("GPS"), p_sem.get("GPS"), f"GPS@{tag}", track_wer=False)
        stats["sem_misc"].add(g_sem.get("pass_reason"), p_sem.get("pass_reason"), f"pass_reason@{tag}", track_wer=False)

        sa = M.align_by_key(
            g_sem.get("subject") or [], p_sem.get("subject") or [],
            key_fn=lambda s: M.normalize(s.get("subject_id"), "strict"),
        )
        subj_align_total.matched.extend(sa.matched)
        subj_align_total.missed.extend(sa.missed)
        subj_align_total.spurious.extend(sa.spurious)

        for g_sub, p_sub in sa.matched:
            key = f"{tag} {g_sub.get('subject_id')}"
            stats["subject_id"].add(g_sub.get("subject_id"), p_sub.get("subject_id"), key, track_wer=False)
            stats["subject_name"].add(g_sub.get("subject_name"), p_sub.get("subject_name"), key, track_wer=False)
            stats["subject_type"].add(g_sub.get("type"), p_sub.get("type"), key, track_wer=False)
            stats["credit"].add(g_sub.get("credit"), p_sub.get("credit"), key, track_wer=False)
            stats["grade"].add(g_sub.get("grade_earn"), p_sub.get("grade_earn"), key, track_wer=False)

        for g_sub in sa.missed:
            k = f"{tag} {g_sub.get('subject_id')} [ตกแถว]"
            stats["subject_id"].add(g_sub.get("subject_id"), "", k, track_wer=False)
            stats["subject_name"].add(g_sub.get("subject_name"), "", k, track_wer=False)
            stats["subject_type"].add(g_sub.get("type"), "", k, track_wer=False)
            stats["credit"].add(g_sub.get("credit"), "", k, track_wer=False)
            stats["grade"].add(g_sub.get("grade_earn"), "", k, track_wer=False)

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

    gtd = gt.get("transcript_detail") or {}
    ptd = pred.get("transcript_detail") or {}
    for f in ("total_credits_earned", "cumulative_gpa",
              "master_comprehensive", "master_thesis", "master_qualify"):
        stats["summary"].add(gtd.get(f), ptd.get(f), f, track_wer=False)

    gf = gt.get("footer_detail") or {}
    pf = pred.get("footer_detail") or {}
    stats["footer"].add(gf.get("updated_at"), pf.get("updated_at"), "updated_at", track_wer=False)
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
    ap.add_argument("-o", "--out", default="output", help="โฟลเดอร์ผลลัพธ์")
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