import re
from .schemas import OCRPageResult

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
    "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
    "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
PRENAMES = ["นางสาว", "นาย", "นาง", "ดร.", "ผศ.ดร.", "ผศ.", "รศ.ดร.", "รศ.", "ศ.ดร.", "ศ."]

SUBJECT_ID_RE = re.compile(r"^(\d{6,9})\s*(.*)$")
CREDIT_RE = re.compile(r"^\d{1,2}$")
GRADE_RE = re.compile(r"^(?:[A-Za-z]{1,2}[+-]?|W|I|S|U|P)$")
SEMESTER_HEADER_RE = re.compile(r"ภาคการศึกษาที่\s*(\d+)\s*ปีการศึกษา\s*(\d{4})")
SEM_GPA_RE = re.compile(r"ประจำภาคการศึกษา\s*[:：]?\s*([\d.]+)")

_TONE_MARK_RE = re.compile(r"[่-๋]")


def _norm(s: str) -> str:
    """Strip Thai tone marks so OCR noise (missing/extra tone marks) doesn't break keyword matching."""
    return _TONE_MARK_RE.sub("", s)


def _flatten_lines(pages: list[OCRPageResult]) -> list[str]:
    lines: list[str] = []
    for page in pages:
        for line in page.lines:
            text = line.text.strip()
            if text:
                lines.append(text)
    return lines


def _parse_thai_date(text: str) -> str | None:
    match = re.search(r"(\d{1,2})\s+(\S+)\s+(\d{4})", text)
    if not match:
        return None
    day, month_name, year_be = match.groups()
    month = THAI_MONTHS.get(month_name)
    if month is None:
        return None
    year = int(year_be) - 543
    return f"{year:04d}-{month:02d}-{int(day):02d}"


def _extract_number_near(lines: list[str], idx: int) -> str | None:
    for offset in (0, 1, 2):
        if idx + offset >= len(lines):
            break
        match = re.search(r"[\d]+(?:\.\d+)?", lines[idx + offset])
        if match:
            return match.group(0)
    return None


def _extract_header(lines: list[str], title_idx: int) -> dict:
    header = {
        "uni_name": None, "uni_address": None, "student_id": None,
        "faculty_name": None, "prename": None, "name": None,
        "date_of_birth": None, "admis_date": None, "grad_date": None,
        "grad_reason": None, "degree": None, "major": None,
        "program": None, "honor": 0,
    }

    top_candidates = [l for l in lines[:title_idx] if len(l) >= 3]
    for line in top_candidates:
        if re.search(r"\d{5}\s*$", line):
            header["uni_address"] = line
        else:
            if header["uni_name"] is None or len(line) > len(header["uni_name"]):
                header["uni_name"] = line

    for line in lines:
        if header["faculty_name"] is None and line.startswith("คณะ"):
            header["faculty_name"] = line
            continue

        m = re.match(r"^(?:ช|ซ)ื[่]?อ-สกุล\s*(.+)$", line)
        if m:
            rest = m.group(1).strip()
            header["prename"] = next((p for p in PRENAMES if rest.startswith(p)), None)
            header["name"] = rest[len(header["prename"]):].strip() if header["prename"] else rest
            continue

        m = re.match(r"^รหัสประจำตัวนักศึกษา\s*(\d+)", line)
        if m:
            header["student_id"] = m.group(1)
            continue

        m = re.match(r"^วันเดือนปีเกิด\s*(.+)$", line)
        if m:
            header["date_of_birth"] = _parse_thai_date(m.group(1))
            continue

        m = re.match(r"^วันที[่]?เข[้]?าศึกษา\s*(.+)$", line)
        if m:
            header["admis_date"] = _parse_thai_date(m.group(1))
            continue

        m = re.match(r"^วันที[่]?สำเร็จการศึกษา\s*(.+)$", line)
        if m:
            rest = m.group(1).strip()
            na_match = re.search(r"N/?A\s*\(([^)]*)\)", rest, re.IGNORECASE)
            if na_match:
                header["grad_date"] = "0000-00-00"
                reason = re.sub(r"\s+", "", na_match.group(1))
                header["grad_reason"] = f"n/a({reason})"
            else:
                header["grad_date"] = _parse_thai_date(rest)
            continue

        m = re.match(r"^(?:ช|ซ)ื[่]?อปริญญา\s*(.+)$", line)
        if m:
            header["degree"] = m.group(1).strip()
            continue

        m = re.match(r"^หลักสูตร\s+(.+)$", line)
        if m:
            header["program"] = m.group(1).strip()
            continue

        if re.search(r"เกียรตินิยมอันดับ\s*1", line):
            header["honor"] = 1
        elif re.search(r"เกียรตินิยมอันดับ\s*2", line):
            header["honor"] = 2

    return header


def _parse_subjects(block_lines: list[str]) -> list[dict]:
    subjects: list[dict] = []
    pending: dict | None = None
    for line in block_lines:
        m = SUBJECT_ID_RE.match(line)
        if m:
            if pending:
                subjects.append(pending)
            pending = {
                "subject_id": m.group(1),
                "subject_name": m.group(2).strip() or None,
                "type": None, "credit": None, "grade_earn": None,
            }
            continue
        if pending is None:
            continue
        if pending["subject_name"] is None and not CREDIT_RE.match(line) and not GRADE_RE.match(line):
            pending["subject_name"] = line
        elif pending["credit"] is None and CREDIT_RE.match(line):
            pending["credit"] = int(line)
        elif pending["credit"] is not None and pending["grade_earn"] is None and GRADE_RE.match(line):
            pending["grade_earn"] = line.lower()
    if pending:
        subjects.append(pending)
    return subjects


def _extract_transcript(lines: list[str]) -> dict:
    transcript = {
        "semesters": [], "master_comprehensive": None, "master_thesis": None,
        "master_qualify": None, "total_credits_earned": None, "cumulative_gpa": None,
    }

    header_positions = [(i, m) for i, l in enumerate(lines) if (m := SEMESTER_HEADER_RE.search(l))]
    end_idx = next(
        (i for i, l in enumerate(lines) if _norm("จำนวนหน่วยกิตที่สอบได้ทั้งหมด") in _norm(l)),
        len(lines),
    )
    boundaries = [pos for pos, _ in header_positions] + [end_idx]

    for k, (start, match) in enumerate(header_positions):
        block = lines[start + 1: boundaries[k + 1]]
        gpa_line = next((l for l in block if "ประจำภาคการศึกษา" in _norm(l)), None)
        cum_line = next(
            (
                l for l in block
                if _norm(l).startswith(_norm("คะแนนเฉลี่ย"))
                and "ประจำภาค" not in _norm(l)
                and "สะสม" not in _norm(l)
            ),
            None,
        )
        gps_val = SEM_GPA_RE.search(_norm(gpa_line)).group(1) if gpa_line and SEM_GPA_RE.search(_norm(gpa_line)) else None
        gpa_val = re.search(r"[\d.]+", cum_line).group(0) if cum_line and re.search(r"[\d.]+", cum_line) else None

        transcript["semesters"].append({
            "year": int(match.group(2)),
            "sem_num": int(match.group(1)),
            "GPA": gpa_val,
            "GPS": gps_val,
            "pass_reason": None,
            "subject": _parse_subjects(block),
        })

    for i, line in enumerate(lines):
        norm_line = _norm(line)
        if _norm("จำนวนหน่วยกิตที่สอบได้ทั้งหมด") in norm_line:
            num = _extract_number_near(lines, i + 1)
            transcript["total_credits_earned"] = int(num) if num else None
        if _norm("คะแนนเฉลี่ยสะสม") in norm_line:
            num = _extract_number_near(lines, i + 1)
            transcript["cumulative_gpa"] = num

    return transcript


def _extract_footer(lines: list[str]) -> dict:
    footer = {"updated_at": None, "by": {"by_signature": None, "by_position": None, "by_reg": None}}
    for line in lines:
        m = re.match(r"^วันที[่]?ออกเอกสาร\s*(.+)$", line)
        if m:
            footer["updated_at"] = _parse_thai_date(m.group(1))
            continue
        m = re.match(r"^\(([^)]+)\)$", line)
        if m and footer["by"]["by_signature"] is None:
            footer["by"]["by_signature"] = m.group(1).strip()
            continue
        if _norm("ผู้อำนวยการ") in _norm(line):
            footer["by"]["by_position"] = line.strip()
            continue
        if "สำนักทะเบียน" in line:
            footer["by"]["by_reg"] = line.strip()
    return footer


def _compact(value):
    """Match the ground-truth convention of whitespace-free, lowercased string values."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\s+", "", value).lower()
    return value


def extract_transcript_fields(pages: list[OCRPageResult]) -> dict:
    lines = _flatten_lines(pages)
    title_idx = next((i for i, l in enumerate(lines) if "ใบแสดงผลการศึกษา" in l), len(lines))
    return _compact({
        "header_detail": _extract_header(lines, title_idx),
        "transcript_detail": _extract_transcript(lines),
        "footer_detail": _extract_footer(lines),
    })
