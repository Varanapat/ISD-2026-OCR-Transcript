import re
from .schemas import OCRPageResult

THAI_MONTHS = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3, "เมษายน": 4,
    "พฤษภาคม": 5, "มิถุนายน": 6, "กรกฎาคม": 7, "สิงหาคม": 8,
    "กันยายน": 9, "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
}
PRENAMES = ["นางสาว", "นาย", "นาง", "ดร.", "ผศ.ดร.", "ผศ.", "รศ.ดร.", "รศ.", "ศ.ดร.", "ศ."]

# SUBJECT_ID_RE = re.compile(r"^(\d{6,9})\s*(.*)$")
SUBJECT_ID_RE = re.compile(r"^(\d{8})\s*(.*)$")
CREDIT_RE = re.compile(r"^\d{1,2}$")
# A whole subject row on one line: id, name, credit, grade. The name is non-greedy so
# the trailing credit and grade win the ambiguity rather than being eaten by the name.
SUBJECT_ROW_RE = re.compile(r"^(\d{8})\s+(.+?)\s+(\d{1,2})\s+([A-Za-z0-9][+-]?)\s*$")
# The last column is always a grade, so a digit there is a misread letter, not a number.
# Tesseract in particular returns "0+" for "D+" and "8+" for "B+".
GRADE_DIGIT_FIX = {"0": "D", "8": "B", "5": "S", "1": "I"}
VALID_GRADES = set("ABCDFWISUP")
GRADE_RE = re.compile(r"^(?:[A-Za-z]{1}[+-]?|W|I|S|U|P)$")
# Matched against _norm()-ed text, so the anchor is written without its tone mark
# ("ที่" -> "ที"). Anything non-numeric is allowed between the two numbers because OCR
# routinely mangles the "ปีการศึกษา" label while still reading both numbers correctly;
# this also recovers the case where the spaces are lost ("ที 12561" -> 1, 2561).
SEMESTER_HEADER_RE = re.compile(r"ภาคการศึกษาที\s*(\d+)\D{0,20}?(\d{4})")

# Column headers of the grade table, used to find where the subject rows begin.
TABLE_HEADER_WORDS = ("รายวิชา", "หน่วยกิต", "เกรด")
SEM_GPA_RE = re.compile(r"ประจำภาคการศึกษา\s*[:：]?\s*([\d.]+)")

_TONE_MARK_RE = re.compile(r"[่-๋]")
_WHITESPACE_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    """Normalise a line before matching Thai keywords against it.

    Two kinds of OCR noise break otherwise-correct reads: missing or extra tone marks,
    and spaces inserted inside a word (Tesseract returns "ภาคกา รศึกษาที" for
    "ภาคการศึกษาที"). Thai does not use spaces within a word, so dropping both is safe
    here and rescues anchors that are only cosmetically damaged.
    """
    # SARA AM (ำ) is one code point, but OCR often returns it decomposed as
    # NIKHAHIT + SARA AA (ํ + า), which compares unequal. Unicode defines no canonical
    # decomposition for it, so NFC will not merge them -- do it here.
    s = s.replace("\u0e4d\u0e32", "\u0e33")
    return _WHITESPACE_RE.sub("", _TONE_MARK_RE.sub("", s))


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


def _extract_int_near(lines: list[str], idx: int) -> str | None:
    """Like _extract_number_near but skips decimals (e.g. a nearby GPA) so it never
    mistakes a fractional number for an integer count."""
    for offset in (0, 1, 2):
        if idx + offset >= len(lines):
            break
        match = re.search(r"(?<!\.)\b\d+\b(?!\.\d)", lines[idx + offset])
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


def _fix_grade(raw: str) -> str | None:
    """Repair a grade whose letter came back as a lookalike digit; None if not a grade."""
    letter, suffix = raw[0], raw[1:]
    letter = GRADE_DIGIT_FIX.get(letter, letter).upper()
    return letter + suffix if letter in VALID_GRADES else None


def _parse_subjects(block_lines: list[str]) -> list[dict]:
    subjects: list[dict] = []
    pending: dict | None = None
    for line in block_lines:
        # Engines disagree on layout: PaddleOCR emits a subject row as four separate
        # lines, Tesseract as one ("13006006 พีชคณิตเชิงเส้น  3  D+"). Try the whole
        # row first so neither shape is lost, then fall back to the four-line form.
        whole = SUBJECT_ROW_RE.match(line)
        if whole:
            grade = _fix_grade(whole.group(4))
            if grade is None:
                whole = None
        if whole:
            if pending:
                subjects.append(pending)
            subjects.append({
                "subject_id": whole.group(1),
                "subject_name": whole.group(2).strip() or None,
                "type": None,
                "credit": int(whole.group(3)),
                "grade_earn": grade.lower(),
            })
            pending = None
            continue
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


def _subject_table_start(lines: list[str]) -> int | None:
    """Index of the grade table's header row, or None when it cannot be located."""
    for i, line in enumerate(lines):
        norm_line = _norm(line)
        # The totals row repeats "หน่วยกิต" but sits AFTER the last subject, so treating
        # it as the table's start would leave nothing to scan.
        if _norm("จำนวน") in norm_line or _norm("ทั้งหมด") in norm_line:
            continue
        if any(_norm(word) in norm_line for word in TABLE_HEADER_WORDS):
            return i
    return None


def _extract_transcript(lines: list[str]) -> dict:
    transcript = {
        "semesters": [], "master_comprehensive": None, "master_thesis": None,
        "master_qualify": None, "total_credits_earned": None, "cumulative_gpa": None,
    }

    header_positions = [(i, m) for i, l in enumerate(lines) if (m := SEMESTER_HEADER_RE.search(_norm(l)))]
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

    # Subjects can end up outside every semester block when OCR mangles a
    # "ภาคการศึกษาที่ ..." line. Discarding them throws away a whole page over one bad
    # line, so they are kept under a semester with no year/sem_num instead. The scan
    # starts at the grade table's header row so that stray 8-digit numbers higher up
    # the page -- the student id above all -- cannot be read as subject ids.
    table_start = _subject_table_start(lines)
    if table_start is not None:
        orphan_end = header_positions[0][0] if header_positions else end_idx
        orphan_subjects = _parse_subjects(lines[table_start + 1: orphan_end])
        if orphan_subjects:
            transcript["semesters"].insert(0, {
                "year": None, "sem_num": None, "GPA": None, "GPS": None,
                "pass_reason": None, "subject": orphan_subjects,
            })

    for i, line in enumerate(lines):
        norm_line = _norm(line)
        if _norm("จำนวนหน่วยกิตที่สอบได้ทั้งหมด") in norm_line:
            num = _extract_int_near(lines, i + 1)
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


def merge_extracted_fields(primary: dict, *fallbacks: dict) -> dict:
    """Fill the blanks in `primary` from the later reads, in the order given.

    A field-level merge, not a line-level one. Merging two engines' LINES fails on this
    document: PaddleOCR emits a subject row as four lines and Tesseract as one, so any
    line-level mix hands the parser a row shape neither engine produces, and PaddleOCR's
    boxes are not even in the input image's coordinate frame unless unwarping is off.
    Merging the extracted fields sidesteps both problems -- each read is parsed by the
    layout that produced it, and only the gaps are filled.

    A value the primary read is kept even when a fallback disagrees: the primary is the
    stronger reader, and a blank is the only signal that it failed. Lists are taken
    whole from the first read that has any items, never element by element, because two
    reads can disagree on how many subjects a semester has.
    """
    merged = primary
    for fallback in fallbacks:
        merged = _fill_blanks(merged, fallback)
    return merged


def _fill_blanks(primary, fallback):
    if isinstance(primary, dict):
        return {k: _fill_blanks(v, (fallback or {}).get(k)) for k, v in primary.items()}
    if isinstance(primary, list):
        return primary if primary else (fallback if isinstance(fallback, list) else primary)
    return primary if primary is not None else fallback
