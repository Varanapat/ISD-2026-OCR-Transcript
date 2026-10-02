import json
import re
from pathlib import Path
from typing import Any
from rapidfuzz import fuzz


EN_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

TH_MONTHS = {
    "มกราคม": 1,
    "กุมภาพันธ์": 2,
    "มีนาคม": 3,
    "เมษายน": 4,
    "พฤษภาคม": 5,
    "มิถุนายน": 6,
    "กรกฎาคม": 7,
    "สิงหาคม": 8,
    "กันยายน": 9,
    "ตุลาคม": 10,
    "พฤศจิกายน": 11,
    "ธันวาคม": 12,
}

# "1st Semester, Academic Year 2019" (set 1) or "1st Semester , 2021" (set G).
# OCR often reads "1st" as "lst" / "Ist".
SEMESTER_RE = re.compile(r"(?i)\b(1st|lst|ist|2nd|3rd|summer)\s+semester\s*,?\s*(?:academic\s+year\s*)?(\d{4})")
SEMESTER_NUM = {"1st": 1, "lst": 1, "ist": 1, "2nd": 2, "3rd": 3, "summer": 0}
# ---- Thai matching helpers -------------------------------------------------------------
# OCR often drops or garbles Thai vowels above/below the line and tone marks, and reads
# ช as ซ. Labels are therefore compared on a "skeleton" without those marks.
TH_MARKS = set("\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e")
TH_CHAR_MAP = {"ซ": "ช", "ำ": "า"}
TH_MARKERS = ("คะแนนเฉลย", "จานวนหนวยกต", "สนสดการแสดง")
TH_PRENAMES = ("นางสาว", "นาง", "นาย", "ว่าที่ร้อยตรี")
TH_HEADER_LABELS = {
    "name": "ชื่อ-สกุล",
    "student_id": "รหัสประจำตัวนักศึกษา",
    "date_of_birth": "วันเดือนปีเกิด",
    "admis_date": "วันที่เข้าศึกษา",
    "degree": "ชื่อปริญญา",
    "grad": "วันที่สำเร็จการศึกษา",
    "program": "หลักสูตร",
    "major": "สาขาวิชา",
}

# Type column (set G): Cr / Nc / Ad. OCR reads "Cr" as "Gr", "0", "๐" and "Nc" as "Ne".
TYPE_VALUES = {"cr": "cr", "gr": "cr", "0": "cr", "๐": "cr", "o": "cr", "nc": "nc", "ne": "nc", "ad": "ad"}
# Digits/Thai letters OCR produces in the grade column: C -> 0/6, B -> 8, S -> 5/ธ/ร/$.
GRADE_OCR_FIXES = {"0": "c", "6": "c", "8": "b", "5": "s", "ธ": "s", "ร": "s", "ธร": "s", "ss": "s", "$": "s"}

GRADE_RE = re.compile(r"^(?:[abcdf][+-]?|s|u|w|i|t\([abcdfs][+-]?\)|-)$", re.IGNORECASE)
SUBJECT_ID_RE = re.compile(r"^\d{8}\.?$")


def extract_transcript_from_file(path: str | Path, language: str | None = None) -> dict[str, Any]:
    payload = _load_ocr_payload(path)
    return extract_transcript(payload["text"], payload.get("lines", []), language=language)


def extract_transcript_from_ocr(ocr_result: dict[str, Any], language: str | None = None) -> dict[str, Any]:
    """Same as extract_transcript_from_file, for an OCR result already in memory (OCRDocumentResult.to_dict())."""
    lines = [line for page in ocr_result.get("pages", []) for line in page.get("lines", [])]
    return extract_transcript(ocr_result.get("text", ""), lines, language=language)


def extract_transcript(
    text: str,
    ocr_lines: list[dict[str, Any]] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    tokens = _clean_tokens(text.splitlines())
    language = language or _detect_language(text)
    is_thai = language.lower().startswith("th")

    header = _extract_header(tokens, is_thai=is_thai)
    if ocr_lines:
        admis_date = header.get("admis_date")
        if is_thai:  # Thai header is parsed from rows below; needed here for the admission year
            admis_date = _extract_header_th(_row_texts(ocr_lines)).get("admis_date")
        admit_year = int(admis_date[:4]) + 543 if admis_date and admis_date[:4].isdigit() else None
        semesters = _extract_subjects_from_boxes(ocr_lines, admit_year)
    else:
        # The student ID has 8 digits like a subject ID; exclude it so subjects do not shift by one row.
        not_subjects = {header["student_id"]} if header.get("student_id") else set()
        semesters = _extract_subjects_from_tokens(tokens, not_subjects)

    transcript = {
        "semesters": semesters,
        "master_comprehensive": None,
        "master_thesis": None,
        "master_qualify": None,
        "total_credits_earned": _to_int(_value_after_label(tokens, ["total", "credits", "earned"])),
        "cumulative_gpa": _clean_gpa(_value_after_label(tokens, ["cumulative", "gpa"])),
    }
    footer = _extract_footer(tokens, is_thai=is_thai)

    if is_thai and ocr_lines:
        # Thai labels are split into syllables by OCR (tesseract) and often lose vowels/tone
        # marks, so Thai documents are parsed per visual row with fuzzy label matching.
        rows = _row_texts(ocr_lines)
        header = _extract_header_th(rows)
        transcript.update(_extract_totals_th(rows))
        footer = _extract_footer_th(rows)

    return {
        "header_detail": header,
        "transcript_detail": transcript,
        "footer_detail": footer,
    }


def _load_ocr_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        lines = []
        for page in data.get("pages", []):
            lines.extend(page.get("lines", []))
        return {"text": data.get("text", ""), "lines": lines}

    return {"text": path.read_text(encoding="utf-8"), "lines": []}


def _detect_language(text: str) -> str:
    # OCR (tha+eng) puts a few stray Thai characters into English documents (<= 1% of letters);
    # Thai documents are > 90% Thai letters.
    thai_chars = len(re.findall(r"[\u0e00-\u0e7f]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    return "th" if thai_chars > 0.3 * (thai_chars + latin_chars) else "en"


def _clean_tokens(lines: list[str]) -> list[str]:
    tokens = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("--- Page"):
            continue
        tokens.append(value)
    return tokens


def _token_key(value: str) -> str:
    # "|" is a table rule that OCR attaches to neighbouring words (e.g. "3|", "|01016238").
    return re.sub(r"[\s:.,|]+", "", value).lower()


def _schema_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    value = value.replace("\u0e4d\u0e32", "\u0e33")  # ํ + า -> ำ (OCR writes sara am as two characters)
    value = re.sub(r"\s+", "", value)
    value = value.replace(".", "")
    # Disable GT-style word correction while measuring OCR extraction quality.
    # value = value.replace("kingmongkutis", "kingmongkut's")
    # value = value.replace("educationalservice", "educationservice")
    return value or None


def _signature_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    value = value.replace("\u0e4d\u0e32", "\u0e33")  # ํ + า -> ำ
    value = value.strip("()")
    value = re.sub(r"\s+", "", value)
    return value or None


def _subject_text(tokens: list[str]) -> str:
    return _schema_text("".join(tokens)) or ""


def _find_sequence(tokens: list[str], label: list[str], start: int = 0) -> int:
    label_keys = [_token_key(item) for item in label]
    token_keys = [_token_key(item) for item in tokens]
    for index in range(start, len(token_keys) - len(label_keys) + 1):
        if token_keys[index : index + len(label_keys)] == label_keys:
            return index
    return -1


def _slice_after_until(tokens: list[str], label: list[str], stops: list[list[str]]) -> list[str]:
    start = _find_sequence(tokens, label)
    if start < 0:
        return []
    start += len(label)
    end = len(tokens)
    for stop in stops:
        index = _find_sequence(tokens, stop, start)
        if index >= 0:
            end = min(end, index)
    return [item for item in tokens[start:end] if item not in {":"}]


def _value_after_label(tokens: list[str], label: list[str]) -> str | None:
    values = _slice_after_until(
        tokens,
        label,
        [
            ["gpa"],
            ["cumulative", "gpa"],
            ["end", "of", "transcript"],
            ["not", "valid"],
        ],
    )
    if not values:
        return None
    return " ".join(values[:3])


def _extract_header(tokens: list[str], is_thai: bool) -> dict[str, Any]:
    student_id = _first_match(tokens, r"^\d{8,12}$")
    name_tokens = _slice_after_until(tokens, ["name"], [["student", "id"], ["รหัส", "นักศึกษา"]])
    prename, name = _split_name(name_tokens, is_thai=is_thai)

    grad_tokens = _slice_after_until(tokens, ["date", "of", "graduation"], [["program"], ["หลักสูตร"]])
    grad_date = _parse_date(" ".join(grad_tokens))
    grad_reason = None if grad_date else _schema_text(" ".join(grad_tokens))

    return {
        "uni_name": _extract_uni_name(tokens, is_thai=is_thai),
        "uni_address": _extract_uni_address(tokens, is_thai=is_thai),
        "student_id": student_id,
        "faculty_name": _extract_faculty_name(tokens, is_thai=is_thai),
        "prename": _schema_text(prename),
        "name": _schema_text(name),
        "date_of_birth": _parse_date(" ".join(_slice_after_until(tokens, ["date", "of", "birth"], [["date", "of", "admission"]]))),
        "admis_date": _parse_date(" ".join(_slice_after_until(tokens, ["date", "of", "admission"], [["degree"]]))),
        "grad_date": grad_date or "0000-00-00",
        "grad_reason": grad_reason,
        "degree": _schema_text(" ".join(_slice_after_until(tokens, ["degree"], [["date", "of", "graduation"]]))),
        "major": None,
        "program": _schema_text(" ".join(_slice_after_until(tokens, ["program"], [["1st", "semester"], ["2nd", "semester"], ["ภาค"]]))),
        "honor": 0,
    }


def _extract_uni_name(tokens: list[str], is_thai: bool) -> str | None:
    if is_thai:
        title_index = _find_sequence(tokens, ["ใบ", "แสดง", "ผล"])
        return _schema_text(" ".join(tokens[:title_index if title_index > 0 else 1]))
    end = _find_sequence(tokens, ["chalongkrung"])
    if end < 0:
        end = _find_sequence(tokens, ["transcript"])
    return _schema_text(" ".join(tokens[:end])) if end > 0 else None


def _extract_uni_address(tokens: list[str], is_thai: bool) -> str | None:
    if is_thai:
        return None
    start = _find_sequence(tokens, ["chalongkrung"])
    end = _find_sequence(tokens, ["transcript"], start)
    return _schema_text(" ".join(tokens[start:end])) if start >= 0 and end > start else None


def _extract_faculty_name(tokens: list[str], is_thai: bool) -> str | None:
    if is_thai:
        value = _slice_after_until(tokens, ["คณะ"], [["ชื่อ"], ["รหัส"]])
        return _schema_text(" ".join(["คณะ", *value])) if value else None
    value = _slice_after_until(tokens, ["college", "of"], [["name"]])
    return _schema_text(" ".join(["college", "of", *value])) if value else None


def _split_name(tokens: list[str], is_thai: bool) -> tuple[str | None, str | None]:
    if not tokens:
        return None, None
    if is_thai:
        return tokens[0], " ".join(tokens[1:])
    if _token_key(tokens[0]) in {"mr", "miss", "mrs", "ms"}:
        return tokens[0], " ".join(tokens[1:])
    return None, " ".join(tokens)


def _extract_footer(tokens: list[str], is_thai: bool) -> dict[str, Any]:
    issued = _slice_after_until(tokens, ["date", "of", "issued"], [["not", "valid"], ["director"]])
    signature = _slice_after_until(tokens, ["without", "seal"], [["director"]])
    position_index = _find_sequence(tokens, ["director"])
    by_reg = tokens[position_index + 1 :] if position_index >= 0 else []

    return {
        "updated_at": _parse_date(" ".join(issued)),
        "by": {
            "by_signature": _signature_text(" ".join(signature)),
            "by_position": _schema_text("director") if not is_thai and position_index >= 0 else None,
            "by_reg": _schema_text(" ".join(by_reg)),
        },
    }


def _extract_subjects_from_boxes(ocr_lines: list[dict[str, Any]], admit_year: int | None = None) -> list[dict[str, Any]]:
    words = _page_words(ocr_lines)
    semesters = _semester_headers_from_boxes(words, admit_year)

    # Only 8-digit numbers inside the table are subject IDs. The student ID above the table
    # also has 8 digits; counting it shifted every subject by one row.
    table_top = semesters[0]["start_y"] - 30 if semesters else _table_header_y(words)
    subject_words = [
        word for word in words
        if SUBJECT_ID_RE.match(word["key"]) and _in_table(word) and (table_top is None or word["y"] > table_top)
    ]
    if not semesters and subject_words:
        first_course = next((word for word in subject_words if word["y"] > 700), subject_words[0])
        semesters = [{"year": None, "sem_num": None, "start_y": first_course["y"] - 30}]

    # Work on visual rows: Thai vowels/tone marks are separate boxes slightly above or below
    # the line, so fixed y windows cut them off or pull in marks from the next subject.
    rows = _rows(words)
    row_of = {id(word): index for index, row in enumerate(rows) for word in row}

    results = []
    for index, semester in enumerate(semesters):
        start_y = semester.get("start_y") or 0
        end_y = semesters[index + 1].get("start_y") if index + 1 < len(semesters) else 10**9
        ids = [word for word in subject_words if start_y <= word["y"] < end_y]
        semester_subjects = [_subject_from_rows(subject_id, rows, row_of[id(subject_id)]) for subject_id in ids]

        gps, gpa = _semester_scores(words, start_y, end_y)
        results.append(
            {
                "year": semester.get("year"),
                "sem_num": semester.get("sem_num"),
                "GPA": gpa,
                "GPS": gps,
                "pass_reason": None,
                "subject": semester_subjects,
            }
        )
    return results


def _page_words(ocr_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Words with positions, the two-column table unfolded into one column, in reading order."""
    words = [word for line in ocr_lines if str(line.get("text", "")).strip() for word in _split_line(_line_word(line))]
    pages = sorted({word["page"] for word in words})
    words = [w for page in pages for w in _unfold_table([word for word in words if word["page"] == page])]
    words.sort(key=lambda item: (item["page"], item["y"], item["x"]))
    return words


def _split_line(word: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a multi-word box (paddle, easyocr, doctr, surya return whole lines, e.g.
    "15006101 CONTEMPORARY SCIENCE") into words; each word's x is estimated from its
    character position in the line. Tesseract boxes are single words and stay as they are."""
    text = word["text"]
    parts = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    if len(parts) <= 1:
        return [word]
    width = (word["x1"] - word["x"]) / max(len(text), 1)
    words = []
    for start, part in parts:
        x0 = word["x"] + start * width
        x1 = x0 + len(part) * width
        words.append({**word, "text": part, "key": _token_key(part), "x": x0, "x1": x1, "cx": (x0 + x1) / 2})
    return words


def _unfold_table(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move the right half of the transcript table below the left half.

    Long transcripts continue in the right half of the table (subject | credit | grade twice
    side by side). Shifting the right half down by the table height (and the footer below it)
    turns the page into one column, so the rest of the code can read top to bottom.
    """
    area = _table_area(words)
    if area is None:
        return words
    top, bottom, mid = area
    in_table = [w for w in words if top <= w["cy"] < bottom]
    right = [w for w in in_table if _is_right_half(w, mid)]
    # Only unfold when the right half really has content (a subject ID or a semester heading).
    has_content = any(SUBJECT_ID_RE.match(w["key"]) for w in right) or any(
        _thai_semester("".join(x["text"] for x in row)) or "semester" in " ".join(x["key"] for x in row)
        for row in _rows(right)
    )
    if not has_content:
        return words

    height = bottom - top
    left_edge = min(w["x"] for w in in_table)
    shift_x = mid - left_edge
    moved = []
    for w in words:
        w = dict(w)
        if top <= w["cy"] < bottom and _is_right_half(w, mid):
            if mid and w["x"] - mid < 80:
                # The double rule in the middle of the table is often read into the subject ID
                # ("|01016238", "101016454", "0190594001"): keep its last 8 digits.
                digits = re.sub(r"\D", "", w["key"])
                if 8 <= len(digits) <= 10 and not SUBJECT_ID_RE.match(w["key"]):
                    w["key"] = w["text"] = digits[-8:]
            for key in ("x", "x1", "cx"):
                w[key] -= shift_x
            for key in ("y", "cy"):
                w[key] += height
        elif w["cy"] >= bottom:  # footer goes after the unfolded right half
            for key in ("y", "cy"):
                w[key] += height
        moved.append(w)
    return moved


def _is_right_half(word: dict[str, Any], mid: float) -> bool:
    region = word.get("region") or ""
    if region.startswith("table_col"):
        return word["_right_col"]
    return word["cx"] >= mid


def _table_area(words: list[dict[str, Any]]) -> tuple[float, float, float] | None:
    """(top, bottom, middle x) of the transcript table body, or None if not found."""
    table_words = [w for w in words if (w.get("region") or "").startswith("table_col")]
    if table_words:  # --layout: table columns are known exactly
        columns = max(int(w["region"].removeprefix("table_col")) for w in table_words)
        for w in words:
            region = w.get("region") or ""
            w["_right_col"] = region.startswith("table_col") and int(region.removeprefix("table_col")) > columns // 2
        top = min(w["cy"] for w in table_words)
        bottom = max(w["cy"] for w in table_words) + 1
        footer = [w["cy"] for w in words if (w.get("region") or "").startswith("footer")]
        bottom = max(bottom, min(footer) - 1) if footer else bottom
        right = [w["x"] for w in table_words if w["_right_col"]]
        return top, bottom, (min(right) - 15 if right else 0.0)

    rows = _rows(words)
    top = None
    for row in rows:
        text = " ".join(w["text"] for w in row)
        loose = _loose_th(text).lower()
        if ("รายวชา" in loose and ("หนวยกต" in loose or "เกรด" in loose)) or ("course" in loose and "grade" in loose):
            top = max(w["cy"] for w in row) + 15  # column headings: the table body starts below
            break
        # OCR may miss the column headings: start at the first semester heading / subject row instead.
        if SEMESTER_RE.search(text) or re.search(r"ภาคการศกษาท\D{0,2}\d|รายวชา", loose) or any(
            SUBJECT_ID_RE.match(w["key"]) for w in row[:1]
        ):
            top = min(w["y"] for w in row) - 5
            break
    if top is None:
        return None
    bottom = max(w["cy"] for w in words) + 1
    for row in rows:  # footer starts at "วันที่ออกเอกสาร" / "Date of Issued"
        loose = _loose_th("".join(w["text"] for w in row)).lower()
        if row[0]["cy"] > top and ("วนทออกเอกสาร" in loose or "dateofissue" in loose):
            bottom = min(w["cy"] for w in row) - 15
            break

    # Middle of the table: halfway across the page, or just left of the right-half subject IDs.
    mid = (min(w["x"] for w in words) + max(w["x1"] for w in words)) / 2
    ids = [w["x"] for w in words if top <= w["cy"] < bottom and SUBJECT_ID_RE.match(w["key"])]
    right_ids = [x for x in ids if x > min(ids) + 0.3 * (max(w["x1"] for w in words) - min(ids))] if ids else []
    if right_ids:
        mid = min(right_ids) - 15
    return top, bottom, mid


def _table_header_y(words: list[dict[str, Any]]) -> float | None:
    """y of the table heading row ("Course" / "รายวิชา"), used when no semester heading is found."""
    ys = [word["y"] for word in words if word["key"].startswith(("course", "รายวิชา"))]
    return min(ys) if ys else None


def _in_table(word: dict[str, Any]) -> bool:
    # With --layout every OCR line has a region; subject IDs are only inside the table.
    return word["region"] is None or word["region"].startswith("table_col")


def _rows(words: list[dict[str, Any]], tolerance: int = 20) -> list[list[dict[str, Any]]]:
    """Group words into visual rows (same page, close vertical centers), each row sorted left to right."""
    rows: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["page"], item["cy"])):
        row = rows[-1] if rows else None
        if row and row[0]["page"] == word["page"] and abs(word["cy"] - row[-1]["cy"]) <= tolerance:
            row.append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda item: item["x"]) for row in rows]


def _semester_headers_from_boxes(words: list[dict[str, Any]], admit_year: int | None = None) -> list[dict[str, Any]]:
    """Semester headings with their y position.

    Works on the text of each visual row, so it finds the heading whether the engine
    returns one box per word (tesseract) or one box per line (paddle, easyocr, surya).
    """
    headers = []
    for row in _rows(words):
        row_text = " ".join(word["text"] for word in row)
        found = [(SEMESTER_NUM[m.group(1).lower()], int(m.group(2))) for m in SEMESTER_RE.finditer(row_text)]
        thai = _thai_semester(row_text)
        if thai:
            found.append(thai)
        if _is_transfer_heading(row_text) and admit_year:
            # "Transferred Credits" / "รายวิชาเทียบโอน": the ground truth files these under
            # sem_num 0 of the admission year.
            found.append((0, admit_year))
        for sem_num, year in found:
            headers.append(
                {
                    "year": year + 543 if year < 2400 else year,
                    "sem_num": sem_num,
                    "start_y": min(word["y"] for word in row) + 30,
                }
            )
    return headers


def _thai_semester(row_text: str) -> tuple[int, int] | None:
    """(sem_num, year) from "ภาคการศึกษาที่ 1 ปีการศึกษา 2561" or "ภาคฤดูร้อน ปีการศึกษา 2562"."""
    loose = _loose_th(row_text)
    if not loose.startswith("ภาค") or len(loose) > 40:
        return None
    year = re.search(r"(\d{4})", loose)
    if not year:
        return None
    if "ฤดรอน" in loose:
        return 0, int(year.group(1))
    sem = re.search(r"ภาค\D*?(\d)\D", loose)
    return (int(sem.group(1)), int(year.group(1))) if sem else None




def _line_word(line: dict[str, Any]) -> dict[str, Any]:
    text = str(line.get("text", "")).strip()
    box = line.get("box") or [[0, 0]]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return {
        "text": text,
        "key": _token_key(text),
        "x": min(xs),
        "x1": max(xs),
        "cx": (min(xs) + max(xs)) / 2,
        "y": min(ys),
        "cy": (min(ys) + max(ys)) / 2,
        "page": line.get("page", 1),
        "region": line.get("region"),
    }


def _subject_from_rows(subject_id: dict[str, Any], rows: list[list[dict[str, Any]]], index: int) -> dict[str, Any]:
    """One subject: "<id> <name> [type] [credit] [grade]" on its row, the name possibly wrapped
    onto following rows (e.g. "RESEARCH METHODOLOGY AND ETHICS IN" / "NANOTECHNOLOGY")."""
    row = rows[index]
    after_id = [word for word in row if word["x"] > subject_id["x"] and word is not subject_id and word["key"]]
    name_words, type_, credit, grade = _split_subject_row(after_id)

    # Continuation rows: indented under the name, until the next subject / heading / summary row.
    previous = row
    for next_row in rows[index + 1 : index + 3]:
        if _ends_subject(next_row) or next_row[0]["x"] < subject_id["x1"] - 10:
            break
        if next_row[0]["cy"] - previous[0]["cy"] > 70:  # more than one line apart
            break
        name_words += [word for word in next_row if word["key"]]
        previous = next_row

    return {
        "subject_id": subject_id["key"].rstrip("."),
        "subject_name": _subject_text([word["text"] for word in name_words]),
        "type": type_,
        "credit": credit,
        "grade_earn": grade,
    }


def _split_subject_row(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None, int | None, str | None]:
    """Read the value columns from the right: grade, credit, type; the rest is the name."""
    i = len(words)
    grade = credit = type_ = None
    if i:
        credit_left = i >= 2 and re.fullmatch(r"\d{1,2}", words[i - 2]["key"]) is not None
        grade = _grade_value(words[i - 1]["key"], credit_left)
        if grade is not None:
            i -= 1
        elif credit_left and not words[i - 1]["key"].isdigit() and len(words[i - 1]["key"]) <= 3:
            i -= 1  # unreadable grade (e.g. "N)"): skip it so the credit is still found
    if i and re.fullmatch(r"\d{1,2}", words[i - 1]["key"]):
        credit = int(words[i - 1]["key"])
        i -= 1
    if i and words[i - 1]["key"] in TYPE_VALUES and credit is not None:
        type_ = TYPE_VALUES[words[i - 1]["key"]]
        i -= 1
    return words[:i], type_, credit, grade


def _grade_value(key: str, credit_left: bool) -> str | None:
    """Grade from the last token of a subject row, fixing common OCR confusions."""
    if GRADE_RE.match(key):
        return key
    fixed = re.sub(r"^[1l]+(?=[a-z])", "", key)  # "1A": table rule read as "1"
    fixed = re.sub(r"(?<=[abcd])t$", "+", fixed)  # "Bt": "+" read as "t"
    if GRADE_RE.match(fixed):
        return fixed
    if credit_left:  # a credit number sits left of it, so this token is the grade column
        match = re.fullmatch(r"([0568]|ธ|ร|ธร|ss|\$)(\+?)", fixed)
        if match:
            return GRADE_OCR_FIXES[match.group(1)] + match.group(2)
    return None


def _ends_subject(row: list[dict[str, Any]]) -> bool:
    text = " ".join(word["text"] for word in row)
    loose = _loose_th(text).lower()
    return (
        any(SUBJECT_ID_RE.match(word["key"]) for word in row)
        or any(word["key"] in {"gps", "gpa", "total", "cumulative", "end"} for word in row)
        or any(marker in loose for marker in TH_MARKERS)
        or SEMESTER_RE.search(text) is not None
        or _thai_semester(text) is not None
        or _is_transfer_heading(text)
    )


def _is_transfer_heading(text: str) -> bool:
    loose = _loose_th(text).lower()
    return loose.startswith(("transferredcredit", "transfercredit")) or loose.startswith("รายวชาเทยบโอน")








def _semester_scores(words: list[dict[str, Any]], start_y: int, end_y: int) -> tuple[str | None, str | None]:
    scoped = [word for word in words if start_y <= word["y"] < end_y]
    gps = _score_after(scoped, "gps")
    gpa = _score_after(scoped, "gpa")
    if gps is None and gpa is None:
        # "คะแนนเฉลี่ยประจำภาคการศึกษา : 2.33   คะแนนเฉลี่ย : 2.33" -> GPS (this semester), GPA (cumulative)
        for row in _rows(scoped):
            text = "".join(word["text"] for word in row)
            if "คะแนนเฉลยประจาภาค" in _loose_th(text):
                scores = re.findall(r"\d+\.\d{2}", text)
                gps = scores[0] if scores else None
                gpa = scores[1] if len(scores) > 1 else None
    return gps, gpa


def _score_after(words: list[dict[str, Any]], label: str) -> str | None:
    labels = [word for word in words if word["key"] == label]
    if not labels:
        return None
    label_word = labels[-1]
    candidates = [
        word for word in words
        if abs(word["y"] - label_word["y"]) <= 20 and word["x"] > label_word["x"] and re.fullmatch(r"\d+\.\d+", word["text"])
    ]
    return candidates[0]["text"] if candidates else None


def _extract_subjects_from_tokens(tokens: list[str], not_subjects: set[str]) -> list[dict[str, Any]]:
    semester = _semester_headers_from_text("\n".join(tokens))
    current = semester[0] if semester else {"year": None, "sem_num": None}
    subjects = []
    index = _semester_token_start(tokens)
    while index < len(tokens):
        if SUBJECT_ID_RE.match(_token_key(tokens[index])) and _token_key(tokens[index]).rstrip(".") not in not_subjects:
            next_index = index + 1
            while next_index < len(tokens) and not SUBJECT_ID_RE.match(_token_key(tokens[next_index])) and _token_key(tokens[next_index]) not in {"gps", "gpa"}:
                next_index += 1
            chunk = tokens[index + 1 : next_index]
            grade_pos = _last_grade_index(chunk)
            credit_pos = _last_credit_index(chunk[:grade_pos]) if grade_pos is not None else None
            name_tokens = chunk
            if credit_pos is not None and grade_pos is not None:
                name_tokens = [*chunk[:credit_pos], *chunk[grade_pos + 1 :]]
            elif credit_pos is not None:
                name_tokens = chunk[:credit_pos]
            subjects.append(
                {
                    "subject_id": _token_key(tokens[index]).rstrip("."),
                    "subject_name": _subject_text(name_tokens),
                    "type": None,
                    "credit": _to_int(chunk[credit_pos]) if credit_pos is not None else None,
                    "grade_earn": _schema_text(chunk[grade_pos]) if grade_pos is not None else None,
                }
            )
            index = next_index
        else:
            index += 1

    gps = _clean_gpa(_value_after_label(tokens, ["gps"]))
    gpa = _clean_gpa(_value_after_label(tokens, ["gpa"]))
    cumulative_gpa = _clean_gpa(_value_after_label(tokens, ["cumulative", "gpa"]))
    return [
        {
            "year": current.get("year"),
            "sem_num": current.get("sem_num"),
            "GPA": gpa or cumulative_gpa,
            "GPS": gps,
            "pass_reason": None,
            "subject": subjects,
        }
    ]


def _semester_token_start(tokens: list[str]) -> int:
    keys = [_token_key(token) for token in tokens]
    for index, key in enumerate(keys):
        if key in {"1st", "2nd", "3rd", "summer"} and keys[index + 1 : index + 4] == ["semester", "academic", "year"]:
            return index + 5
    return 0


def _semester_headers_from_text(text: str) -> list[dict[str, Any]]:
    """Year/semester of each heading in plain text (no positions; used by the token-based path)."""
    headers = []
    for match in SEMESTER_RE.finditer(text):
        year = int(match.group(2))
        headers.append({"year": year + 543 if year < 2400 else year, "sem_num": SEMESTER_NUM[match.group(1).lower()]})
    return headers


def _loose_th(text: str) -> str:
    """Text without whitespace and Thai vowel/tone marks (digits kept), e.g. for regex on headings."""
    return "".join(TH_CHAR_MAP.get(ch, ch) for ch in text if not ch.isspace() and ch not in TH_MARKS)


def _skeleton(text: str) -> tuple[str, list[int]]:
    """Thai letters only (no marks, digits, Latin, punctuation) + index of each kept char in `text`."""
    chars, index = [], []
    for i, ch in enumerate(text):
        if "\u0e01" <= ch <= "\u0e4f" and ch not in TH_MARKS:
            chars.append(TH_CHAR_MAP.get(ch, ch))
            index.append(i)
    return "".join(chars), index


def _find_label(text: str, label: str, min_score: float = 80) -> tuple[int, int, float] | None:
    """(start, end, score) of `label` in `text`, tolerant to OCR errors in Thai."""
    skel, index = _skeleton(text)
    target = _skeleton(label)[0]
    if not skel or not target:
        return None
    pos = skel.find(target)
    if pos >= 0:
        return index[pos], index[pos + len(target) - 1] + 1, 100.0
    if len(target) < 5:  # too short for fuzzy matching
        return None
    match = fuzz.partial_ratio_alignment(target, skel)
    if match.score < min_score or match.dest_end <= match.dest_start:
        return None
    return index[match.dest_start], index[match.dest_end - 1] + 1, match.score


def _row_texts(ocr_lines: list[dict[str, Any]]) -> list[tuple[float, str]]:
    """(y, text) of each visual row, words joined without spaces (Thai has no spaces between words)."""
    words = _page_words(ocr_lines)
    return [(min(word["y"] for word in row), "".join(word["text"] for word in row)) for row in _rows(words)]


def _labeled_values(text: str, labels: dict[str, str]) -> dict[str, str]:
    """Values of all labels found in one row; each value runs until the next label."""
    found = []
    for key, label in labels.items():
        hit = _find_label(text, label)
        if hit:
            found.append((hit[0], hit[1], hit[2], key))
    found.sort()
    kept = []
    for item in found:  # drop overlapping matches, keep the better one
        if kept and item[0] < kept[-1][1]:
            if item[2] > kept[-1][2]:
                kept[-1] = item
            continue
        kept.append(item)
    values = {}
    for i, (_, end, _, key) in enumerate(kept):
        stop = kept[i + 1][0] if i + 1 < len(kept) else len(text)
        values[key] = text[end:stop].strip(" :")
    return values


def _thai_chars(text: str) -> int:
    return sum("\u0e01" <= ch <= "\u0e4f" for ch in text)


def _table_start_y(rows: list[tuple[float, str]]) -> float:
    for y, text in rows:
        if _thai_semester(text) or _find_label(text, "รายวิชา", 90):
            return y
    return float("inf")


def _extract_header_th(rows: list[tuple[float, str]]) -> dict[str, Any]:
    above_table = [(y, text) for y, text in rows if y < _table_start_y(rows)]

    values: dict[str, str] = {}
    for _, text in above_table:
        for key, value in _labeled_values(text, TH_HEADER_LABELS).items():
            values.setdefault(key, value)

    # Rows above the title "ใบแสดงผลการศึกษา": university name and address.
    title_index = next((i for i, (_, text) in enumerate(above_table) if _find_label(text, "ใบแสดงผลการศึกษา")), None)
    top = [text for _, text in above_table[:title_index]] if title_index is not None else []
    address = next((text for text in top if _find_label(text, "เลขที่") or re.search(r"\d{5}", text)), None)
    names = [text for text in top if text is not address and _thai_chars(text) >= 10]
    uni_name = max(names, key=_thai_chars) if names else None
    after_title = above_table[title_index + 1:] if title_index is not None else above_table
    faculty = next((text for _, text in after_title if _loose_th(text).startswith("คณะ")), None)

    prename, name = None, values.get("name")
    if name:
        prename = next((p for p in TH_PRENAMES if name.startswith(p)), None)
        name = name[len(prename):] if prename else name

    grad = values.get("grad") or ""
    grad_date = _parse_date_th(grad)
    grad_reason = None if grad_date or _schema_text(grad) in (None, "n/a", "-") else _schema_text(grad)
    student_id = re.search(r"\d{8,12}", values.get("student_id", ""))

    return {
        "uni_name": _schema_text(uni_name),
        "uni_address": _schema_text(address),
        "student_id": student_id.group(0) if student_id else None,
        "faculty_name": _schema_text(faculty),
        "prename": prename,
        "name": _schema_text(name),
        "date_of_birth": _parse_date_th(values.get("date_of_birth")),
        "admis_date": _parse_date_th(values.get("admis_date")),
        "grad_date": grad_date or "0000-00-00",
        "grad_reason": grad_reason,
        "degree": _schema_text(values.get("degree")),
        "major": _schema_text(values.get("major")),
        "program": _schema_text(values.get("program")),
        "honor": 0,
    }


def _extract_totals_th(rows: list[tuple[float, str]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for _, text in rows:
        if "total_credits_earned" not in totals and _find_label(text, "จำนวนหน่วยกิตที่สอบได้ทั้งหมด"):
            totals["total_credits_earned"] = _to_int(text.split(":")[-1])
        elif "cumulative_gpa" not in totals and _find_label(text, "คะแนนเฉลี่ยสะสม", 85):
            totals["cumulative_gpa"] = _clean_gpa(text.split(":")[-1])
    return totals


def _extract_footer_th(rows: list[tuple[float, str]]) -> dict[str, Any]:
    issued = next((i for i, (_, text) in enumerate(rows) if _find_label(text, "วันที่ออกเอกสาร")), None)
    if issued is None:
        return {"updated_at": None, "by": {"by_signature": None, "by_position": None, "by_reg": None}}

    issued_text = rows[issued][1]
    hit = _find_label(issued_text, "วันที่ออกเอกสาร")
    note = _find_label(issued_text, "เอกสารจะสมบูรณ์")  # note printed on the same row
    updated_at = _parse_date_th(issued_text[hit[1]:note[0] if note and note[0] > hit[1] else None])

    # Below: "(signature)", position, office. Skip noise rows (stamp / signature line).
    below = [text for _, text in rows[issued + 1:] if _thai_chars(text) >= 3]
    signature_index = next((i for i, text in enumerate(below) if text.lstrip().startswith("(")), None)
    signature = position = office = None
    if signature_index is not None:
        signature = below[signature_index]
        rest = below[signature_index + 1:]
        position = rest[0] if rest else None
        office = rest[1] if len(rest) > 1 else None

    return {
        "updated_at": updated_at,
        "by": {
            "by_signature": _signature_text(signature),
            "by_position": _schema_text(position),
            "by_reg": _schema_text(office),
        },
    }


def _parse_date_th(value: str | None) -> str | None:
    """"3 กันยายน 2542" -> "1999-09-03", tolerant to missing vowels in the month name."""
    if not value:
        return None
    loose = _loose_th(value)
    match = re.search(r"(\d{1,2})(\D+?)(\d{4})", loose)
    if not match:
        return _parse_date(value)
    day, month_text, year = int(match.group(1)), _skeleton(match.group(2))[0], int(match.group(3))
    months = {_skeleton(name)[0]: number for name, number in TH_MONTHS.items()}
    month = months.get(month_text)
    if month is None and month_text:
        best = max(months, key=lambda name: fuzz.ratio(name, month_text))
        month = months[best] if fuzz.ratio(best, month_text) >= 70 else None
    if month is None or not 1 <= day <= 31:
        return _parse_date(value)
    year = year - 543 if year > 2400 else year
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    match = re.search(r"(?i)\b([a-z]+)\s+(\d{1,2}),?\s+(\d{4})", value)
    if match and match.group(1).lower() in EN_MONTHS:
        year = int(match.group(3))
        return f"{year:04d}-{EN_MONTHS[match.group(1).lower()]:02d}-{int(match.group(2)):02d}"

    for month_name, month in TH_MONTHS.items():
        match = re.search(rf"(\d{{1,2}})\s*{re.escape(month_name)}\s*(\d{{4}})", value)
        if match:
            year = int(match.group(2))
            year = year - 543 if year > 2400 else year
            return f"{year:04d}-{month:02d}-{int(match.group(1)):02d}"

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        return match.group(0)
    return None


def _first_match(tokens: list[str], pattern: str) -> str | None:
    for token in tokens:
        match = re.search(pattern, token)
        if match:
            return match.group(0)
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _clean_gpa(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\d+\.\d+", value)
    return match.group(0) if match else None


def _last_grade_index(tokens: list[str]) -> int | None:
    for index in range(len(tokens) - 1, -1, -1):
        if GRADE_RE.match(_token_key(tokens[index])):
            return index
    return None


def _last_credit_index(tokens: list[str]) -> int | None:
    for index in range(len(tokens) - 1, -1, -1):
        if re.fullmatch(r"\d", _token_key(tokens[index])):
            return index
    return None