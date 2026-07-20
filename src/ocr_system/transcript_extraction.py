import json
import re
from pathlib import Path
from typing import Any


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

GRADE_RE = re.compile(r"^(?:[abcdf][+-]?|s|u|w|i|t\([abcdfs][+-]?\)|-)$", re.IGNORECASE)
SUBJECT_ID_RE = re.compile(r"^\d{8}\.?$")


def extract_transcript_from_file(path: str | Path, language: str | None = None) -> dict[str, Any]:
    payload = _load_ocr_payload(path)
    return extract_transcript(payload["text"], payload.get("lines", []), language=language)


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
        semesters = _extract_subjects_from_boxes(ocr_lines, text)
    else:
        semesters = _extract_subjects_from_tokens(tokens)

    transcript = {
        "semesters": semesters,
        "master_comprehensive": None,
        "master_thesis": None,
        "master_qualify": None,
        "total_credits_earned": _to_int(_value_after_label(tokens, ["total", "credits", "earned"])),
        "cumulative_gpa": _clean_gpa(_value_after_label(tokens, ["cumulative", "gpa"])),
    }

    return {
        "header_detail": header,
        "transcript_detail": transcript,
        "footer_detail": _extract_footer(tokens, is_thai=is_thai),
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
    thai_chars = len(re.findall(r"[\u0e00-\u0e7f]", text))
    return "th" if thai_chars else "en"


def _clean_tokens(lines: list[str]) -> list[str]:
    tokens = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("--- Page"):
            continue
        tokens.append(value)
    return tokens


def _token_key(value: str) -> str:
    return re.sub(r"[\s:.,]+", "", value).lower()


def _schema_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
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


def _extract_subjects_from_boxes(ocr_lines: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    words = [_line_word(line) for line in ocr_lines if _line_word(line)["text"]]
    words.sort(key=lambda item: (item["page"], item["y"], item["x"]))
    subject_words = [word for word in words if SUBJECT_ID_RE.match(word["key"])]

    semesters = _semester_headers_from_boxes(words) or _semester_headers_from_text(text)
    if not semesters and subject_words:
        first_course = next((word for word in subject_words if word["y"] > 700), subject_words[0])
        semesters = [{"year": None, "sem_num": None, "start_y": first_course["y"] - 30}]

    results = []
    for index, semester in enumerate(semesters):
        start_y = semester.get("start_y") or 0
        end_y = semesters[index + 1].get("start_y") if index + 1 < len(semesters) else 10**9
        semester_subjects = []
        ids = [word for word in subject_words if start_y <= word["y"] < end_y]
        for subject_index, subject_id in enumerate(ids):
            next_y = ids[subject_index + 1]["y"] if subject_index + 1 < len(ids) else end_y
            marker_y = _first_marker_y_after(words, subject_id["y"], next_y)
            if marker_y is not None:
                next_y = marker_y
            row_words = [word for word in words if subject_id["y"] - 10 <= word["y"] < next_y - 5]
            semester_subjects.append(_subject_from_box_words(subject_id, row_words))

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


def _semester_headers_from_boxes(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers = []
    keys = [word["key"] for word in words]
    for index, key in enumerate(keys):
        if key not in {"1st", "2nd", "3rd", "summer"}:
            continue
        window = keys[index : index + 5]
        if len(window) < 5 or window[1] != "semester" or window[2] != "academic" or window[3] != "year":
            continue
        if not re.fullmatch(r"\d{4}", window[4]):
            continue
        year = int(window[4])
        sem_num = {"1st": 1, "2nd": 2, "3rd": 3, "summer": 0}[key]
        headers.append(
            {
                "year": year + 543 if year < 2400 else year,
                "sem_num": sem_num,
                "start_y": words[index]["y"] + 30,
            }
        )
    return headers


def _first_marker_y_after(words: list[dict[str, Any]], start_y: int, fallback_y: int) -> int | None:
    markers = [
        word["y"]
        for word in words
        if start_y < word["y"] < fallback_y and word["key"] in {"gps", "gpa", "total", "cumulative", "end"}
    ]
    return min(markers) if markers else None


def _line_word(line: dict[str, Any]) -> dict[str, Any]:
    text = str(line.get("text", "")).strip()
    box = line.get("box") or [[0, 0]]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return {
        "text": text,
        "key": _token_key(text),
        "x": min(xs),
        "y": min(ys),
        "page": line.get("page", 1),
    }


def _subject_from_box_words(subject_id: dict[str, Any], row_words: list[dict[str, Any]]) -> dict[str, Any]:
    same_line = [word for word in row_words if abs(word["y"] - subject_id["y"]) <= 18]
    grade = _last_grade(same_line) or _last_grade(row_words)
    credit = _credit_before_grade(same_line, grade) or _credit_before_grade(row_words, grade)
    name_candidates = [
        word
        for word in row_words
        if word["x"] > subject_id["x"] + 80
        and word["x"] < 1000
        and not SUBJECT_ID_RE.match(word["key"])
        and word is not grade
        and word is not credit
    ]
    name_candidates.sort(key=lambda word: (round((word["y"] - subject_id["y"]) / 30), word["x"]))
    return {
        "subject_id": subject_id["key"].rstrip("."),
        "subject_name": _subject_text([word["text"] for word in name_candidates]),
        "type": None,
        "credit": _to_int(credit["text"] if credit else None),
        "grade_earn": _schema_text(grade["text"] if grade else None),
    }


def _last_grade(words: list[dict[str, Any]]) -> dict[str, Any] | None:
    grades = [word for word in words if GRADE_RE.match(word["key"])]
    return max(grades, key=lambda item: (item["x"], item["y"]), default=None)


def _credit_before_grade(words: list[dict[str, Any]], grade: dict[str, Any] | None) -> dict[str, Any] | None:
    candidates = [word for word in words if re.fullmatch(r"\d", word["key"])]
    if grade:
        before_grade = [word for word in candidates if word["x"] < grade["x"] + 20]
        if before_grade:
            return max(before_grade, key=lambda item: item["x"])
    return max(candidates, key=lambda item: item["x"], default=None)


def _semester_scores(words: list[dict[str, Any]], start_y: int, end_y: int) -> tuple[str | None, str | None]:
    scoped = [word for word in words if start_y <= word["y"] < end_y]
    gps = _score_after(scoped, "gps")
    gpa = _score_after(scoped, "gpa")
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


def _extract_subjects_from_tokens(tokens: list[str]) -> list[dict[str, Any]]:
    semester = _semester_headers_from_text("\n".join(tokens))
    current = semester[0] if semester else {"year": None, "sem_num": None}
    subjects = []
    index = _semester_token_start(tokens)
    while index < len(tokens):
        if SUBJECT_ID_RE.match(_token_key(tokens[index])):
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
    headers = []
    for match in re.finditer(r"(?i)\b(1st|2nd|3rd|summer)\s+semester,?\s+academic\s+year\s+(\d{4})", text):
        sem_num = {"1st": 1, "2nd": 2, "3rd": 3, "summer": 0}[match.group(1).lower()]
        year = int(match.group(2))
        headers.append({"year": year + 543 if year < 2400 else year, "sem_num": sem_num, "start_y": _rough_y_from_offset(text, match.end())})
    return headers


def _rough_y_from_offset(text: str, offset: int) -> int:
    # Used only to place the first subject block after a detected semester header.
    return 0 if offset < 0 else 1


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

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="ดึง field จากผล OCR ของ transcript")
#     parser.add_argument("ocr_json", help="ไฟล์ผล OCR เช่น outputs/71010001_ocr.json")
#     parser.add_argument("-o", "--output", help="ไฟล์ผลลัพธ์ (ถ้าไม่ใส่จะพิมพ์ออกหน้าจอ)")
#     parser.add_argument("--language", default=None, help="th หรือ en (ปล่อยว่างให้ตรวจอัตโนมัติ)")
#     args = parser.parse_args()

#     result = extract_transcript_from_file(args.ocr_json, language=args.language)
#     text = json.dumps(result, ensure_ascii=False, indent=2)

#     if args.output:
#         Path(args.output).parent.mkdir(parents=True, exist_ok=True)
#         Path(args.output).write_text(text, encoding="utf-8")
#         print(f"บันทึกแล้ว -> {args.output}")
#     else:
#         print(text)