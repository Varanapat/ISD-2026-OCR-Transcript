def parse_transcript_by_coordinates(ocr_lines, y_threshold=12):
    """
    รับ List ของ OCRLine object เข้ามาจัดกลุ่มตามพิกัด X, Y
    """
    words = []
    for line_obj in ocr_lines:
        # box คือพิกัด [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        bbox = line_obj.box
        text = line_obj.text
        
        if text and text.strip():
            words.append({
                'text': text.strip(),
                'x': bbox[0][0],  # พิกัด X มุมซ้ายบน
                'y': bbox[0][1]   # พิกัด Y มุมซ้ายบน
            })

    # Sort ตามแนวตั้ง (บนลงล่าง)
    words.sort(key=lambda w: w['y'])

    lines = []
    current_line = []
    current_y = None

    for word in words:
        if current_y is None or abs(word['y'] - current_y) <= y_threshold:
            current_line.append(word)
            if current_y is None:
                current_y = word['y']
        else:
            # Sort คำในบรรทัดเดียวกันตามแนวนอน (ซ้ายไปขวา)
            current_line.sort(key=lambda w: w['x'])
            lines.append(" \t ".join([w['text'] for w in current_line]))
            current_line = [word]
            current_y = word['y']

    if current_line:
        current_line.sort(key=lambda w: w['x'])
        lines.append(" \t ".join([w['text'] for w in current_line]))

    return lines