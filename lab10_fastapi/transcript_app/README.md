# Transcript Application

แอปนี้รับ PDF/ภาพ → Lab 8A preprocessing → Typhoon OCR → Qwen จัดโครงสร้าง → postprocessing → JSON

## เอาไฟล์ไปวางที่ไหน

วางโฟลเดอร์ `transcript_app` ไว้ภายใน `ocr_system/lab10_fastapi/` และรักษาโครงสร้างนี้:

```text
ocr_system/
├── lab10_fastapi/
│   ├── __init__.py
│   └── transcript_app/      ← ไฟล์ Lab 10 ของกลุ่มนี้
└── src/ocr_system/
    ├── lab8a_denoise.py      ← preprocessing/postprocessing
    ├── lab7a_transcript.py  ← Typhoon OCR + Qwen
    └── lab7_metrics.py       ← dependency ที่ Lab 7A ใช้
```

ห้ามย้าย `main.py` ออกจาก `transcript_app` และให้รันคำสั่งจากโฟลเดอร์ `ocr_system`

ให้รันจากรากโปรเจกต์ `ocr_system` ใน VS Code Terminal:

```bash
python -m pip install -r lab10_fastapi/transcript_app/requirements.txt
ollama pull scb10x/typhoon-ocr1.5-3b
ollama pull qwen3:4b
python -m uvicorn lab10_fastapi.transcript_app.main:app --reload --port 8001
```

เปิด <http://127.0.0.1:8001/> หรือ <http://127.0.0.1:8001/docs>

ก่อนรันต้องมี:

- `lab10_fastapi/transcript_app/.env`
- `src/ocr_system/lab8a_denoise.py`
- `src/ocr_system/lab7a_transcript.py`
- dependency ภายใน `src/ocr_system` ที่ Lab 7A/8A import

ถ้าแจกเฉพาะกลุ่ม Transcript ให้ก็อปโฟลเดอร์นี้พร้อม `src/ocr_system` ของ Lab 7A/8A

> ไฟล์สะอาดควรเริ่มด้วย preprocessing=`none`; ผล OCR ต้องตรวจเทียบ Ground Truth ก่อนใช้งานจริง

## Wireframe

หน้าเว็บมีหน้าเดียว แต่แสดงผลต่างกันตามสถานะ 4 แบบ: **Idle → Loading → Success หรือ Error**
ภาพนี้ตรงกับ `static/index.html` + `static/app.js` + `static/style.css` (ข้อมูลในภาพเป็นค่าสมมติ)
ธีมมืด `#242424` การ์ดขาว 3 ใบ: (1) อัปโหลด + สถานะ (2) ผลลัพธ์ + ปุ่มดาวน์โหลด JSON (3) คำเตือน + OCR Markdown ฟอนต์ Poppins

![Wireframe ของหน้า Transcript OCR ทั้ง 4 สถานะ](wireframe/wireframe.png)

ไฟล์ต้นฉบับแบบเวกเตอร์: [`wireframe/wireframe.svg`](wireframe/wireframe.svg)

ภาพนี้ถูกวาดด้วยสคริปต์ [`wireframe/make_wireframe.py`](wireframe/make_wireframe.py) (ไม่ได้ส่งออกจาก Figma/Canva หรือแคปจากหน้าเว็บจริง) ถ้าแก้หน้าเว็บต้องแก้สคริปต์แล้วรันใหม่ด้วย `python make_wireframe.py`

## API Contract

ข้อตกลงระหว่าง Frontend (`static/index.html`, `static/app.js`) และ Backend (`main.py`)
ถ้าแก้ฝั่งใดฝั่งหนึ่ง ต้องแก้ตารางนี้และอีกฝั่งให้ตรงกันด้วย
ตัวอย่างข้อมูลด้านล่างเป็นค่าสมมติ ไม่ใช่ข้อมูลของนักศึกษาจริง

### 1. Endpoint และ 2. Method

| Method | Endpoint | หน้าที่ | Content-Type ของ request |
|---|---|---|---|
| `GET` | `/api/health` | ตรวจว่าแอปทำงานและตั้งค่าอะไรไว้ | ไม่มี body |
| `POST` | `/api/transcript/extract` | อัปโหลด Transcript แล้วสกัดข้อมูลเป็น JSON | `multipart/form-data` |

หน้าเว็บ (`GET /`) และไฟล์ static (`/static/...`) ไม่ใช่ API ข้อมูล

### 3. Field Name และ 4. Data Types

**Request ของ `POST /api/transcript/extract`** (ส่งเป็นฟอร์ม ไม่ใช่ JSON)

| Field | ชนิด | บังคับ | ค่าเริ่มต้น | ข้อกำหนด |
|---|---|---|---|---|
| `file` | ไฟล์ | ใช่ | — | นามสกุล `.pdf` `.png` `.jpg` `.jpeg` `.tif` `.tiff` ขนาดไม่เกิน `TRANSCRIPT_MAX_UPLOAD_MB` (ค่าเริ่มต้น 20 MB) |
| `preprocessing` | string | ไม่ | `none` | ต้องเป็น `none`, `light` หรือ `heavy` |
| `include_markdown` | boolean | ไม่ | `false` | `true` = แนบ Markdown ที่ OCR อ่านได้มาใน response |

**Response สำเร็จ (`200`)** เป็น JSON ตาม `TranscriptResponse` ใน `schemas.py`

| Field | ชนิด | ความหมาย |
|---|---|---|
| `filename` | string | ชื่อไฟล์ที่อัปโหลด |
| `pages` | integer | จำนวนหน้า |
| `preprocessing` | string | วิธีทำความสะอาดภาพที่ใช้ |
| `header_detail` | object | ข้อมูลส่วนหัว (ชื่อ รหัส วันที่ หลักสูตร ฯลฯ) |
| `transcript_detail` | object | ภาคเรียน รายวิชา และสรุปผลการศึกษา |
| `footer_detail` | object | ข้อมูลท้ายเอกสาร |
| `warnings` | array ของ string | ข้อสังเกตจากการตรวจความสอดคล้องภายใน (ว่างได้) |
| `postprocessing_changes` | array ของ string | ค่าที่โค้ดแก้ให้หลัง LLM ตอบ (ว่างได้) |
| `processing_seconds` | number | เวลาที่ใช้ประมวลผล (วินาที) |
| `markdown` | string หรือ `null` | Markdown ที่ OCR อ่านได้ มีค่าเมื่อส่ง `include_markdown=true` เท่านั้น |

โครงของ `transcript_detail` ที่หน้าเว็บอ่าน:

```text
transcript_detail
├── semesters[]                     ← array ของภาคเรียน
│   ├── year            integer     ปีการศึกษา (พ.ศ.)
│   ├── sem_num         integer     1, 2 หรือ 3
│   ├── GPS, GPA        string      เกรดเฉลี่ยประจำภาค / สะสม
│   ├── pass_reason     string      เช่น "maintain" สำหรับภาครักษาสภาพ
│   └── subject[]                   ← array ของรายวิชา (ภาครักษาสภาพเป็น array ว่าง)
│       ├── subject_id, subject_name, type, grade_earn   string
│       └── credit      integer
├── total_credits_earned            integer
├── cumulative_gpa                  string
└── master_comprehensive / master_thesis / master_qualify   string (ใช้กับบัณฑิตศึกษา)
```

ฟิลด์ข้อมูลเอกสาร **เป็น `null` ได้ทุกตัว** เมื่ออ่านไม่ออกหรือเอกสารไม่มี Frontend ต้องรองรับค่า `null`
(`app.js` ใช้ `subject[key] ?? ""`) และ API **ไม่มีค่า confidence ต่อฟิลด์** เพราะใช้ LLM จัดโครงสร้าง
ให้ดูจาก `warnings` และ `postprocessing_changes` แทน

**Response ของ `GET /api/health`**

| Field | ชนิด | ตัวอย่าง |
|---|---|---|
| `status` | string | `"ok"` |
| `ocr_model` | string | `"scb10x/typhoon-ocr1.5-3b"` |
| `text_model` | string | `"qwen3:4b"` |
| `max_upload_mb` | integer | `20` |
| `lab8a_module` | string | path ของ `lab8a_denoise.py` ที่โหลดอยู่ |

endpoint นี้ตรวจเฉพาะค่าตั้งต้น **ไม่ได้ทดสอบว่า Ollama ทำงานอยู่**

### 5. Error Format

Error ที่ระบบส่งเองเป็น JSON รูปแบบเดียวกันเสมอ:

```json
{ "detail": "ข้อความอธิบาย" }
```

| HTTP | เกิดเมื่อ | ตัวอย่าง `detail` | Frontend แสดง |
|---|---|---|---|
| `413` | ไฟล์ใหญ่เกินกำหนด | `"ไฟล์ใหญ่เกิน 20 MB"` | ข้อความจาก `detail` |
| `415` | นามสกุลไฟล์ไม่รองรับ | `"รองรับเฉพาะ PDF/PNG/JPG/TIFF"` | ข้อความจาก `detail` |
| `422` | ประมวลผลไม่สำเร็จ เช่น `preprocessing` ผิดค่า, ไฟล์เสียหรือว่าง, ติดต่อ Ollama ไม่ได้, โมเดลตอบกลับไม่ใช่ JSON | `"preprocessing ต้องเป็น none, light หรือ heavy"` | `อ่านเอกสารไม่สำเร็จ: ...` |
| `422` | ไม่ส่ง `file` มา (FastAPI ตรวจเอง) | `"detail"` เป็น **array** ไม่ใช่ string | `ข้อมูลที่ส่งไปไม่ถูกต้อง` |
| `405` | ใช้ Method ผิด เช่น `GET` ที่ `/api/transcript/extract` (หน้าเว็บไม่มีทางส่งแบบนี้) | `"Method Not Allowed"` | ข้อความจาก `detail` |
| `500` | ข้อผิดพลาดที่ไม่ได้คาดไว้ | ตอบกลับเป็นข้อความธรรมดา **ไม่ใช่ JSON** | `เซิร์ฟเวอร์ตอบผิดพลาด (รหัส 500)` |
| — | ติดต่อ server ไม่ได้เลย (ไม่มี HTTP response) | — | `เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ...` |

ข้อควรระวังสำหรับ Frontend:
- `detail` เป็น string ในกรณีที่ระบบส่งเอง แต่เป็น **array** เมื่อ FastAPI ตรวจ input ไม่ผ่าน ต้องเช็กชนิดก่อนนำไปแสดง
- ต้องเช็ก `response.ok` และดักกรณีที่ `response.json()` อ่านไม่ได้ (เช่นข้อ `500`)
- `detail` ของ `422` บางกรณีมี path ชั่วคราวของเซิร์ฟเวอร์ปนอยู่ (เช่นไฟล์ PDF เสีย)

### Contract ในรูป JSON

ข้อมูลเดียวกับตารางด้านบน เขียนในรูปที่โปรแกรมอ่านได้ ชนิดข้อมูลเขียนแบบย่อ (`string|null` = เป็น string หรือ null ก็ได้)

```json
{
  "contract_version": "1.0",
  "base_url": "http://127.0.0.1:8001",
  "endpoints": [
    {
      "name": "health",
      "method": "GET",
      "path": "/api/health",
      "request": null,
      "responses": {
        "200": {
          "status": "string",
          "ocr_model": "string",
          "text_model": "string",
          "max_upload_mb": "integer",
          "ocr_engine": "string",
          "tesseract_available": "boolean",
          "lab8a_module": "string"
        }
      }
    },
    {
      "name": "extract_transcript",
      "method": "POST",
      "path": "/api/transcript/extract",
      "request": {
        "content_type": "multipart/form-data",
        "fields": {
          "file": {
            "type": "file",
            "required": true,
            "accept": [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
            "max_mb": 20
          },
          "method": {
            "type": "string",
            "required": false,
            "default": "vlm",
            "enum": ["vlm", "ocr"]
          },
          "preprocessing": {
            "type": "string",
            "required": false,
            "default": "none",
            "enum": ["none", "light", "heavy"]
          },
          "include_markdown": {
            "type": "boolean",
            "required": false,
            "default": false
          }
        }
      },
      "responses": {
        "200": {
          "filename": "string",
          "method": "string",
          "pages": "integer",
          "preprocessing": "string",
          "header_detail": "object",
          "transcript_detail": "object",
          "footer_detail": "object",
          "warnings": "array<string>",
          "postprocessing_changes": "array<string>",
          "processing_seconds": "number",
          "markdown": "string|null"
        }
      },
      "errors": {
        "413": { "body": { "detail": "string" }, "when": "ไฟล์ใหญ่เกินกำหนด" },
        "415": { "body": { "detail": "string" }, "when": "นามสกุลไฟล์ไม่รองรับ" },
        "422": { "body": { "detail": "string" }, "when": "ประมวลผลไม่สำเร็จ (ไฟล์เสีย, method/preprocessing ผิดค่า, ติดต่อ Ollama ไม่ได้, ไม่พบ tesseract)" },
        "422 (FastAPI)": { "body": { "detail": "array<object>" }, "when": "ไม่ส่ง field ที่บังคับ เช่น file" }
      }
    }
  ],
  "error_format": { "detail": "string | array<object>" }
}
```

Contract แบบเต็มที่ FastAPI สร้างจากโค้ดโดยอัตโนมัติ (มาตรฐาน OpenAPI) ดูได้ตอนรัน server:

- <http://127.0.0.1:8001/openapi.json> (JSON)
- <http://127.0.0.1:8001/docs> (หน้าสำหรับลองเรียก)

ถ้าแก้ `main.py` หรือ `schemas.py` ต้องแก้ JSON ข้างบนตามด้วย

#### ตรวจ Contract กับโค้ด

`check_contract.py` อ่านบล็อก JSON ข้างบนแล้วเทียบกับ OpenAPI ที่ FastAPI สร้างจากโค้ดจริง
(ชื่อ field, ชนิด, บังคับหรือไม่, ค่าเริ่มต้น, นามสกุลไฟล์, ขนาดสูงสุด, รหัส error)
ไม่เรียก Ollama และไม่ต้องรัน server รันจากรากโปรเจกต์ `ocr_system`:

```bash
python -m lab10_fastapi.transcript_app.check_contract
```

ควรได้ `ผ่าน 20 · ไม่ผ่าน 0` ถ้ามีข้อใดไม่ผ่านจะแสดงค่าใน README เทียบกับค่าในโค้ดจริง
รันทุกครั้งหลังแก้ API เพื่อให้ Frontend กับ Backend ตกลงกันตามเอกสารเสมอ

### วิธีอ่านเอกสาร (`method`)

| ค่า | ทำอะไร | จุดเด่น / ข้อจำกัด |
|---|---|---|
| `vlm` (ค่าเริ่มต้น) | Typhoon-OCR → qwen3:4b → กฎ postprocessing ของ Lab 8A | ทนเค้าโครงที่หลากหลายกว่า แต่ช้า (1–3 นาทีต่อไฟล์) ต้องมี Ollama |
| `ocr` | tesseract (`ocr_system.pipeline`) → สกัดข้อมูลด้วยกฎ (`transcript_extraction`) | เร็ว ไม่ต้องใช้ Ollama ต้องติดตั้งโปรแกรม `tesseract` + ภาษาไทย (`brew install tesseract tesseract-lang`) ไม่ผ่านกฎ snap ของ Lab 8A (`postprocessing_changes` เป็น `[]`) |

ช่อง `preprocessing` ใช้ร่วมกันแต่ความหมายต่างกัน (ค่าที่ส่งไป API ยังเป็น `none` / `light` / `heavy` เหมือนกันทั้งสองโหมด):

| ค่า | โหมด `vlm` | โหมด `ocr` (ข้อความที่หน้าเว็บแสดง) |
|---|---|---|
| `none` | ภาพดิบ | ไม่ preprocess |
| `light` | ทำความสะอาดแบบเบา (Lab 8A) | preprocess แค่ deskew (ปรับภาพเอียง) — ค่าเริ่มต้นของโหมดนี้ |
| `heavy` | ทำความสะอาดแบบหนัก (Lab 8A) | preprocess ทั้ง pipeline ของ `ocr_system` (deskew + ทำความสะอาดเต็มรูปแบบ) |
ถ้าเลือก `include_markdown` โหมด `ocr` จะส่งข้อความที่ OCR ได้กลับมาในฟิลด์ `markdown`

ค่า engine และภาษาของโหมด `ocr` ตั้งใน `.env` ได้ (`TRANSCRIPT_OCR_ENGINE`, `TRANSCRIPT_OCR_LANGS`)

### ข้อควรรู้เรื่องเวลาและการใช้งานพร้อมกัน

- OCR ใช้เวลาราว 1–3 นาทีต่อเอกสาร (ที่ 300 DPI) Client ต้องตั้ง timeout ให้นานกว่านี้ และไม่ควรสรุปว่าระบบค้างเมื่อรอไม่กี่สิบวินาที
- โหมด `ocr` ใช้เวลาน้อยกว่ามาก (ส่วนใหญ่ไม่กี่วินาทีต่อหน้า ขึ้นกับเครื่อง)
- Ollama ประมวลผลทีละงาน จึงควรส่งทีละไฟล์ (หน้าเว็บปิดปุ่มระหว่างรอ)
- ไฟล์ที่อัปโหลดถูกเก็บในโฟลเดอร์ชั่วคราวและลบทันทีเมื่อ request จบ (PDPA) ทั้งสองโหมด
  รวมถึงไฟล์ที่โหมด `ocr` สร้างระหว่างทาง (ภาพหน้า, `*_ocr.json`, `*_ocr.txt`) ส่วนการรันผ่าน CLI (`python -m ocr_system.cli ...`) ยังเขียนที่ `outputs/` ตามเดิม

### ตัวอย่างการเรียก

```bash
curl -X POST http://127.0.0.1:8001/api/transcript/extract \
  -F "file=@transcript.pdf" \
  -F "method=vlm" \
  -F "preprocessing=none" \
  -F "include_markdown=false"
```

ตัวอย่าง response (ย่อ ค่าสมมติ):

```json
{
  "filename": "transcript.pdf",
  "method": "vlm",
  "pages": 1,
  "preprocessing": "none",
  "header_detail": { "student_id": "00000000", "name": "ชื่อสมมติ", "degree": "วิทยาศาสตรบัณฑิต" },
  "transcript_detail": {
    "semesters": [
      {
        "year": 2561, "sem_num": 1, "GPS": "3.00", "GPA": "3.00", "pass_reason": null,
        "subject": [
          { "subject_id": "00000001", "subject_name": "วิชาตัวอย่าง", "type": null, "credit": 3, "grade_earn": "A" }
        ]
      }
    ],
    "total_credits_earned": 3,
    "cumulative_gpa": "3.00"
  },
  "footer_detail": { "updated_at": "2026-01-01" },
  "warnings": [],
  "postprocessing_changes": [],
  "processing_seconds": 95.2,
  "markdown": null
}
```

ตัวอย่าง error:

```json
{ "detail": "รองรับเฉพาะ PDF/PNG/JPG/TIFF" }
```

## ไฟล์ผลลัพธ์สำหรับ Challenge

1. เปิด `http://127.0.0.1:8001/` อัปโหลด Transcript ด้วย `preprocessing = none` (ไฟล์สะอาด) รอจนสถานะเป็นสีเขียว
2. กดปุ่ม **ดาวน์โหลด JSON** จะได้ไฟล์ `<รหัสนักศึกษา>_transcript.json`
   ไฟล์มี 3 ส่วนเหมือนไฟล์เฉลย (`header_detail`, `transcript_detail`, `footer_detail`) ไม่มี `warnings` หรือ `markdown` ปนอยู่
3. วัด Accuracy / CER / WER เทียบเฉลย (รันจากรากโปรเจกต์ `ocr_system`):

   ```bash
   python -m ocr_system.vlm.lab7a_transcript --eval-only ~/Downloads/<ไฟล์>.json --gt data/ground_truth/Json_<รหัส>_th.json
   ```

   แถว **รวม (micro)** ในตารางคือค่าที่ต้องส่ง และโปรแกรมบันทึกสรุปเป็นไฟล์
   `outputs/lab7a/<ไฟล์>__metrics.json` ให้ด้วย (มีแต่ตัวเลข ไม่มีค่าข้อมูลของนักศึกษา)
4. ส่งไฟล์ JSON ผลและไฟล์ `__metrics.json` ลงกลุ่ม Discord

ก่อนวัน Challenge ต้องติดตั้ง `pythainlp` ใน venv ที่ใช้ (`pip install pythainlp`) แล้วตรวจว่าตารางขึ้นบรรทัด
`tokenizer สำหรับ WER: pythainlp/newmm` ถ้าขึ้น `whitespace (fallback)` แปลว่ายังไม่ได้ติดตั้ง
อธิบายการคำนวณและข้อจำกัดของ WER ได้ที่ README หลักของโปรเจกต์ หัวข้อ "วัด Accuracy / CER / WER สำหรับไฟล์ผลลัพธ์"

## คำถามที่พบบ่อย

### เปิดเว็บแล้วขึ้น `ERR_CONNECTION_REFUSED`

FastAPI ยังไม่ได้รัน ให้รันคำสั่ง Uvicorn ด้านบนและเปิด Terminal ค้างไว้

### ขึ้น `No module named fastapi` หรือ `No module named multipart`

ยังไม่ได้ activate venv หรือยังไม่ได้ติดตั้ง `transcript_app/requirements.txt`

### อัปโหลดแล้วขึ้น `ติดต่อ Ollama ไม่ได้`

เปิด Ollama แล้วรัน `ollama list` เพื่อตรวจว่ามี Typhoon และ Qwen

### OCR นานหรือค่าที่อ่านได้ผิด

โมเดลต้องประมวลผลภาพและสร้าง JSON จึงช้ากว่า OCR ปกติ ไฟล์สะอาดให้เลือก `none` และตรวจผลกับ Ground Truth
