# Thai-English OCR System

โปรเจกต์นี้เป็น OCR pipeline สำหรับเอกสารภาพเดี่ยวและหลายหน้า เช่น `.jpg`, `.png`, `.tif`, `.pdf` โดยรองรับเอกสารภาษาไทยและอังกฤษปนกัน
และมีคำสั่ง `benchmark` สำหรับวัดผล OCR หลาย engine แข่งกันเทียบกับ ground truth

OCR engines ที่มีให้ (`--engine`):

| engine | ภาษาไทย | ความเร็ว (CPU, ต่อหน้า) | หมายเหตุ |
|---|---|---|---|
| `tesseract` | ได้ | ~1.5 วินาที | เร็วที่สุด ต้องติดตั้งโปรแกรม Tesseract แยก |
| `paddle` | ได้ | ~10 วินาที | PaddleOCR 3.x ค่าเริ่มต้นใช้โมเดล mobile |
| `easyocr` | ได้ | ~16 วินาที | ภาษาไทยใช้คู่กับอังกฤษได้เท่านั้น (`th,en`) |
| `doctr` | **ไม่ได้** | ~2 วินาที | อ่านได้แค่ภาษาอังกฤษ/ตัวอักษรละติน เหมาะกับเอกสาร `_en` |
| `surya` | ได้ | ขึ้นกับเครื่อง | Surya OCR 2 (โมเดล AI อ่านทั้งหน้า) ต้องติดตั้ง `llama.cpp` แยก |
| `trocr` | ไม่ได้ | - | อ่านได้ทีละบรรทัด ไม่เหมาะกับเอกสารทั้งหน้า |
| `ensemble` | ได้ | รวมทุกตัว | ใช้ Paddle + Tesseract + EasyOCR แล้วรวมผล |

---

## Project Structure

```text
ocr_system/
├── README.md
├── requirements.txt           # รายชื่อ Python packages
├── pyproject.toml             # ตั้งค่าให้ import ocr_system จาก src/ ได้
├── data/
│   ├── input/                 # เอกสาร PDF ที่ต้องการ OCR (ชุดที่ 1)
│   ├── ground_truth/          # เฉลยของชุดที่ 1: Json_<id>_th.json / Json_<id>_en.json
│   ├── input_G/               # เอกสาร PDF ชุด G
│   └── ground_truth_G/        # เฉลยของชุด G
├── outputs/                   # ผลลัพธ์ OCR / evaluation / benchmark (สร้างอัตโนมัติ ลบได้)
└── src/
    └── ocr_system/
        ├── cli.py             # command line: ocr / evaluate / benchmark
        ├── config.py          # ค่าตั้งค่าหลัก (OCRConfig)
        ├── pipeline.py        # OCR pipeline หลัก: โหลด → preprocess → OCR → เซฟผล
        ├── document_loader.py # โหลดภาพ / แปลง PDF เป็นภาพทีละหน้า
        ├── preprocessing.py   # resize, denoise, contrast, deskew, threshold
        ├── engine_factory.py  # เลือก OCR engine ตามชื่อ + กำหนดสมาชิกของ ensemble
        ├── evaluation.py      # วัดผล: field_recall, CER, WER
        ├── benchmark.py       # วัดผลหลาย engine × หลายไฟล์ แล้วสรุปเป็นตาราง
        ├── field_extraction.py# ดึง field เช่น email, date, id, phone ด้วย regex
        ├── schemas.py         # dataclass ของผลลัพธ์ (OCRLine, OCRPageResult, ...)
        ├── engines/
        │   ├── base.py            # แม่แบบที่ทุก engine ต้องทำตาม
        │   ├── tesseract_engine.py
        │   ├── paddle_engine.py
        │   ├── easyocr_engine.py
        │   ├── doctr_engine.py
        │   ├── surya_engine.py
        │   ├── trocr_engine.py
        │   └── ensemble_engine.py
        └── utils/
            └── io.py          # เซฟ JSON/TXT, สร้างโฟลเดอร์
```

### Pipeline ทำงานอย่างไร

```text
python -m ocr_system.cli ocr <ไฟล์> --engine <engine>
   │
   ├─ cli.py               อ่าน option → สร้าง OCRConfig
   ├─ document_loader.py   PDF → รูป JPG ทีละหน้า (outputs/pages/)
   ├─ engine_factory.py    สร้าง engine ตาม --engine
   ├─ ทีละหน้า:
   │    ├─ preprocessing.py   ทำความสะอาดรูป (ปิดได้ด้วย --no-preprocess)
   │    └─ engines/*.py       อ่านข้อความ → รายการ OCRLine (text, confidence, box)
   ├─ pipeline.py          รวมทุกหน้า → เซฟ <ชื่อไฟล์>_ocr.json / _ocr.txt
   └─ field_extraction.py  ดึง field → เซฟ <ชื่อไฟล์>_fields.json
```

---

## ใช้งานผ่าน VS Code

แนะนำให้ใช้ **VS Code** เพราะเปิดดูโครงสร้างไฟล์ แก้โค้ด และรันคำสั่งใน Terminal ได้ในที่เดียว

1. เปิด VS Code
2. ไปที่เมนู `File > Open Folder` แล้วเลือกโฟลเดอร์ `ocr_system`
3. เปิด Terminal: `Terminal > New Terminal`

หลังจากนี้ให้พิมพ์คำสั่งต่าง ๆ ใน Terminal ของ VS Code ได้เลย

---

## Installation

แนะนำใช้ Python 3.10–3.12 เช็กเวอร์ชันก่อน:

```bash
python --version
```

บน Windows บางเครื่องอาจต้องใช้ `py --version`

### 1. สร้าง Virtual Environment

Virtual Environment คือพื้นที่แยกสำหรับติดตั้ง package ของโปรเจกต์นี้โดยเฉพาะ เพื่อไม่ให้ชนกับโปรเจกต์อื่น

```bash
cd ocr_system
python -m venv .venv
```

ถ้าใช้ Windows แล้วคำสั่ง `python` ไม่ได้ ให้ลองใช้ `py -m venv .venv`

### 2. เปิดใช้งาน Virtual Environment (ต้องทำทุกครั้งที่เปิด Terminal ใหม่)

macOS / Linux:
```bash
source .venv/bin/activate
```

Windows CMD:
```bash
.venv\Scripts\activate
```

Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```

ถ้า PowerShell ขึ้น error เรื่อง policy ให้รัน `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` แล้วลอง activate ใหม่

ถ้าสำเร็จ จะเห็น `(.venv)` ขึ้นต้นบรรทัด

> **ถ้าเจอ `No module named 'ocr_system'`** แปลว่ายังไม่ได้ activate `.venv` (กำลังใช้ Python ของเครื่องแทน) ให้รันคำสั่ง activate ด้านบนก่อน

### 3. ติดตั้ง Python Packages

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` ทำให้เรียกใช้ด้วย `python -m ocr_system.cli` ได้

### 4. ติดตั้งโปรแกรมภายนอก

**Tesseract + Poppler** (Poppler ใช้แปลง PDF เป็นภาพ)

macOS:
```bash
brew install tesseract tesseract-lang poppler
```

Ubuntu / Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-tha poppler-utils
```

Windows: ติดตั้ง Tesseract OCR จาก UB Mannheim build และเลือกภาษา English + Thai ระหว่างติดตั้ง

ตรวจสอบว่ามีภาษา `tha` และ `eng`:
```bash
tesseract --list-langs
```

**llama.cpp** (จำเป็นเฉพาะ engine `surya`)

macOS:
```bash
brew install llama.cpp
```

Linux ที่มี NVIDIA GPU Surya จะใช้ vLLM แทนได้ ไม่ต้องใช้ llama.cpp

### 5. โมเดลที่ดาวน์โหลดอัตโนมัติตอนใช้ครั้งแรก

| engine | ขนาดโดยประมาณ | เก็บไว้ที่ |
|---|---|---|
| paddle | ~100MB | `~/.paddlex/official_models/` |
| easyocr | ~300MB | `~/.EasyOCR/model/` |
| doctr | ~100MB | `~/.cache/doctr/models/` |
| surya | ~1.5GB | `~/.cache/huggingface/hub/` |

ครั้งแรกจะช้าเพราะต้องดาวน์โหลด ครั้งต่อไปจะใช้โมเดลที่โหลดไว้แล้ว

---

## Usage

มี 3 คำสั่งหลัก:

| คำสั่ง | ใช้ทำอะไร |
|---|---|
| `ocr` | OCR ไฟล์เดียว (รูปหรือ PDF) |
| `evaluate` | วัดผล OCR ไฟล์เดียวเทียบกับ ground truth |
| `benchmark` | วัดผลหลาย engine × ทุกไฟล์ในโฟลเดอร์ แล้วสรุปเป็นตาราง |

ดูวิธีใช้ทั้งหมดได้ด้วย:
```bash
python -m ocr_system.cli --help
python -m ocr_system.cli ocr --help
python -m ocr_system.cli benchmark --help
```

---

## 1. คำสั่ง `ocr`

```bash
python -m ocr_system.cli ocr <ไฟล์> --engine <engine> [options]
```

ตัวอย่างแต่ละ engine:

```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --languages tha+eng
python -m ocr_system.cli ocr data/input/71010001.pdf --engine paddle --paddle-lang th
python -m ocr_system.cli ocr data/input/71010001.pdf --engine easyocr --easyocr-langs th,en
python -m ocr_system.cli ocr data/input/72100002.pdf --engine doctr
python -m ocr_system.cli ocr data/input/71010001.pdf --engine surya
python -m ocr_system.cli ocr data/input/71010001.pdf --engine ensemble
```

Tesseract แบบภาษาเดียว:
```bash
python -m ocr_system.cli ocr data/input/72100002.pdf --engine tesseract --languages eng
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --languages tha
```

PaddleOCR สำหรับเอกสารอังกฤษ:
```bash
python -m ocr_system.cli ocr data/input/72100002.pdf --engine paddle --paddle-lang en
```

PaddleOCR แบบโมเดลใหญ่ (อาจแม่นกว่า แต่ช้ามาก ~200 วินาที/หน้าบน CPU):
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine paddle --paddle-det-model PP-OCRv5_server_det
```

TrOCR (อ่านได้ทีละบรรทัด ไม่เหมาะกับเอกสารทั้งหน้า):
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine trocr --device cpu
```

ดูรูปหลัง preprocess (เซฟที่ `outputs/debug/page_001_preprocessed.png`):
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --save-debug-images
```

OCR แบบไม่ preprocess รูป:
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --no-preprocess
```

เซฟผลไปโฟลเดอร์อื่น:
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --output-dir outputs/71010001
```

### Options ทั้งหมดของ `ocr`

| option | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--engine` | `ensemble` | `paddle`, `tesseract`, `easyocr`, `doctr`, `surya`, `trocr`, `ensemble` |
| `--output-dir` | `outputs` | โฟลเดอร์เก็บผลลัพธ์ |
| `--languages` | `tha+eng` | ภาษาของ Tesseract เช่น `tha`, `eng`, `tha+eng` |
| `--paddle-lang` | `th` | ภาษาของ PaddleOCR เช่น `th`, `en` |
| `--paddle-det-model` | `PP-OCRv5_mobile_det` | โมเดลหาตำแหน่งข้อความของ Paddle (`PP-OCRv5_server_det` = ใหญ่/ช้า) |
| `--easyocr-langs` | `th,en` | ภาษาของ EasyOCR คั่นด้วย `,` (ภาษาไทยคู่ได้กับ `en` เท่านั้น) |
| `--dpi` | `300` | ความละเอียดตอนแปลง PDF เป็นรูป |
| `--no-preprocess` | ปิด | ไม่ทำความสะอาดรูปก่อน OCR |
| `--no-deskew` | ปิด | ไม่หมุนแก้รูปเอียง |
| `--save-debug-images` | ปิด | เซฟรูปหลัง preprocess ไว้ที่ `outputs/debug/` |
| `--min-confidence` | `0.0` | ตัดข้อความที่ความมั่นใจต่ำกว่าค่านี้ (0–1) |
| `--device` | `cpu` | `cpu` หรือ `cuda` (สำหรับ easyocr, doctr, trocr) |

### ผลลัพธ์ของ `ocr`

```text
outputs/71010001_ocr.json     ผล OCR แบบละเอียด: text, confidence, ตำแหน่ง, หน้า
outputs/71010001_ocr.txt      ข้อความ OCR รวมทั้งหมด อ่านง่าย
outputs/71010001_fields.json  field ที่ดึงได้ เช่น วันที่ รหัส เบอร์โทร
outputs/pages/                ภาพแต่ละหน้าที่แปลงจาก PDF
outputs/debug/                ภาพหลัง preprocess (เฉพาะเมื่อใส่ --save-debug-images)
```

---

## 2. คำสั่ง `evaluate`

วัดผล OCR ไฟล์เดียวเทียบกับ ground truth (ต้องรัน `ocr` ก่อน):

```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract
python -m ocr_system.cli evaluate data/ground_truth/Json_71010001_th.json outputs/71010001_ocr.json
```

เปลี่ยนที่เซฟผล (ค่าเริ่มต้น `outputs/evaluation_result.json`):
```bash
python -m ocr_system.cli evaluate data/ground_truth/Json_71010001_th.json outputs/71010001_ocr.json --output outputs/eval_71010001.json
```

รองรับ ground truth 2 แบบ:

**แบบที่ 1: ใบเกรดแบบมีโครงสร้าง** (ไฟล์ใน `data/ground_truth/`) มี `header_detail`, `transcript_detail`, `footer_detail`
ระบบจะดึงทุกค่าออกมา (ชื่อมหาวิทยาลัย, ชื่อวิชา, เกรด, ...) แล้วเช็กว่าแต่ละค่าอยู่ในข้อความ OCR หรือไม่ (ไม่สนช่องว่างและตัวเล็ก/ใหญ่)

| metric | ความหมาย |
|---|---|
| `field_recall` | สัดส่วนค่าที่หาเจอ **ยิ่งสูงยิ่งดี ใช้ดูเป็นหลัก** |
| `field_found` / `field_total` | จำนวนค่าที่หาเจอ / จำนวนค่าทั้งหมด (ไม่นับค่าที่ยาว 1 ตัวอักษร) |
| `cer` | อัตราตัวอักษรผิด ยิ่งต่ำยิ่งดี ใช้ดูคร่าว ๆ เพราะลำดับใน JSON ไม่ตรงกับบนกระดาษ |
| `missing_values` | รายการค่าที่หาไม่เจอ ใช้ดูว่า OCR พลาดตรงไหน |

**แบบที่ 2: ข้อความเต็ม** เช่น `data/ground_truth/example_ground_truth.json`
```json
{
  "71010001.pdf": "ข้อความจริงทั้งหมดในเอกสาร"
}
```
ได้ metric `cer`, `wer` (อัตราคำผิด) และ `exact_match` (ตรงทั้งหมดหรือไม่)

---

## 3. คำสั่ง `benchmark` — วัดผล engine แข่งกัน

รันหลาย engine กับทุกไฟล์ในโฟลเดอร์ที่มี ground truth แล้วสรุปผลเป็นตาราง

ทดลองเร็ว ๆ 2 ไฟล์แรก:
```bash
python -m ocr_system.cli benchmark --limit 2
```

วัดผลทั้งหมดในชุดที่ 1 (ค่าเริ่มต้น: paddle, tesseract, easyocr, doctr, ensemble):
```bash
python -m ocr_system.cli benchmark
```

วัดผลชุด G:
```bash
python -m ocr_system.cli benchmark --input-dir data/input_G --ground-truth-dir data/ground_truth_G --output-dir outputs/benchmark_G
```

เลือก engine เอง:
```bash
python -m ocr_system.cli benchmark --engines tesseract,easyocr,doctr
```

รวม Surya ด้วย (ต้องติดตั้ง llama.cpp ก่อน):
```bash
python -m ocr_system.cli benchmark --engines paddle,tesseract,easyocr,doctr,surya,ensemble
```

ไม่ preprocess รูป:
```bash
python -m ocr_system.cli benchmark --no-preprocess --output-dir outputs/benchmark_nopre
```

### Options ของ `benchmark`

| option | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--input-dir` | `data/input` | โฟลเดอร์เอกสาร |
| `--ground-truth-dir` | `data/ground_truth` | โฟลเดอร์เฉลย (จับคู่ `71010001.pdf` ↔ `Json_71010001_th.json`) |
| `--output-dir` | `outputs/benchmark` | โฟลเดอร์ผลลัพธ์ |
| `--engines` | `paddle,tesseract,easyocr,doctr,ensemble` | engine ที่จะเทียบ คั่นด้วย `,` |
| `--limit` | ทั้งหมด | ใช้แค่ N ไฟล์แรก (ไว้ทดสอบ) |

และใช้ option ของ `ocr` ได้ทั้งหมด เช่น `--languages`, `--paddle-lang`, `--paddle-det-model`, `--easyocr-langs`, `--dpi`, `--no-preprocess`, `--no-deskew`, `--min-confidence`, `--device`

### ผลลัพธ์ของ `benchmark`

```text
outputs/benchmark/summary.csv    ตารางสรุปต่อ engine (ดูอันนี้ก่อน)
outputs/benchmark/results.csv    ผลรายไฟล์ × ราย engine
outputs/benchmark/<engine>/      ข้อความ OCR ของแต่ละ engine (_ocr.json / _ocr.txt)
outputs/benchmark/pages/         ภาพแต่ละหน้า
```

- ไฟล์ CSV เปิดใน Excel แล้วภาษาไทยแสดงถูกต้อง
- `results.csv` เซฟทุกครั้งที่ทำเสร็จ 1 ไฟล์ ถ้ากด Ctrl+C กลางทางก็ยังได้ผลเท่าที่ทำไปแล้ว
- `summary.csv` แยกผลเป็น `all` (ทุกไฟล์), `th` (เอกสารไทย), `en` (เอกสารอังกฤษ) ตามชื่อไฟล์ ground truth `_th` / `_en`
- `field_recall` ในสรุปคิดจาก ค่าที่หาเจอทั้งหมด ÷ ค่าทั้งหมด ของทุกไฟล์รวมกัน

### ข้อควรรู้ตอนอ่านผล

- **ensemble**: เอาข้อความของ Paddle + Tesseract + EasyOCR มารวมกัน ข้อความจึงยาวกว่าตัวอื่นเกือบ 3 เท่า ทำให้ `cer` เกิน 1 และ `field_recall` สูงขึ้นเพราะ "ได้ลองหลายครั้ง" ใช้ `cer` ของ ensemble เทียบกับตัวอื่นไม่ได้
- **ensemble ใน benchmark** ใช้ผลของ engine สมาชิกซ้ำ ไม่ได้รันใหม่ เวลาที่แสดงจึงเป็นผลรวมเวลาของสมาชิก
- **doctr** อ่านภาษาไทยไม่ได้ ให้ดูผลที่แถว `en` เป็นหลัก
- ถ้า engine ไหนพังกับไฟล์ไหน จะบันทึกในคอลัมน์ `error` แล้วทำต่อไฟล์ถัดไป

---

## ปรับแต่ง / แก้ไขได้ที่ไหน

| อยากทำอะไร | แก้ที่ไหน |
|---|---|
| เปลี่ยน engine, ภาษา, dpi, ปิด preprocess | ใช้ option ตอนรัน ไม่ต้องแก้โค้ด |
| เปลี่ยนค่าเริ่มต้น | `config.py` และ default ใน `cli.py` |
| ปรับการทำความสะอาดรูป | `preprocessing.py` |
| ปรับ Tesseract (เช่น `psm`) | `engines/tesseract_engine.py` |
| เปลี่ยนสมาชิกของ ensemble | `ENSEMBLE_MEMBERS` ใน `engine_factory.py` |
| เปลี่ยนวิธีรวมผลของ ensemble | `merge()` ใน `engines/ensemble_engine.py` |
| ดึง field เพิ่ม (ชื่อ, คณะ, เกรด) | `field_extraction.py` |
| เปลี่ยนวิธีวัดผล | `evaluation.py` |
| เพิ่ม engine ใหม่ | สร้างไฟล์ใน `engines/` ที่สืบทอด `BaseOCREngine` แล้วเพิ่มใน `engine_factory.py` และ `ENGINES` ใน `cli.py` |

ไม่ควรแก้: `.venv/`, `src/ocr_system.egg-info/`, `__pycache__/` และไฟล์ใน `outputs/` (โปรแกรมเขียนทับเอง)

---

## Output JSON Format

```json
{
  "source_path": "data/input/71010001.pdf",
  "engine": "tesseract",
  "text": "--- Page 1 ---\n...",
  "pages": [
    {
      "page": 1,
      "text": "...",
      "lines": [
        {
          "text": "ข้อความที่ OCR อ่านได้",
          "confidence": 0.95,
          "box": [[0, 0], [100, 0], [100, 30], [0, 30]],
          "engine": "tesseract",
          "page": 1
        }
      ],
      "image_path": "outputs/pages/71010001_page_001.jpg"
    }
  ]
}
```

`confidence` ของ `surya` และ `trocr` เป็น `null` เพราะ engine ไม่ได้ให้ค่านี้มา

---

## Troubleshooting

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `No module named 'ocr_system'` | ยังไม่ได้ `source .venv/bin/activate` |
| `Unsupported file type` / `Cannot read image` | path ไฟล์ผิด หรือไฟล์ไม่มีอยู่จริง ตรวจด้วย `ls data/input` |
| `KeyError: No ground truth found` | ไฟล์ ground truth ไม่ตรงกับเอกสาร หรือรูปแบบไม่ถูกต้อง |
| `llama-server binary not found` | ยังไม่ได้ `brew install llama.cpp` (ใช้กับ surya) |
| รันครั้งแรกช้ามาก | กำลังดาวน์โหลดโมเดล (ดูตารางขนาดในหัวข้อ Installation) |
| Paddle ช้ามาก | ใช้โมเดล `PP-OCRv5_server_det` อยู่ เปลี่ยนกลับเป็นค่าเริ่มต้น `PP-OCRv5_mobile_det` |
