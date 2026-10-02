# Thai-English OCR System

โปรเจกต์นี้เป็น OCR pipeline สำหรับเอกสารภาพเดี่ยวและหลายหน้า เช่น `.jpg`, `.png`, `.tif`, `.pdf` โดยรองรับเอกสารภาษาไทยและอังกฤษปนกัน
ดึงข้อมูลใบเกรด (transcript) ออกมาเป็น JSON แบบเดียวกับ ground truth และมีคำสั่ง `benchmark` สำหรับวัดผล OCR หลาย engine แข่งกัน

> **เริ่มต้นเร็ว:** ดูหัวข้อ [สรุปคำสั่งทั้งหมด](#สรุปคำสั่งทั้งหมด) และ [ค่าที่แนะนำ](#ค่าที่แนะนำจากผลทดลอง)

OCR engines ที่มีให้ (`--engine`):

| engine | ภาษาไทย | ความเร็ว (CPU, ต่อหน้า) | หมายเหตุ |
|---|---|---|---|
| `tesseract` | ได้ | ~1.5 วินาที | เร็วที่สุด ต้องติดตั้งโปรแกรม Tesseract แยก |
| `paddle` | ได้ | ~10–20 วินาที | PaddleOCR 3.x ค่าเริ่มต้นใช้โมเดล mobile |
| `easyocr` | ได้ | ~16 วินาที | ภาษาไทยใช้คู่กับอังกฤษได้เท่านั้น (`th,en`) |
| `doctr` | **ไม่ได้** | ~2 วินาที | อ่านได้แค่ภาษาอังกฤษ/ตัวอักษรละติน เหมาะกับเอกสาร `_en` |
| `surya` | ได้ | ~90 วินาที | Surya OCR 2 (โมเดล AI อ่านทั้งหน้า) อ่านแม่นที่สุด ต้องติดตั้ง `llama.cpp` แยก |
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
│   ├── ground_truth_G/        # เฉลยของชุด G
│   ├── Augmentation_input/    # ชุดที่ 1 แบบ augment (สร้างด้วยคำสั่ง augment)
│   │   ├── images/            #   71010001_original.jpg, 71010001_rotation.jpg, ...
│   │   ├── ground_truth/      #   Json_71010001_rotation_th.json, ... (label = copy ของเฉลยต้นฉบับ)
│   │   └── manifest.csv       #   ภาพไหนมาจากไฟล์ไหน augment แบบไหน ค่าที่สุ่มได้
│   └── Augmentation_input_G/  # ชุด G แบบ augment (โครงเดียวกัน)
├── outputs/                   # ผลลัพธ์ OCR / evaluation / benchmark (สร้างอัตโนมัติ ลบได้)
└── src/
    └── ocr_system/
        ├── cli.py             # command line: ocr / extract / evaluate / benchmark / augment
        ├── config.py          # ค่าตั้งค่าหลัก (OCRConfig)
        ├── pipeline.py        # OCR pipeline หลัก: โหลด → preprocess → OCR → เซฟผล
        ├── document_loader.py # โหลดภาพ / แปลง PDF เป็นภาพทีละหน้า
        ├── preprocessing.py   # resize, denoise, contrast, deskew, threshold
        ├── layout.py          # layout analysis: แบ่งหน้าเป็น header / info / ตาราง / footer
        ├── engine_factory.py  # เลือก OCR engine ตามชื่อ + กำหนดสมาชิกของ ensemble
        ├── evaluation.py      # วัดผล: field_recall, CER, WER, field_accuracy (ทีละช่อง)
        ├── benchmark.py       # วัดผลหลาย engine × หลายไฟล์ แล้วสรุปเป็นตาราง
        ├── augmentation.py    # สร้าง dataset แบบ augment + label สำหรับทดสอบความทนทาน
        ├── field_extraction.py# ดึง field เช่น email, date, id, phone ด้วย regex
        ├── transcript_extraction.py # แปลงผล OCR เป็น JSON ใบเกรดแบบ ground truth
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
   │    ├─ layout.py          แบ่งหน้าเป็นส่วน ๆ (เฉพาะเมื่อใส่ --layout)
   │    └─ engines/*.py       อ่านข้อความ → รายการ OCRLine (text, confidence, box, region)
   ├─ pipeline.py          รวมทุกหน้า → เซฟ <ชื่อไฟล์>_ocr.json / _ocr.txt
   ├─ field_extraction.py  ดึง field → เซฟ <ชื่อไฟล์>_fields.json
   └─ transcript_extraction.py  จัดข้อมูลใบเกรด → เซฟ <ชื่อไฟล์>_transcript.json
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

## สรุปคำสั่งทั้งหมด

ทุกคำสั่งต้องเปิด `.venv` ก่อน (`source .venv/bin/activate`)

| อยากได้อะไร | คำสั่ง | ผลลัพธ์ |
|---|---|---|
| อ่านข้อความจากเอกสาร 1 ไฟล์ | `python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract` | `outputs/71010001_ocr.txt` (ข้อความ), `_ocr.json` (ละเอียด + ตำแหน่ง) |
| ข้อมูลใบเกรดเป็น JSON | คำสั่ง `ocr` ด้านบนสร้างให้อัตโนมัติ | `outputs/71010001_transcript.json` |
| ดึงข้อมูลใบเกรดใหม่โดยไม่ต้อง OCR ซ้ำ | `python -m ocr_system.cli extract outputs/71010001_ocr.json` | `outputs/71010001_transcript.json` |
| ข้อความแยกตามส่วนของหน้า | `python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --layout` | `_ocr.txt` มีหัวข้อ `[header]`, `[info_left]`, `[table_col1]`, ... |
| ดูว่ารูปถูกทำความสะอาด / แบ่ง layout ยังไง | เพิ่ม `--save-debug-images` ในคำสั่ง `ocr` | `outputs/debug/page_001_preprocessed.png`, `page_001_layout.png` |
| วัดผล 1 ไฟล์เทียบเฉลย | `python -m ocr_system.cli evaluate data/ground_truth/Json_71010001_th.json outputs/71010001_ocr.json` | สรุปบนจอ + `outputs/evaluation_result.json` |
| เทียบหลาย engine ทุกไฟล์ | `python -m ocr_system.cli benchmark` | `outputs/benchmark/summary.csv`, `results.csv` |
| เทียบชุด G | `python -m ocr_system.cli benchmark --input-dir data/input_G --ground-truth-dir data/ground_truth_G --output-dir outputs/benchmark_G` | `outputs/benchmark_G/summary.csv` |
| สร้าง dataset augment + label | `python -m ocr_system.cli augment` | `data/Augmentation_input/` (images, ground_truth, manifest.csv) |
| วัดความทนทานต่อการบิดภาพ | `python -m ocr_system.cli benchmark --input-dir data/Augmentation_input/images --ground-truth-dir data/Augmentation_input/ground_truth --output-dir outputs/benchmark_aug` | `summary.csv` แยกผลตาม augmentation |
| ดูวิธีใช้ทุก option | `python -m ocr_system.cli <คำสั่ง> --help` | |

### ขั้นตอนที่ใช้บ่อย

ลองแก้ `transcript_extraction.py` แล้ววัดผลว่าดีขึ้นไหม (OCR ครั้งเดียว ดึงใหม่กี่ครั้งก็ได้):
```bash
python -m ocr_system.cli ocr data/input/72100002.pdf --engine tesseract --deskew-only
python -m ocr_system.cli extract outputs/72100002_ocr.json
python -m ocr_system.cli evaluate data/ground_truth/Json_72100002_en.json outputs/72100002_transcript.json
```

วัดผลทั้งชุดด้วยค่าที่แนะนำ:
```bash
python -m ocr_system.cli benchmark --engines tesseract,paddle --deskew-only
```

---

## ค่าที่แนะนำจากผลทดลอง

ผลจากเอกสารชุดที่ 1 (24 ไฟล์, ไทย 12 + อังกฤษ 12):

| เรื่อง | ผลทดลอง | แนะนำ |
|---|---|---|
| Preprocessing | ไม่ preprocess ดีกว่าทุก engine (Tesseract 0.790 → **0.820** field_recall) เพราะ PDF ชุดนี้สร้างจากคอมพิวเตอร์ คมชัดอยู่แล้ว การทำความสะอาดรูปไปทำลายสระ/วรรณยุกต์ไทย แต่ถ้าภาพเอียง การดึงข้อมูลใบเกรดพังหนัก (ภาพหมุน 0.5–3°: field_accuracy 0.241) | ใส่ `--deskew-only` (หมุนให้ตรงอย่างเดียว): ภาพหมุน 0.241 → **0.702**, ภาพตรงได้เท่าเดิม |
| Layout | `--layout` เพิ่ม field_recall ทุก engine (Tesseract 0.820 → **0.838**, doctr 0.675 → 0.701) แต่ field_accuracy ของ transcript extraction ไม่ได้ดีขึ้น (ลดลงเล็กน้อย) | ใส่ `--layout` เมื่อต้องการข้อความ, ไม่ใส่เมื่อต้องการ JSON ใบเกรด (จนกว่า extraction จะใช้ region) |
| Engine (อ่านข้อความ) | Surya แม่นสุด (0.885 วัดแบบเปิด preprocess แต่ ~90 วินาที/ไฟล์), Tesseract 0.820 (~3 วินาที), Paddle 0.812 | Tesseract สำหรับทดลองเร็ว ๆ, Surya เมื่อต้องการความแม่น |
| Engine (ดึงข้อมูลใบเกรด) | `transcript_extraction.py` ออกแบบสำหรับผล Tesseract (ทีละคำ): field_accuracy Tesseract **0.707** (ไทย 0.623, อังกฤษ 0.810; ชุด G **0.849**), Paddle 0.556, docTR 0.190, EasyOCR 0.182, Surya 0.061 | ใช้ Tesseract |

> ค่าเริ่มต้นของโปรแกรมยังเป็น preprocess = เปิด, layout = ปิด, engine = ensemble ต้องใส่ option เองตามตารางด้านบน

---

## Usage

มี 5 คำสั่งหลัก:

| คำสั่ง | ใช้ทำอะไร |
|---|---|
| `ocr` | OCR ไฟล์เดียว (รูปหรือ PDF) + สร้าง `_transcript.json` |
| `extract` | สร้าง `_transcript.json` จากผล OCR (`_ocr.json`) ที่มีอยู่แล้ว |
| `evaluate` | วัดผล OCR / transcript ไฟล์เดียวเทียบกับ ground truth (ข้อความ + ทีละช่อง) |
| `benchmark` | วัดผลหลาย engine × ทุกไฟล์ในโฟลเดอร์ แล้วสรุปเป็นตาราง |
| `augment` | สร้าง dataset ภาพบิด (rotation, crop, ...) พร้อม label สำหรับทดสอบความทนทาน |

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

แบ่งหน้าเป็นส่วน ๆ ด้วย layout analysis (ดูหัวข้อ Layout Analysis ด้านล่าง):
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine tesseract --layout --no-preprocess
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
| `--deskew-only` | ปิด | หมุนแก้รูปเอียง**อย่างเดียว** ไม่ทำขั้นอื่น (ลด noise, contrast, ขาว/ดำ) ภาพยังเป็นสี/เทาเหมือนเดิม **แนะนำแทน `--no-preprocess`** |
| `--save-debug-images` | ปิด | เซฟรูปหลัง preprocess ไว้ที่ `outputs/debug/` |
| `--min-confidence` | `0.0` | ตัดข้อความที่ความมั่นใจต่ำกว่าค่านี้ (0–1) |
| `--device` | `cpu` | `cpu` หรือ `cuda` (สำหรับ easyocr, doctr, trocr) |
| `--layout` | ปิด | แบ่งหน้าเป็นส่วน (header, info, คอลัมน์ตาราง, footer) |
| `--layout-mode` | `auto` | `assign`, `crop` หรือ `auto` (ดูหัวข้อ Layout Analysis) |

### ผลลัพธ์ของ `ocr`

```text
outputs/71010001_ocr.json     ผล OCR แบบละเอียด: text, confidence, ตำแหน่ง, หน้า
outputs/71010001_ocr.txt      ข้อความ OCR รวมทั้งหมด อ่านง่าย
outputs/71010001_fields.json  field ที่ดึงได้ เช่น วันที่ รหัส เบอร์โทร
outputs/71010001_transcript.json  ข้อมูลใบเกรดแบบเดียวกับ ground truth (header_detail / transcript_detail / footer_detail)
outputs/pages/                ภาพแต่ละหน้าที่แปลงจาก PDF
outputs/debug/                ภาพหลัง preprocess และภาพตีกรอบ layout (เฉพาะเมื่อใส่ --save-debug-images)
```

---

## 2. คำสั่ง `extract` — ดึงข้อมูลใบเกรดเป็น JSON

คำสั่ง `ocr` สร้าง `_transcript.json` ให้อัตโนมัติอยู่แล้ว ใช้ `extract` เมื่อมี `_ocr.json` อยู่แล้ว และอยากดึงใหม่ (เช่น หลังแก้ `transcript_extraction.py`) โดยไม่ต้อง OCR ใหม่:

```bash
python -m ocr_system.cli extract outputs/72100002_ocr.json
```

ได้ไฟล์ `outputs/72100002_transcript.json` เลือกที่เซฟ หรือบังคับภาษา (ค่าเริ่มต้นตรวจจากข้อความเอง):
```bash
python -m ocr_system.cli extract outputs/72100002_ocr.json --output outputs/my_transcript.json --language en
```

ต้องใช้ไฟล์ `_ocr.json` (ไม่ใช่ `.txt`) เพราะการแยกแถววิชาใช้ตำแหน่ง box ของแต่ละคำ

---

## 3. คำสั่ง `evaluate`

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

### วัดผลทีละช่อง (field accuracy)

ถ้า ground truth เป็นใบเกรดแบบมีโครงสร้าง `evaluate` จะดึงข้อมูลใบเกรดจากผล OCR แล้วเทียบ**ทีละช่อง**ให้ด้วย หรือส่งไฟล์ `_transcript.json` เข้าไปตรง ๆ ก็ได้:

```bash
python -m ocr_system.cli evaluate data/ground_truth/Json_72100002_en.json outputs/72100002_ocr.json
python -m ocr_system.cli evaluate data/ground_truth/Json_72100002_en.json outputs/72100002_transcript.json
```

ตัวอย่างผล (`72100002` ด้วย Tesseract):
```text
Text:   field_recall=0.795 (31/39)  cer=0.561
Fields: field_accuracy=0.932 (55/59)
  header_detail      0.923 (12/13)     ชื่อสถาบัน, ชื่อ, วันเกิด, ปริญญา, ...
  semester           1.000 (4/4)       year, sem_num, GPA, GPS
  subject            1.000 (36/36)     subject_id, subject_name, credit, grade_earn
  transcript_detail  1.000 (2/2)       total_credits_earned, cumulative_gpa
  footer_detail      0.250 (1/4)       วันที่ออกเอกสาร, ลายเซ็น, ตำแหน่ง
  x header_detail.uni_name: expected="kingmongkut'sinstituteoftechnologyladkrabang" predicted="kingmongkuti'sinstituteoftechnologyladkrabang"
  ...
Full result (incl. every mismatched field): outputs/evaluation_result.json
```

- `Text:` = OCR อ่านข้อความออกไหม (ส่งไฟล์ `_ocr.json` เท่านั้นถึงจะมีบรรทัดนี้)
- `Fields:` = JSON ที่ดึงได้ถูกทีละช่องไหม

- ทุกค่าที่ไม่ใช่ `null` ใน ground truth นับเป็น 1 ช่อง ช่องที่ ground truth เป็น `null` ไม่นับ
- ถูกเมื่อค่าที่ตำแหน่งเดียวกันเท่ากัน หลังตัดช่องว่างและเป็นตัวพิมพ์เล็ก (`3` กับ `"3"` ถือว่าเท่ากัน)
- เทียบตามตำแหน่ง: ถ้าวิชาเลื่อนไปหนึ่งแถว ทุกวิชาถัดไปจะผิดหมด
- รายการช่องที่ผิดทั้งหมดอยู่ใน `mismatches` ของไฟล์ผลลัพธ์ (`outputs/evaluation_result.json`)

---

## 4. คำสั่ง `benchmark` — วัดผล engine แข่งกัน

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

และใช้ option ของ `ocr` ได้ทั้งหมด เช่น `--languages`, `--paddle-lang`, `--paddle-det-model`, `--easyocr-langs`, `--dpi`, `--no-preprocess`, `--no-deskew`, `--min-confidence`, `--device`, `--layout`, `--layout-mode`

เทียบผลแบบมี/ไม่มี layout:
```bash
python -m ocr_system.cli benchmark --no-preprocess --output-dir outputs/benchmark_nolayout
python -m ocr_system.cli benchmark --no-preprocess --layout --output-dir outputs/benchmark_layout
```

### ผลลัพธ์ของ `benchmark`

```text
outputs/benchmark/summary.csv    ตารางสรุปต่อ engine (ดูอันนี้ก่อน)
outputs/benchmark/results.csv    ผลรายไฟล์ × ราย engine
outputs/benchmark/<engine>/      ผลของแต่ละ engine (_ocr.json / _ocr.txt / _transcript.json)
outputs/benchmark/pages/         ภาพแต่ละหน้า
```

- ไฟล์ CSV เปิดใน Excel แล้วภาษาไทยแสดงถูกต้อง
- `results.csv` เซฟทุกครั้งที่ทำเสร็จ 1 ไฟล์ ถ้ากด Ctrl+C กลางทางก็ยังได้ผลเท่าที่ทำไปแล้ว
- `summary.csv` แยกผลเป็น `all` (ทุกไฟล์), `th` (เอกสารไทย), `en` (เอกสารอังกฤษ) ตามชื่อไฟล์ ground truth `_th` / `_en` และแยกตาม augmentation เมื่อใช้กับ dataset จากคำสั่ง `augment`
- `field_recall` ในสรุปคิดจาก ค่าที่หาเจอทั้งหมด ÷ ค่าทั้งหมด ของทุกไฟล์รวมกัน (วัดว่า **OCR อ่านข้อความออก**ไหม)
- `field_accuracy` = ช่องที่ดึงถูกทั้งหมด ÷ ช่องทั้งหมด ของทุกไฟล์รวมกัน (วัดว่า **OCR + transcript extraction ได้ JSON ถูก**ไหม)

คอลัมน์ใน `summary.csv`:

| คอลัมน์ | ความหมาย |
|---|---|
| `group` | `all` / `th` / `en` และถ้าเป็น dataset augment: `original` / `rotation` / `crop` / ... |
| `engine` | ชื่อ engine |
| `documents` / `errors` | จำนวนไฟล์ที่สำเร็จ / พัง |
| `field_recall` | OCR อ่านค่าใน ground truth ออกกี่ % |
| `avg_cer` | อัตราตัวอักษรผิดเฉลี่ย (ดูคร่าว ๆ) |
| `field_accuracy` | JSON ใบเกรดถูกทีละช่องกี่ % |
| `avg_seconds_per_doc` | เวลาเฉลี่ยต่อไฟล์ |

### ข้อควรรู้ตอนอ่านผล

- **ensemble**: เอาข้อความของ Paddle + Tesseract + EasyOCR มารวมกัน ข้อความจึงยาวกว่าตัวอื่นเกือบ 3 เท่า ทำให้ `cer` เกิน 1 และ `field_recall` สูงขึ้นเพราะ "ได้ลองหลายครั้ง" ใช้ `cer` ของ ensemble เทียบกับตัวอื่นไม่ได้
- **ensemble ใน benchmark** ใช้ผลของ engine สมาชิกซ้ำ ไม่ได้รันใหม่ เวลาที่แสดงจึงเป็นผลรวมเวลาของสมาชิก
- **doctr** อ่านภาษาไทยไม่ได้ ให้ดูผลที่แถว `en` เป็นหลัก
- ถ้า engine ไหนพังกับไฟล์ไหน จะบันทึกในคอลัมน์ `error` แล้วทำต่อไฟล์ถัดไป

---

## 5. คำสั่ง `augment` — Augmentation และ Labeling Dataset

สร้างภาพเอกสารที่ถูกบิดแบบเอกสารสแกนจริง เพื่อวัดว่า OCR แต่ละ engine ทนต่อสภาพเอกสารแย่ ๆ แค่ไหน

```bash
python -m ocr_system.cli augment
python -m ocr_system.cli augment --input-dir data/input_G --ground-truth-dir data/ground_truth_G --output-dir data/Augmentation_input_G
```

ต่อ 1 หน้า ได้ 7 ภาพ: ภาพต้นฉบับ (`original` ไว้เป็นตัวเทียบ) + augmentation 6 แบบ **แบบละ 1 ภาพ** (ไม่รวมหลายแบบในภาพเดียว เพื่อให้รู้ว่าการบิดแบบไหนทำให้ engine ไหนพัง)

| augmentation | ทำอะไร (สุ่มค่าในช่วงนี้) |
|---|---|
| `rotation` | หมุน ±0.5–3° ขยายกรอบภาพไม่ให้มุมกระดาษถูกตัด |
| `crop` | ตัดขอบกระดาษแต่ละด้านแบบสุ่ม **ไม่ตัดเข้าเนื้อหา** (เหลือขอบขาว ≥ 10 px รอบตัวหนังสือ) |
| `translation` | เลื่อนภาพ ≤ 3% ของขนาด แต่ไม่เกินขอบกระดาษที่มี (ตัวหนังสือไม่หลุดภาพ) |
| `brightness_contrast` | contrast ×0.75–1.25, ความสว่าง ±25% |
| `noise` | Gaussian noise σ = 5–20 |
| `dropout` | pixel 1–3% และจุดเล็ก ๆ 50–150 จุด (4–12 px) กลายเป็นสีขาว เหมือนหมึกจาง |

ผลลัพธ์ (`data/Augmentation_input/`):
```text
images/71010001_rotation.jpg               ภาพ (JPEG คุณภาพ 95)
ground_truth/Json_71010001_rotation_th.json label = copy ของ ground truth ต้นฉบับ (การบิดไม่เปลี่ยนเนื้อหาเอกสาร)
manifest.csv                                ทุกภาพ: ไฟล์ต้นฉบับ, augmentation, ค่าที่สุ่มได้ (เช่น {"angle_deg": -1.2}), ขนาดภาพ, seed
```

| option | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `--input-dir` / `--ground-truth-dir` | `data/input` / `data/ground_truth` | ชุดข้อมูลต้นฉบับ |
| `--output-dir` | `data/Augmentation_input` | ที่เก็บ dataset ใหม่ |
| `--augmentations` | ทั้ง 6 แบบ | เลือกบางแบบ เช่น `rotation,noise` |
| `--seed` | `42` | seed เดิม = ค่าสุ่มเดิม = ภาพเดิมทุกครั้ง |
| `--dpi` | `300` | ความละเอียดตอนแปลง PDF |
| `--no-original` | ปิด | ไม่ใส่ภาพต้นฉบับ |

วัดผลด้วย benchmark (ได้ตารางแยกตาม augmentation):
```bash
python -m ocr_system.cli benchmark --input-dir data/Augmentation_input/images --ground-truth-dir data/Augmentation_input/ground_truth --output-dir outputs/benchmark_aug --engines tesseract,paddle --deskew-only
```

ใช้ `--deskew-only` เพื่อให้ภาพ `rotation` ถูกหมุนกลับก่อน OCR ถ้าใช้ `--no-preprocess` ภาพหมุนจะได้ field_accuracy ต่ำมาก (Tesseract 0.241 vs 0.702) เพราะ transcript extraction จัดข้อความเป็นแถวตามแนวนอน

ปรับช่วงการสุ่มได้ที่ฟังก์ชันแต่ละแบบใน `augmentation.py` (`rotation()`, `crop()`, ...) เพิ่มแบบใหม่ได้โดยเขียนฟังก์ชัน `(image, rng) -> (image, params)` แล้วใส่ใน `AUGMENTATIONS`

---

## Transcript Extraction

`transcript_extraction.py` แปลงผล OCR (`_ocr.json`) เป็น JSON โครงเดียวกับ ground truth:

```text
header_detail       uni_name, uni_address, student_id, faculty_name, prename, name,
                    date_of_birth, admis_date, grad_date, grad_reason, degree, major, program, honor
transcript_detail   semesters[] → year, sem_num, GPA, GPS, subject[] → subject_id, subject_name, credit, grade_earn
                    total_credits_earned, cumulative_gpa
footer_detail       updated_at, by → by_signature, by_position, by_reg
```

วิธีทำงานคร่าว ๆ:
- **ภาษา:** ถ้าตัวอักษรไทย > 30% ของตัวอักษรทั้งหมด = เอกสารไทย
- **header / footer (อังกฤษ):** หา label (เช่น `Name`, `Date of Birth`) แล้วเอาคำถัดไปจนถึง label ถัดไป
- **header / footer / ยอดรวม (ไทย):** จัดกล่องข้อความเป็นแถว แล้วหา label ไทย (`ชื่อ-สกุล`, `รหัสประจำตัวนักศึกษา`, `วันเดือนปีเกิด`, `วันที่เข้าศึกษา`, `ชื่อปริญญา`, `วันที่สำเร็จการศึกษา`, `หลักสูตร`, `จำนวนหน่วยกิตที่สอบได้ทั้งหมด`, `คะแนนเฉลี่ยสะสม`, `วันที่ออกเอกสาร`) แบบทนต่อ OCR ผิด: เทียบเฉพาะพยัญชนะ/สระหลัก (ตัดวรรณยุกต์และสระบน/ล่าง ถือ ซ = ช) และใช้ fuzzy matching (`rapidfuzz`) ค่าของแต่ละ label = ข้อความจนถึง label ถัดไปในแถวเดียวกัน
- **หัวภาคเรียนไทย:** `ภาคการศึกษาที่ 1 ปีการศึกษา 2561` (และ `ภาคฤดูร้อน` = sem_num 0), GPS/GPA จาก `คะแนนเฉลี่ยประจำภาคการศึกษา : 2.33  คะแนนเฉลี่ย : 2.33`
- **เดือนไทย:** `3 กันยายน 2542` → `1999-09-03` แม้ชื่อเดือนสระหาย
- **หัวภาคเรียน:** จัดกล่องข้อความเป็นแถวตามตำแหน่ง แล้วหา `1st Semester, Academic Year 2019` (ชุดที่ 1) หรือ `1st Semester , 2021` (ชุด G) ในแต่ละแถว รองรับ OCR อ่าน `1st` เป็น `lst` / `Ist` (ใช้ได้ทั้ง engine ที่ให้ทีละคำและทีละบรรทัด)
- **ตารางสองซีก:** ใบเกรดยาวจะต่อจากตารางซีกซ้ายไปซีกขวา ระบบ "คลี่" ตาราง โดยย้ายข้อความซีกขวาไปต่อท้ายซีกซ้าย (และเลื่อน footer ลงไปท้ายสุด) จะได้อ่านจากบนลงล่างได้เป็นคอลัมน์เดียว
  - เส้นแบ่งกลางตาราง: ถ้าใช้ `--layout` ใช้ region `table_col1..N` (ครึ่งหลัง = ซีกขวา) ถ้าไม่ใช้ ประมาณจากรหัสวิชาซีกขวาหรือกึ่งกลางหน้า
  - ขอบบน: แถวหัวตาราง (`รายวิชา หน่วยกิต เกรด` / `Course ... Grade`) หรือหัวภาคเรียน/แถววิชาแรกถ้า OCR อ่านหัวตารางไม่ออก ขอบล่าง: แถว `วันที่ออกเอกสาร` / `Date of Issued`
  - รหัสวิชาซีกขวาที่ OCR อ่านเส้นคู่กลางตารางติดมา (`|01016238`, `101016454`) จะใช้ 8 หลักสุดท้าย
- **กล่องหลายคำ:** engine ที่ให้ผลทีละบรรทัด (paddle, easyocr, doctr) ถูกแยกเป็นคำ โดยประมาณตำแหน่ง x ของแต่ละคำจากตำแหน่งตัวอักษร
- **วิชา:** เลข 8 หลักที่อยู่ในตาราง (ใต้หัวภาคเรียนแรก / หัวตาราง) = รหัสวิชา แล้วอ่านแถวของรหัสนั้น**จากขวาไปซ้าย**: เกรด → หน่วยกิต → type (ชุด G: `Cr` / `Nc` / `Ad`) ที่เหลือคือชื่อวิชา
  - ชื่อวิชาที่ยาวจนขึ้นบรรทัดใหม่ (เช่น `RESEARCH METHODOLOGY AND ETHICS IN` / `NANOTECHNOLOGY`) รวมแถวถัดไปที่เยื้องอยู่ใต้ชื่อ จนเจอวิชาถัดไป / หัวภาคเรียน / แถวคะแนนเฉลี่ย
  - ใช้แถวแบบ visual row (ไม่ใช่ช่วง y ตายตัว) จึงได้สระบน/ล่างของภาษาไทยครบ และไม่ดึงสระของแถวถัดไปมาปน
  - แก้เกรดที่ OCR อ่านผิดเมื่ออยู่ในตำแหน่งเกรด (มีหน่วยกิตอยู่ทางซ้าย): `0`/`6` → c, `8` → b, `5`/`ธ`/`ร` → s, `1A` → a, `Bt` → b+ และ type `Gr`/`0`/`๐` → cr, `Ne` → nc
- **วิชาเทียบโอน:** `Transferred Credits` / `รายวิชาเทียบโอน` เป็นภาคเรียน `sem_num 0` ของปีที่เข้าศึกษา (ตาม ground truth)
- วันที่แปลงเป็น `YYYY-MM-DD` (พ.ศ. → ค.ศ.), ข้อความแปลงเป็นตัวพิมพ์เล็กไม่มีช่องว่างตาม ground truth

ข้อจำกัดตอนนี้:

| ข้อจำกัด | ผลกระทบ |
|---|---|
| ยังไม่รองรับ "รายวิชาเทียบโอน" / transfer courses (ground truth ใช้ sem_num 0) | ใบเกรดที่มีวิชาเทียบโอนได้ภาคเรียนเลื่อน |
| OCR อ่านวรรณยุกต์/สระบนล่างหาย (เช่น `สถาบน` แทน `สถาบัน`) | ค่าภาษาไทยหลายช่องผิดแม้หา label ถูก (เป็นปัญหาความแม่นของ OCR) |
| Surya คืนผลเป็น block ใหญ่หลายบรรทัด (แยกเป็นแถวไม่ได้) | Surya ได้ field_accuracy ต่ำมาก (~0.06) แม้อ่านข้อความแม่นที่สุด |
| เกรด `-` ที่ OCR อ่านไม่ออกเลย, หัวภาคเรียนที่ OCR อ่านพลาด | วิชา/ภาคเรียนนั้นผิดหรือรวมกับภาคก่อนหน้า |
| ตำแหน่งบางอย่างเป็นค่า pixel ตายตัว (เช่น ระยะห่างแถว 70 px, `y > 700`) | ออกแบบสำหรับ 300 dpi |
| ground truth บางช่องไม่ตรงกับเอกสาร (เช่น `updated_at`, `educationservice`, ชุด G บางไฟล์มีภาคเรียนสุดท้ายที่ไม่ได้พิมพ์ในเอกสาร) | บางช่องผิดเสมอ แก้ด้วยโค้ดไม่ได้ |

---

## Layout Analysis

ใส่ `--layout` เพื่อแบ่งแต่ละหน้าเป็นส่วน ๆ แบบ rule-based (ไม่ต้องใช้โมเดล ใช้เวลา < 0.1 วินาที/หน้า) อยู่ในไฟล์ `layout.py`

```text
header          ชื่อสถาบัน, ที่อยู่, ชื่อเอกสาร, คณะ
info_left       ชื่อ, วันเกิด, ปริญญา, หลักสูตร
info_right      รหัสนักศึกษา, วันที่เข้าศึกษา, วันที่สำเร็จการศึกษา
table_header    หัวตาราง (รายวิชา / หน่วยกิต / เกรด ...)
table_col1..N   แต่ละคอลัมน์ของตาราง (ชุดที่ 1 มี 6 คอลัมน์, ชุด G มี 8 คอลัมน์เพราะมีคอลัมน์ Type)
footer_left     วันที่ออกเอกสาร
footer_right    หมายเหตุ, ลายเซ็น, ตำแหน่ง
```

### วิธีที่ใช้หาส่วนต่าง ๆ

1. **Line-based morphology (หาตาราง):** ใช้ `MORPH_OPEN` ด้วย kernel ยาว `(1, ความสูงรูป/30)` หาเส้นแนวตั้ง และ `(ความกว้างรูป/30, 1)` หาเส้นแนวนอน
   - เส้นแนวตั้งที่ยาวเกินครึ่งตาราง = เส้นแบ่งคอลัมน์
   - เส้นแนวนอนที่ยาวเกือบเต็มตาราง (≥ 80%) ใต้ขอบบน = เส้นใต้หัวตาราง (เส้นใต้ชื่อภาคการศึกษาสั้นกว่า จึงไม่ถูกนับ)
   - kernel ตั้งตามสัดส่วนรูป ไม่ใช้ค่าตายตัวอย่าง `(1, 40)` เพราะตัวอักษรไทยที่ 300 dpi สูง ~40–50 px จะถูกนับเป็นเส้นด้วย
2. **Projection profile (แบ่งส่วนนอกตาราง):**
   - รวม pixel หมึกทีละแถว → ช่วงว่างที่สูง ≥ 1% ของหน้า แบ่งเป็น block
   - รวม pixel หมึกทีละคอลัมน์ใน block → ช่องว่างกว้าง ≥ 4% ของหน้าแถวกลางหน้า แบ่งเป็นซ้าย/ขวา
   - ยอมให้ 1 บรรทัดยาวล้ำเข้าไปในช่องว่างได้ (เช่น บรรทัด Program ยาว ๆ) และรวมสระบน/ล่าง วรรณยุกต์ เข้ากับบรรทัดหลักก่อนนับ

### `--layout-mode`: จับข้อความเข้าส่วนแบบไหน

| mode | ทำอะไร | เหมาะกับ |
|---|---|---|
| `assign` (B) | OCR ทั้งหน้าครั้งเดียว แล้วจัดกล่องข้อความเข้าส่วนตามจุดกึ่งกลางกล่อง | engine ที่ให้กล่องเล็ก: tesseract (ทีละคำ), paddle, easyocr, ensemble |
| `crop` (A) | ตัดรูปทีละส่วนแล้ว OCR แยกกัน (ข้ามส่วนที่ว่าง) | engine ที่ให้กล่องใหญ่คร่อมหลายส่วน: doctr, surya, trocr (ช้าลงเพราะ OCR หลายครั้ง) |
| `auto` (ค่าเริ่มต้น) | เลือกให้ตาม engine ตามตารางด้านบน | |

### ผลลัพธ์เมื่อใช้ `--layout`

- `_ocr.txt` เรียงข้อความตามส่วน มีหัวข้อ `[header]`, `[info_left]`, `[table_col1]`, ...
- `_ocr.json` ทุกบรรทัดมีช่อง `"region"` และแต่ละหน้ามี `"regions"` (ชื่อ + ตำแหน่งกรอบ)
- ใส่ `--save-debug-images` จะได้รูป `outputs/debug/page_001_layout.png` ที่ตีกรอบสีแต่ละส่วนไว้ตรวจว่าแบ่งถูกไหม

```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine paddle --layout --no-preprocess --save-debug-images
```

### ปรับแต่งได้ที่ `layout.py`

| อยากปรับ | ตรงไหน |
|---|---|
| ความยาว kernel หาเส้นตาราง | `h // 30`, `w // 30` ใน `detect_table()` |
| ความยาวเส้นที่นับเป็นเส้นคอลัมน์ / เส้นหัวตาราง | `0.5 * th`, `0.8 * tw` ใน `detect_table()` |
| ระยะว่างที่แบ่ง block | `min_gap = int(0.01 * h)` ใน `detect_blocks()` |
| ความกว้างช่องว่างที่แบ่งซ้าย/ขวา | `0.04 * w` และช่วงกลางหน้า `0.3 * w`–`0.7 * w` ใน `_best_gap()` |
| ชื่อส่วน | `"header"`, `"info"`, `"footer"` ใน `detect_regions()` |
| engine ไหนใช้แบบ crop | `CROP_ENGINES` ใน `pipeline.py` |

---

## ปรับแต่ง / แก้ไขได้ที่ไหน

| อยากทำอะไร | แก้ที่ไหน |
|---|---|
| เปลี่ยน engine, ภาษา, dpi, ปิด preprocess | ใช้ option ตอนรัน ไม่ต้องแก้โค้ด |
| เปลี่ยนค่าเริ่มต้น | `config.py` และ default ใน `cli.py` |
| ปรับการทำความสะอาดรูป | `preprocessing.py` |
| ปรับการแบ่ง layout | `layout.py` (ดูหัวข้อ Layout Analysis) |
| ปรับ / เพิ่ม augmentation | `augmentation.py` |
| ปรับ Tesseract (เช่น `psm`) | `engines/tesseract_engine.py` |
| เปลี่ยนสมาชิกของ ensemble | `ENSEMBLE_MEMBERS` ใน `engine_factory.py` |
| เปลี่ยนวิธีรวมผลของ ensemble | `merge()` ใน `engines/ensemble_engine.py` |
| ดึง field ทั่วไป (email, วันที่, เบอร์โทร) | `field_extraction.py` |
| ดึงข้อมูลใบเกรด (ชื่อ, วิชา, เกรด, ...) | `transcript_extraction.py` |
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
          "page": 1,
          "region": "info_left"
        }
      ],
      "image_path": "outputs/pages/71010001_page_001.jpg",
      "regions": [{"name": "header", "x0": 481, "y0": 30, "x1": 1996, "y1": 432}]
    }
  ]
}
```

`confidence` ของ `surya` และ `trocr` เป็น `null` เพราะ engine ไม่ได้ให้ค่านี้มา

`region` และ `regions` มีเฉพาะเมื่อใส่ `--layout` (ไม่งั้นเป็น `null`)

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
