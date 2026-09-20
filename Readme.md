# Thai-English OCR System

โปรเจกต์นี้เป็น OCR pipeline สำหรับเอกสารภาพเดี่ยวและหลายหน้า เช่น `.jpg`, `.png`, `.tif`, `.pdf` โดยรองรับเอกสารภาษาไทยและอังกฤษปนกัน

ในโปรเจกต์มี 2 pipeline ที่ใช้ข้อมูลใน `data/` ชุดเดียวกัน:

1. **OCR pipeline แบบดั้งเดิม** (`ocr_system.cli`) — อ่านข้อความด้วย OCR engine แล้วดึง field
   - PaddleOCR: เหมาะกับภาษาไทยและเอกสารทั่วไป
   - Tesseract OCR: ใช้ `tha+eng` ได้ดีเมื่อมีภาษาไทย/อังกฤษปนกัน
   - TrOCR: OCR แบบ Transformer เหมาะกับ printed English เป็นหลัก
   - Ensemble: ใช้ PaddleOCR + Tesseract แล้วรวมผลแบบง่าย
2. **VLM pipeline** (`ocr_system.vlm`, Lab 7A / Lab 8A) — ใช้ Typhoon-OCR + qwen3 ผ่าน Ollama บนเครื่องตัวเอง
   ดึง transcript นักศึกษาเป็น JSON และวัดผลกระทบของ noise ต่อความแม่นยำ
   ดูรายละเอียดที่หัวข้อ [VLM Pipeline (Lab 7A / Lab 8A)](#vlm-pipeline-lab-7a--lab-8a)

---

## Project Structure

```text
ocr_system/
├── README.md
├── requirements.txt
├── pyproject.toml
├── evaluate_levels.py         # วัดผล OCR บน dataset ที่ augment แล้ว (Lab 6)
├── data/                      # ข้อมูลที่ทุก pipeline ใช้ร่วมกัน
│   ├── input/                 # PDF/ภาพ ปริญญาตรี
│   ├── input_G/               # PDF/ภาพ บัณฑิตศึกษา
│   ├── ground_truth/          # เฉลย ปริญญาตรี   (Json_<รหัส>_th.json / Json_<รหัส>_en.json)
│   └── ground_truth_G/        # เฉลย บัณฑิตศึกษา
├── dataset/                   # dataset ที่ augment แล้ว (สร้างจากคำสั่ง dataset)
├── outputs/                   # ผลลัพธ์ OCR และ evaluation ของ pipeline ดั้งเดิม
│   └── lab7a/                 # ผลลัพธ์ของ Lab 7A (VLM)
├── work/                      # ผลลัพธ์ของ Lab 8A แยกตามเอกสาร (ดูหัวข้อ VLM)
└── src/
    └── ocr_system/
        ├── cli.py                  # command line interface
        ├── config.py               # config หลักของระบบ
        ├── document_loader.py      # โหลดภาพ / แปลง PDF เป็นภาพ
        ├── preprocessing.py        # resize, denoise, contrast, deskew, threshold
        ├── pipeline.py             # OCR pipeline หลัก
        ├── evaluation.py           # CER, WER, exact match
        ├── field_extraction.py     # ดึง field เช่น email, date, id, phone
        ├── transcript_extraction.py# จัดข้อความ OCR เป็นโครงสร้าง transcript
        ├── dataset_builder.py      # สร้าง augmented dataset
        ├── augmentation.py         # สร้าง noise ให้ภาพ
        ├── schemas.py              # dataclass ของผลลัพธ์
        ├── engine_factory.py       # เลือก OCR engine
        ├── engines/
        │   ├── base.py
        │   ├── paddle_engine.py
        │   ├── tesseract_engine.py
        │   ├── trocr_engine.py
        │   └── ensemble_engine.py
        ├── utils/
        │   └── io.py
        └── vlm/                    # VLM pipeline (Lab 7A / Lab 8A)
            ├── lab7a_transcript.py # Typhoon-OCR -> Markdown -> qwen3 -> JSON
            ├── lab8a_denoise.py    # สร้าง noise, ทำความสะอาดภาพ, วัดผล, ให้ LLM แก้
            └── lab7_metrics.py     # CER / WER / การจับคู่ภาค-วิชา
```

### กติกาการวางข้อมูลใน `data/`

- ปริญญาตรีอยู่ `data/input/` คู่กับ `data/ground_truth/` ส่วนบัณฑิตศึกษาอยู่ `data/input_G/` คู่กับ `data/ground_truth_G/`
- ชื่อไฟล์เฉลยต้องเป็น `Json_<รหัสเอกสาร>_th.json` (เอกสารไทย) หรือ `Json_<รหัสเอกสาร>_en.json` (เอกสารอังกฤษ)
  เช่น `data/input/71010001.pdf` คู่กับ `data/ground_truth/Json_71010001_th.json`
- ชื่อโฟลเดอร์บัณฑิตต้องลงท้าย `_G` **ตัวพิมพ์ใหญ่** เพราะโค้ดใช้ชื่อโฟลเดอร์นี้แยกว่าเอกสารเป็นปริญญาตรีหรือบัณฑิต

---

## ใช้งานผ่าน VS Code

แนะนำให้ใช้ **VS Code** เพราะเปิดดูโครงสร้างไฟล์ แก้โค้ด และรันคำสั่งใน Terminal ได้ในที่เดียว

---

## วิธีเปิดโปรเจกต์ใน VS Code
1. แตกไฟล์ `ocr_system.zip`
2. จะได้โฟลเดอร์ชื่อ `ocr_system`
3. เปิด VS Code
4. ไปที่เมนู
```text
File > Open Folder
```

5. เลือกโฟลเดอร์ `ocr_system`
6. เปิด Terminal ใน VS Code
```text
Terminal > New Terminal
```
หลังจากนี้ให้พิมพ์คำสั่งต่าง ๆ ใน Terminal ของ VS Code ได้เลย

---

## Installation
แนะนำใช้ Python 3.10 ขึ้นไป
เช็กเวอร์ชัน Python ก่อน:

```bash
python --version
```
หรือบางเครื่องอาจต้องใช้:
```bash
py --version
```
ถ้าเวอร์ชันเป็น Python 3.10, 3.11 หรือ 3.12 สามารถใช้ได้

---

## สร้าง Virtual Environment
Virtual Environment คือพื้นที่แยกสำหรับติดตั้ง package ของโปรเจกต์นี้โดยเฉพาะ เพื่อไม่ให้ชนกับโปรเจกต์อื่น
ให้เข้าไปในโฟลเดอร์โปรเจกต์ก่อน:
```bash
cd ocr_system
```
จากนั้นสร้าง environment:
```bash
python -m venv .venv
```

ถ้าใช้ Windows แล้วคำสั่ง `python` ไม่ได้ ให้ลองใช้:
```bash
py -m venv .venv
```

---

## เปิดใช้งาน Virtual Environment

### Windows CMD
```bash
.venv\Scripts\activate
```

### Windows PowerShell
```bash
.venv\Scripts\Activate.ps1
```

ถ้า PowerShell ขึ้น error เรื่อง policy ให้รัน:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

แล้วลอง activate ใหม่อีกครั้ง

### macOS / Linux
```bash
source .venv/bin/activate
```
ถ้าสำเร็จ จะเห็นชื่อ environment ขึ้นต้นบรรทัดประมาณนี้:
```text
(.venv) C:\...\ocr_system>
```

---

## ติดตั้ง Python Packages
หลังจาก activate `.venv` แล้ว ให้ติดตั้ง package ทั้งหมด:
```bash
pip install -r requirements.txt
```

จากนั้นติดตั้งโปรเจกต์แบบ editable:
```bash
pip install -e .
```

คำสั่งนี้ทำให้สามารถเรียกใช้งานโปรเจกต์ด้วยรูปแบบนี้ได้:
```bash
python -m ocr_system.cli
```

ถ้าไม่ได้ติดตั้งแบบ editable ต้องตั้ง `PYTHONPATH=src` ก่อนรันทุกครั้ง (ทำครั้งเดียวต่อ Terminal):

macOS / Linux
```bash
export PYTHONPATH=src
```
Windows PowerShell
```bash
$env:PYTHONPATH = "src"
```

ถ้าจะใช้ VLM pipeline ต้องติดตั้งเพิ่ม ดูหัวข้อ [ติดตั้งสำหรับ VLM pipeline](#ติดตั้งสำหรับ-vlm-pipeline)

---

## Install Tesseract Engine
ในโปรเจกต์นี้มี OCR หลายตัว เช่น PaddleOCR, Tesseract และ TrOCR
แต่สำหรับ Tesseract ต้องติดตั้งโปรแกรม Tesseract OCR แยกต่างหาก เพราะ `pytesseract` เป็นแค่ Python package ที่ใช้เรียกโปรแกรม Tesseract เท่านั้น

---

## ติดตั้ง Tesseract บน Windows
ให้ติดตั้ง Tesseract OCR จาก UB Mannheim build
ระหว่างติดตั้ง ให้เลือกภาษา:
```text
English
Thai
```

หลังติดตั้งเสร็จ ให้เปิด CMD หรือ VS Code Terminal ใหม่ แล้วตรวจสอบ:
```bash
tesseract --version
```

จากนั้นตรวจสอบภาษาที่ติดตั้ง:
```bash
tesseract --list-langs
```
ควรเห็นอย่างน้อย:
```text
eng
tha
```
ถ้าไม่เห็น `tha` แปลว่ายังไม่ได้ติดตั้งภาษาไทย

---

## ติดตั้ง Tesseract บน Ubuntu / Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-tha poppler-utils
```
---

## ติดตั้ง Tesseract บน macOS
```bash
brew install tesseract poppler
brew install tesseract-lang
```
หมายเหตุ: `poppler` จำเป็นสำหรับแปลง PDF เป็นภาพผ่าน `pdf2image`

---

## เตรียมไฟล์สำหรับทดสอบ OCR
นำไฟล์เอกสารไปวางในโฟลเดอร์ `data/` ตาม[กติกาการวางข้อมูล](#กติกาการวางข้อมูลใน-data) เช่น:
```text
data/input/71010001.pdf
data/input/sample.jpg
data/input_G/73036003.pdf
```

รองรับทั้ง:
```text
PDF หลายหน้า
JPG
PNG
TIFF
BMP
```

---

## Usage
### 1. OCR ด้วย Ensemble
Ensemble คือการใช้หลาย OCR engine ช่วยกัน แล้วเลือกผลลัพธ์ที่เหมาะสมที่สุด
เหมาะสำหรับเอกสารที่มีทั้งภาษาไทยและอังกฤษปนกัน

```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine ensemble
```

หลังรันเสร็จ ผลลัพธ์จะอยู่ในโฟลเดอร์:
```text
outputs/
```

จะได้ไฟล์ประมาณนี้:
```text
outputs/71010001_ocr.json
outputs/71010001_ocr.txt
outputs/71010001_fields.json
outputs/pages/
```

ความหมายของไฟล์:
```text
<ชื่อ>_ocr.json     ผล OCR แบบละเอียด เช่น text, confidence, page
<ชื่อ>_ocr.txt      ข้อความ OCR รวมทั้งหมด อ่านง่าย
<ชื่อ>_fields.json  field ที่ระบบพยายาม extract เช่น วันที่ ชื่อ รหัส
outputs/pages/      ภาพแต่ละหน้าที่แปลงจาก PDF
```

---

### 2. OCR ด้วย PaddleOCR
เหมาะกับเอกสารทั่วไป โดยเฉพาะภาษาไทยและอังกฤษปนกัน
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine paddle --paddle-lang th
```
ถ้าเอกสารเป็นอังกฤษล้วน อาจลองใช้:
```bash
python -m ocr_system.cli ocr data/input/72100002.pdf --engine paddle --paddle-lang en
```

---

### 3. OCR ด้วย Tesseract ไทย + อังกฤษ
เหมาะกับเอกสาร scan ที่ตัวหนังสือชัด หรือเอกสารราชการ/ฟอร์มที่ layout ไม่ซับซ้อนมาก
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine tesseract --languages tha+eng
```

ถ้าเป็นอังกฤษอย่างเดียว:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine tesseract --languages eng
```

ถ้าเป็นไทยอย่างเดียว:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine tesseract --languages tha
```

---

### 4. OCR ด้วย TrOCR
TrOCR เป็นโมเดล OCR จาก Transformer
ในโปรเจกต์นี้ใช้เป็น fallback สำหรับข้อความสั้น ๆ หรือภาพที่ crop เป็นบรรทัดแล้ว
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine trocr --device cpu
```
ถ้ามี GPU และติดตั้ง PyTorch แบบ CUDA แล้ว สามารถใช้:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine trocr --device cuda
```
หมายเหตุ: TrOCR ในโปรเจกต์นี้ยังไม่เหมาะกับเอกสารยาวทั้งหน้า แนะนำใช้ PaddleOCR หรือ Tesseract เป็นหลัก

---

## Evaluation
Evaluation คือการวัดว่า OCR อ่านถูกแค่ไหน โดยเทียบกับข้อความจริง หรือ Ground Truth

ขั้นที่ 1 — รัน OCR:
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine ensemble
```

ขั้นที่ 2 — จัดข้อความ OCR เป็นโครงสร้าง transcript:
```bash
python -m ocr_system.cli transcript outputs/71010001_ocr.json
```

ขั้นที่ 3 — เทียบกับเฉลย:
```bash
python -m ocr_system.cli evaluate data/ground_truth/Json_71010001_th.json outputs/71010001_ocr_transcript.json
```

ถ้าเป็นเอกสารบัณฑิตศึกษา ให้ใช้เฉลยใน `data/ground_truth_G/` เช่น `data/ground_truth_G/Json_73036003_th.json`

Metric ที่ได้:

```text
cer           Character Error Rate ยิ่งต่ำยิ่งดี
wer           Word Error Rate ยิ่งต่ำยิ่งดี
exact_match   ข้อความตรงทั้งหมดหรือไม่
```

ตัวอย่างการอ่านผล:
```text
CER = 0.05 หมายถึงผิดประมาณ 5% ระดับตัวอักษร
WER = 0.12 หมายถึงผิดประมาณ 12% ระดับคำ
exact_match = false หมายถึงยังไม่ตรง 100%
```

---

## สร้าง Augmented Dataset
สร้างภาพหลายเวอร์ชัน (noise ต่าง ๆ) จาก PDF พร้อมเฉลย เพื่อใช้วัดผลด้วย `evaluate_levels.py`
```bash
python -m ocr_system.cli dataset data/input dataset/augmented --n 8
```
ค่าเริ่มต้นของ `--ground-truth-dir` คือ `data/ground_truth` (ปริญญาตรี) ถ้าใช้ชุดบัณฑิตศึกษาต้องระบุเอง:
```bash
python -m ocr_system.cli dataset data/input_G dataset/augmented_G --ground-truth-dir data/ground_truth_G --n 8
```

---

## คำสั่งที่ใช้บ่อย
OCR ไฟล์ PDF ด้วยระบบรวม:
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine ensemble
```

OCR รูปภาพด้วย PaddleOCR:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine paddle --paddle-lang th
```

OCR รูปภาพด้วย Tesseract:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine tesseract --languages tha+eng
```

Evaluate ผล OCR:
```bash
python -m ocr_system.cli evaluate data/ground_truth/Json_71010001_th.json outputs/71010001_ocr_transcript.json
```

---

## Recommended Engine

สำหรับเอกสารไทย+อังกฤษปนกัน แนะนำเริ่มจาก:
```bash
python -m ocr_system.cli ocr data/input/71010001.pdf --engine ensemble --languages tha+eng --paddle-lang th --save-debug-images
```

ถ้าเอกสารเป็นอังกฤษเกือบทั้งหมด:
```bash
python -m ocr_system.cli ocr data/input/72100002.pdf --engine paddle --paddle-lang en
```

ถ้า Tesseract อ่านไทยเพี้ยน ให้ลอง OCR แบบไม่ preprocess:
```bash
python -m ocr_system.cli ocr data/input/sample.jpg --engine tesseract --no-preprocess
```

---

## Output JSON Format
```json
{
  "source_path": "data/input/71010001.pdf",
  "engine": "ensemble",
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
          "engine": "paddle",
          "page": 1
        }
      ],
      "image_path": "outputs/pages/71010001_page_001.jpg"
    }
  ]
}
```

---

# VLM Pipeline (Lab 7A / Lab 8A)

pipeline นี้ดึงข้อมูล transcript นักศึกษาออกมาเป็น JSON ตาม schema ของ ground truth โดยใช้ Vision-Language Model
ทุกขั้นรันบนเครื่องตัวเองผ่าน Ollama — **ห้ามส่ง transcript ขึ้น cloud / API ภายนอก** เพราะเป็นข้อมูลส่วนบุคคลตาม PDPA

```text
PDF/ภาพ ─▶ Typhoon-OCR ─▶ Markdown/HTML ─▶ normalize_typhoon_table() ─▶ qwen3 ─▶ JSON ─▶ แปลงปี พ.ศ./ค.ศ.
           (อ่านภาพ)                      (จัดตาราง ไม่ทิ้งแถว)          (จัดรูป)          (ด้วยโค้ด)
```

| ไฟล์ | หน้าที่ |
|---|---|
| `vlm/lab7a_transcript.py` | pipeline หลัก: อ่านเอกสาร 1 ฉบับแล้วคืน JSON + เทียบเฉลย (Lab 7A) |
| `vlm/lab8a_denoise.py` | สร้าง noise 5 ระดับ, ทำความสะอาดภาพ, รัน pipeline, วัดผล, ให้ LLM แก้แล้ว triage (Lab 8A) |
| `vlm/lab7_metrics.py` | CER / WER / exact match และการจับคู่ภาค-วิชาด้วยกุญแจ (ไม่จับคู่ตาม index) |

---

## ติดตั้งสำหรับ VLM pipeline

### 1. Python packages เพิ่มเติม
หลังติดตั้ง `requirements.txt` แล้ว ให้ติดตั้งเพิ่ม:
```bash
pip install requests pymupdf augraphy matplotlib pythainlp
```
- `pymupdf` แปลง PDF เป็นภาพ, `augraphy` สร้าง noise, `matplotlib` วาดกราฟรายงาน
- `pythainlp` ไม่บังคับ ใช้ตัดคำไทยตอนคำนวณ WER

### 2. Ollama และโมเดล
ติดตั้ง Ollama จาก https://ollama.com แล้วดาวน์โหลดโมเดล 2 ตัว:
```bash
ollama pull scb10x/typhoon-ocr1.5-3b
```
```bash
ollama pull qwen3:4b
```

ต้องเปิด Ollama ไว้ก่อนรันทุกครั้ง (เปิดแอป Ollama หรือรันคำสั่งนี้ใน Terminal แยก):
```bash
ollama serve
```

ตรวจความพร้อมของเครื่อง:
```bash
python -m ocr_system.vlm.lab8a_denoise check
```

### ⚠️ ล็อกเวอร์ชัน Ollama ระหว่างทดลอง
แอป Ollama อัปเดตตัวเองอัตโนมัติ และ **เวอร์ชันของ Ollama เปลี่ยนผลลัพธ์ได้** แม้โค้ด โมเดล และ PDF จะเหมือนเดิมทุกไบต์
(เคยเจอจริง: 71010001 ได้ accuracy 0.925 ก่อนอัปเดต และ 0.688 หลังอัปเดตเป็น 0.33.3)

- ปิด auto-update ในหน้า Settings ของแอป Ollama ก่อนเริ่มทดลองชุดใหญ่
- ตรวจเวอร์ชันก่อนรันทุกครั้ง: `curl -s http://127.0.0.1:11434/api/version`
- ผลทุกไฟล์มีฟิลด์ `ollama_version` กำกับอยู่แล้ว ใช้ตรวจว่าผลที่นำมาเทียบกันมาจากเวอร์ชันเดียวกัน

---

## Lab 7A — ดึง transcript เป็น JSON
รันจากรากโปรเจกต์ `ocr_system/` ผลลัพธ์อยู่ที่ `outputs/lab7a/`

รัน VLM pipeline กับเอกสาร 1 ฉบับแล้วเทียบเฉลย:
```bash
python -m ocr_system.vlm.lab7a_transcript -i data/input/71010001.pdf -g data/ground_truth/Json_71010001_th.json -p vlm
```

เทียบกับ baseline (Tesseract + กฎ) ด้วย:
```bash
python -m ocr_system.vlm.lab7a_transcript -i data/input/71010001.pdf -g data/ground_truth/Json_71010001_th.json -p all
```

ประเมินผลจาก JSON ที่รันไว้แล้ว (ไม่เรียกโมเดลซ้ำ):
```bash
python -m ocr_system.vlm.lab7a_transcript --eval-only outputs/lab7a/pred_vlm.json -g data/ground_truth/Json_71010001_th.json
```

ไฟล์ผลลัพธ์:
```text
outputs/lab7a/pred_vlm.json          JSON ที่ได้
outputs/lab7a/intermediate_vlm.md    Markdown หลัง normalize (ใช้ debug ว่าผิดที่ขั้น OCR หรือขั้น LLM)
outputs/lab7a/evaluation.json        ผลเทียบเฉลยแยกตามกลุ่มฟิลด์
outputs/lab7a/comparison.csv         ตารางเดียวกัน เปิดใน Excel ได้
```

---

## Lab 8A — ผลกระทบของ noise
ทุกคำสั่งใช้ `--doc <รหัสเอกสาร>` ได้ ระบบจะหา PDF ใน `data/input*/`, หาเฉลยใน `data/ground_truth*/`
และเก็บผลใน `work/<รหัสเอกสาร>/` ให้เอง (ถ้าระบุ `-i` / `-o` / `-g` เองจะใช้ค่าที่ระบุแทน)

### ระดับ noise และวิธีทำความสะอาด
| ระดับ | ลักษณะ |
|---|---|
| `L0_clean` | ต้นฉบับ ไม่มี noise |
| `L1_light` | เอียง 0.3° + noise เบา |
| `L2_watermark` | เอียง 1° + ตราปั๊มวงกลม (35%) |
| `L3_tilted_copy` | ย่อ 95% + เอียง 2.5° + ลายน้ำ COPY (45%) |
| `L4_rescan` | ย่อ 85% + เอียง 4° + ตราปั๊ม + COPY + เข้มขึ้น (จำลองสแกนซ้ำ) |

วิธีทำความสะอาด (`-m`): `none` ไม่ทำอะไร · `light` ลบลายน้ำ + แก้เอียง + ลด noise (คงระดับสีเทา) · `heavy` light + แปลงเป็นขาวดำ

### ขั้นตอน
ตรวจตรรกะการวัดผล (ไม่ต้องใช้ Ollama):
```bash
python -m ocr_system.vlm.lab8a_denoise selftest
```

1) สร้างภาพ noise ทั้ง 5 ระดับ:
```bash
python -m ocr_system.vlm.lab8a_denoise noise --doc 71010001
```

2) ทดสอบภาพเดียว (เร็ว ใช้ตอน debug) — เลือกระดับด้วย `--level` (ค่าเริ่มต้น `L0_clean`):
```bash
python -m ocr_system.vlm.lab8a_denoise denoise --doc 71010001 -m none
```
```bash
python -m ocr_system.vlm.lab8a_denoise denoise --doc 71010001 -m light --level L3_tilted_copy
```

3) รันทุกระดับ noise × ทุกวิธีทำความสะอาด (15 ชุด ใช้เวลาราว 2 นาทีต่อชุด):
```bash
python -m ocr_system.vlm.lab8a_denoise sweep --doc 71010001
```
รันบางส่วนได้ด้วย `--levels L0_clean,L2_watermark` และ `--methods none,light` (ผลรอบก่อนจะไม่ถูกลบ)

4) ให้ LLM แก้ข้อความหลัง OCR แล้วจำแนกว่า "แก้ถูก" หรือ "ทำพัง":
```bash
python -m ocr_system.vlm.lab8a_denoise fix --doc 71010001
```

5) สรุปผลของเอกสารเดียว หรือรวมทุกเอกสาร:
```bash
python -m ocr_system.vlm.lab8a_denoise report --doc 71010001
```
```bash
python -m ocr_system.vlm.lab8a_denoise report --all
```

### ผลลัพธ์ใน `work/`
```text
work/<รหัสเอกสาร>/
├── noisy/<ระดับ>/page_01.png           ภาพ noise แต่ละระดับ + manifest.json
├── denoise_out/                        ผลของคำสั่ง denoise
│   ├── page_01.md                      Markdown หลัง normalize
│   ├── page_01__pred.json              JSON ที่ได้
│   └── page_01__<วิธี>__metrics.json    accuracy / CER / hallucinated / ollama_version
├── out/<ระดับ>__<วิธี>/extracted.json   ผลของ sweep + out/sweep.json
├── fixed/<ชุด>__fix_<วิธี>/             ผลของ fix + fixed/fix_report.json
└── report/                             sweep.csv, triage.csv, curve.png, compare.png
work/summary/                           ผลรวมทุกเอกสารจาก report --all
                                        (sweep_all.csv, triage_all.csv, denoise_all.csv)
```

### การวัดผล
- **accuracy** = สัดส่วนฟิลด์ที่ตรงเฉลยหลัง normalize (ตัดช่องว่างทั้งหมด, ตัวพิมพ์เล็ก, เลขไทยเป็นอารบิก เพราะเฉลยเก็บแบบไม่มีช่องว่าง)
- จับคู่ภาคเรียนด้วย `ปี/ภาค` และจับคู่วิชาด้วย `รหัสวิชา` — อ่านตกแถวเดียวจะไม่ทำให้แถวหลังจากนั้นผิดทั้งหมด
- **hallucinated** = ฟิลด์ที่เฉลยว่าง แต่โมเดลใส่ค่ามา รวมถึงวิชา/ภาคที่ไม่มีในเฉลย
- **triage** (คำสั่ง fix) แยก แก้ถูก / ทำพัง / งดตอบ และนับ "ทำพังในฟิลด์สำคัญ" (`grade_earn`, `credit`, `subject_id`, `student_id`) แยกไว้
  วิธีที่ทำพังในฟิลด์สำคัญแม้ช่องเดียว ไม่ควรนำไปใช้กับ transcript จริง

### ผลปัจจุบัน (L0_clean, `-m none`, Ollama 0.33.3)
| เอกสาร | ประเภท | accuracy |
|---|---|---|
| 71010001 | ปริญญาตรี / ไทย | 0.828 |
| 72100002 | ปริญญาตรี / อังกฤษ | 0.877 |
| 73036003 | บัณฑิตศึกษา / ไทย | 0.780 |
| 74106005 | บัณฑิตศึกษา / อังกฤษ | 0.813 |

---

## การตั้งค่าผ่าน environment variable
| ตัวแปร | ค่าเริ่มต้น | ใช้ใน |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | lab7a |
| `LAB8_OLLAMA_URL` | `http://127.0.0.1:11434` | lab8a |
| `LAB7_MODEL_OCR` / `LAB8_MODEL_OCR` | `scb10x/typhoon-ocr1.5-3b` | โมเดลอ่านภาพ |
| `LAB7_MODEL_TEXT` / `LAB8_MODEL_TEXT` | `qwen3:4b` | โมเดลจัด JSON |
| `LAB7_DPI` | `150` | ความละเอียดตอนแปลง PDF ใน lab7a (lab8a สร้างภาพที่ 300 DPI) |
| `LAB7_SKIP_BASELINE` | ว่าง | ตั้งเป็น `1` เพื่อข้าม baseline ตอนใช้ `-p all` |

---

## Known limitations (VLM)
- **หน่วยกิตคอลัมน์ขวาของตารางสองคอลัมน์** — Typhoon ไม่ได้อ่านมาเลย (เช่น ภาค 2 ของ 71010001) แก้ในโค้ดไม่ได้
- **ประเภทวิชาเกินมา** — บางแถว OCR อ่านเป็น `Cr Nc` ทั้งที่เอกสารพิมพ์ `Cr` อย่างเดียว
- **แถว "รักษาสภาพ" (Maintain)** ของบัณฑิตศึกษา — เฉลยเก็บเป็น `pass_reason` แต่ pipeline อ่านเป็นรายวิชา
- **`grad_reason`** — ได้ `N/A` ในกรณีที่เฉลยเป็น null และรูปแบบข้อความยังไม่ตรงเฉลยทุกกรณี
- **`major`** — โมเดลมักเติมค่าให้ ทั้งที่เฉลยเป็น null
- **GPS/GPA ของเอกสารภาษาอังกฤษ** — บรรทัดสรุปภาคภาษาอังกฤษยังถูกส่งต่อแบบดิบ (`[ดิบ]`) ให้ LLM ตีความเอง บางภาคจึงได้ null
- **ลายน้ำทับเนื้อหาโดยตรง (L2_watermark)** — `remove_watermark_tint()` ยังกู้คืนตัวอักษรใต้ลายน้ำได้ไม่สมบูรณ์
- **qwen3:4b อ่อนไหวต่อถ้อยคำใน prompt** — แก้ prompt บรรทัดที่ไม่เกี่ยวข้องก็ทำให้ฟิลด์อื่นเปลี่ยนได้ ±1–3% ควรรันเทียบทั้ง 4 เอกสารทุกครั้งที่แก้ prompt
