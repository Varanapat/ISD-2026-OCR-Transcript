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

## คำถามที่พบบ่อย

### เปิดเว็บแล้วขึ้น `ERR_CONNECTION_REFUSED`

FastAPI ยังไม่ได้รัน ให้รันคำสั่ง Uvicorn ด้านบนและเปิด Terminal ค้างไว้

### ขึ้น `No module named fastapi` หรือ `No module named multipart`

ยังไม่ได้ activate venv หรือยังไม่ได้ติดตั้ง `transcript_app/requirements.txt`

### อัปโหลดแล้วขึ้น `ติดต่อ Ollama ไม่ได้`

เปิด Ollama แล้วรัน `ollama list` เพื่อตรวจว่ามี Typhoon และ Qwen

### OCR นานหรือค่าที่อ่านได้ผิด

โมเดลต้องประมวลผลภาพและสร้าง JSON จึงช้ากว่า OCR ปกติ ไฟล์สะอาดให้เลือก `none` และตรวจผลกับ Ground Truth
