// app.js — หน้า Transcript OCR: ส่งไฟล์ไปที่ API แล้วแสดงผลให้ครบ 4 สถานะ
//
//   idle (รอไฟล์) -> loading (กำลังอ่าน) -> success (แสดงผล)
//                                        \-> error   (แจ้งข้อผิดพลาด)

const $ = (id) => document.getElementById(id);
const pretty = (value) => JSON.stringify(value, null, 2);

let timer = null;        // ตัวนับวินาทีที่ใช้ตอน loading
let lastResult = null;   // ผลล่าสุดที่อ่านสำเร็จ ไว้ให้ปุ่มดาวน์โหลดใช้

// ---------- 1) เปลี่ยนสถานะ ----------
// เป็นจุดเดียวที่เปลี่ยนสถานะของหน้าเว็บ ทุกที่ที่อยากเปลี่ยนสถานะให้เรียกฟังก์ชันนี้
//   - data-state บน <body> ให้ style.css ใช้เปลี่ยนสีกล่องสถานะ
//   - ปุ่มอัปโหลดถูกปิดระหว่าง loading กันกดซ้ำ (OCR ใช้เวลาหลายนาที)
function setState(state, message) {
    if (state !== "loading") stopTimer();
    document.body.dataset.state = state;          // "idle" | "loading" | "success" | "error"
    $("status").textContent = message;
    document.querySelector("#upload-form button[type=submit]").disabled = (state === "loading");
}

// ---------- 2) loading: แสดงเวลาที่ผ่านไป ให้ผู้ใช้รู้ว่าระบบไม่ได้ค้าง ----------
function startLoading() {
    const started = Date.now();
    const tick = () => {
        const seconds = Math.floor((Date.now() - started) / 1000);
        const note = $("method").value === "vlm" ? " (อาจใช้เวลาหลายนาที)" : "";
        setState("loading", `กำลังอ่านเอกสาร... ผ่านไป ${seconds} วินาที${note}`);
    };
    tick();
    timer = setInterval(tick, 1000);
}

function stopTimer() {
    if (timer) {
        clearInterval(timer);
        timer = null;
    }
}

// ---------- 3) ล้างผลของครั้งก่อน ----------
// ไม่ล้าง จะเห็นตารางของไฟล์เก่าค้างอยู่ข้างข้อความ error ของไฟล์ใหม่
function clearResults() {
    $("header").textContent = "-";
    $("footer").textContent = "-";
    $("warnings").textContent = "-";
    $("courses").textContent = "-";
    $("markdown").textContent = "";
    $("markdown-section").hidden = true;
    lastResult = null;                 // ผลเก่าที่ดาวน์โหลดได้ต้องหายไปพร้อมกัน
    $("download-btn").hidden = true;
}

// ---------- 4) แปลรหัสผิดพลาดจาก server เป็นข้อความที่ผู้ใช้เข้าใจ ----------
// รหัสเหล่านี้ตรงกับที่ main.py ส่งมา: 415 ชนิดไฟล์, 413 ไฟล์ใหญ่, 422 อ่านเอกสารไม่สำเร็จ
function errorMessage(status, detail) {
    // detail เป็น string เมื่อ main.py ส่งมาเอง แต่ FastAPI ตรวจ input ไม่ผ่านจะส่ง list มา
    const text = typeof detail === "string" ? detail : "";
    if (status === 413) return text || "ไฟล์ใหญ่เกินที่กำหนด";
    if (status === 415) return text || "ชนิดไฟล์ไม่รองรับ (รองรับ PDF, PNG, JPG, TIFF)";
    if (status === 422) return text ? `อ่านเอกสารไม่สำเร็จ: ${text}` : "ข้อมูลที่ส่งไปไม่ถูกต้อง";
    return text || `เซิร์ฟเวอร์ตอบผิดพลาด (รหัส ${status})`;
}

// ---------- 5) แสดงผลลัพธ์ (success) ----------
function showResults(data) {
    $("header").textContent = pretty(data.header_detail);
    $("footer").textContent = pretty(data.footer_detail);
    $("warnings").textContent = pretty({
        warnings: data.warnings,
        postprocessing_changes: data.postprocessing_changes,
    });

    const container = $("courses");
    container.textContent = "";
    const semesters = (data.transcript_detail && data.transcript_detail.semesters) || [];
    if (semesters.length === 0) {
        container.textContent = "ไม่พบรายวิชาในเอกสาร";
    }
    for (const semester of semesters) {
        const title = document.createElement("h3");
        title.textContent = `ปี ${semester.year} ภาค ${semester.sem_num}`;
        container.appendChild(title);

        const table = document.createElement("table");
        const headRow = table.insertRow();
        for (const label of ["รหัส", "ชื่อวิชา", "หน่วยกิต", "เกรด"]) {
            const th = document.createElement("th");
            th.textContent = label;           // textContent ไม่ตีความ HTML จึงกัน script injection
            headRow.appendChild(th);
        }
        for (const subject of (semester.subject || [])) {
            const row = table.insertRow();
            for (const key of ["subject_id", "subject_name", "credit", "grade_earn"]) {
                row.insertCell().textContent = subject[key] ?? "";
            }
        }
        container.appendChild(table);
    }

    if (data.markdown) {
        $("markdown-section").hidden = false;
        $("markdown").textContent = data.markdown;
    }
}

// ---------- 6) ดาวน์โหลดผลเป็นไฟล์ JSON (สำหรับส่งตรวจ / ประเมินกับเฉลย) ----------
// ไฟล์มีแค่ 3 ส่วนเหมือนไฟล์เฉลย (header_detail, transcript_detail, footer_detail)
// ไม่ใส่ warnings / filename / markdown เพื่อให้ตัวประเมินอ่านได้ตรง ๆ
function downloadName(data) {
    const studentId = String((data.header_detail || {}).student_id || "");
    const stem = (data.filename || "").replace(/\.[^.]+$/, "");
    const safeStem = stem.replace(/[^\w-]+/g, "_").replace(/^_+|_+$/g, "");
    const base = /^\d+$/.test(studentId) ? studentId : (safeStem || "result");
    return `${base}_transcript.json`;
}

$("download-btn").addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult.data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = lastResult.name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// ---------- 7) ตอนกดปุ่มอัปโหลด ----------
$("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = $("file").files[0];
    if (!file) {
        setState("error", "กรุณาเลือกไฟล์ก่อน");
        return;
    }

    clearResults();
    startLoading();                                   // -> สถานะ loading

    const body = new FormData();
    body.append("file", file);
    body.append("method", $("method").value);
    body.append("preprocessing", $("preprocessing").value);
    body.append("include_markdown", $("include-markdown").checked);

    try {
        let response;
        try {
            response = await fetch("/api/transcript/extract", { method: "POST", body });
        } catch (networkError) {
            // fetch โยน error เมื่อติดต่อ server ไม่ได้เลย (server ปิด / เน็ตหลุด)
            throw new Error("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ตรวจว่า uvicorn และ Ollama ยังทำงานอยู่");
        }

        // ถ้า server ตอบกลับมาเป็นอย่างอื่นที่ไม่ใช่ JSON (เช่น หน้า error จาก proxy)
        // response.json() จะพัง จึงดักไว้แล้วใช้รหัสสถานะบอกแทน
        let data = null;
        try {
            data = await response.json();
        } catch (parseError) {
            // ปล่อยให้ data เป็น null
        }

        if (!response.ok || !data) {
            throw new Error(errorMessage(response.status, data && data.detail));
        }

        showResults(data);
        lastResult = {
            name: downloadName(data),
            data: {
                header_detail: data.header_detail,
                transcript_detail: data.transcript_detail,
                footer_detail: data.footer_detail,
            },
        };
        $("download-btn").hidden = false;
        const warningCount = (data.warnings || []).length;
        setState("success",                                         // -> สถานะ success
            `${data.filename}: ${data.pages} หน้า, ${data.processing_seconds} วินาที` +
            (warningCount ? ` · มีคำเตือน ${warningCount} รายการ` : ""));
    } catch (error) {
        clearResults();
        setState("error", `ผิดพลาด: ${error.message}`);             // -> สถานะ error
    }
});

setState("idle", "รอไฟล์...");                                      // -> สถานะ idle ตอนเปิดหน้า

// ---------- 8) คำอธิบายตัวเลือกตามวิธีอ่านที่เลือก ----------
// ช่อง preprocessing ใช้ร่วมกันสองโหมดแต่ความหมายต่างกัน จึงบอกผู้ใช้ตรงนี้
// ค่าที่ส่งไป API ยังเป็น none / light / heavy เหมือนเดิม (contract ไม่เปลี่ยน) แต่ข้อความและลำดับที่ผู้ใช้เห็นต่างกันตามโหมด
//   โหมด OCR: none = ไม่ preprocess · heavy = preprocess ทั้ง pipeline · light = preprocess แค่ deskew
const PREPROCESS_OPTIONS = {
    vlm: { default: "none", options: [
        ["none", "none (แนะนำสำหรับไฟล์สะอาด)"],
        ["light", "light"],
        ["heavy", "heavy"],
    ] },
    ocr: { default: "light", options: [          // ค่าเริ่มต้น = deskew เหมือนที่กลุ่มใช้ (ทนภาพเอียงและไม่ทำให้ไฟล์สะอาดแย่ลง)
        ["none", "ไม่ preprocess"],
        ["heavy", "preprocess ทั้ง pipeline"],
        ["light", "preprocess แค่ deskew"],
    ] },
};
const METHOD_HINTS = {
    vlm: "VLM: ช้า (1–3 นาทีต่อไฟล์) ต้องมี Ollama | none=ภาพดิบ, light/heavy=ทำความสะอาดภาพ (ไฟล์สะอาดควรใช้ none)",
    ocr: "OCR ปกติ (tesseract): เร็ว (ไม่กี่วินาที) | deskew=ปรับภาพเอียง ช่วยมากเมื่อสแกนหรือถ่ายเอียง ส่วน preprocess ทั้ง pipeline อาจทำให้ผลแย่ลงในไฟล์สะอาด | ไม่มี postprocessing ของ Lab 8A",
};
function applyMethod() {
    const method = $("method").value;
    const config = PREPROCESS_OPTIONS[method];
    const select = $("preprocessing");
    select.textContent = "";
    for (const [value, label] of config.options) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        select.appendChild(option);
    }
    select.value = config.default;
    $("method-hint").textContent = METHOD_HINTS[method] || "";
}
$("method").addEventListener("change", applyMethod);
applyMethod();

const clearBtn = document.getElementById("clear-file-btn");
const inputFile = document.getElementById("file");

inputFile.addEventListener('change', () => {
    clearBtn.hidden = inputFile.files.length === 0;
});

clearBtn.addEventListener('click', () => {
    inputFile.value = "";
    clearBtn.hidden = true;
    setState("idle", "รอไฟล์...");
});