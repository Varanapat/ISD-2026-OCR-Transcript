"""ตรวจว่า API Contract (บล็อก JSON ใน README.md) ตรงกับที่ FastAPI สร้างจริงจากโค้ด

รันจากรากโปรเจกต์ (ocr_system):

    python -m lab10_fastapi.transcript_app.check_contract

ไม่เรียก Ollama และไม่ต้องรัน server
"""

import json
import re
import sys
from pathlib import Path

from .config import settings
from .main import app, health
from .pipeline_service import ALLOWED_SUFFIXES, METHODS

APP_DIR = Path(__file__).resolve().parent
MAIN_SOURCE = (APP_DIR / "main.py").read_text(encoding="utf-8")

_results: list[bool] = []


def check(name: str, got, want) -> None:
    ok = got == want
    _results.append(ok)
    print(("  [ ok ] " if ok else "  [FAIL] ") + name
          + ("" if ok else f"\n           README: {want!r}\n           โค้ดจริง: {got!r}"))


def load_contract() -> dict:
    """หาบล็อก ```json ใน README.md ที่มี contract_version"""
    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.S):
        if '"contract_version"' in block:
            return json.loads(block)
    raise SystemExit("ไม่พบบล็อก JSON ของ contract ใน README.md")


def short_type(schema: dict, components: dict) -> str:
    """แปลง JSON Schema ของ OpenAPI เป็นชนิดแบบย่อที่ใช้ใน README"""
    if "$ref" in schema:
        return short_type(components[schema["$ref"].split("/")[-1]], components)
    if "anyOf" in schema:
        kinds = [short_type(s, components) for s in schema["anyOf"]]
        return "|".join(sorted(kinds, key=lambda k: (k == "null", k)))   # null ไว้ท้ายเสมอ
    kind = schema.get("type")
    if kind == "array":
        return f"array<{short_type(schema.get('items', {}), components)}>"
    return kind or "any"


def python_type(value) -> str:
    return {str: "string", bool: "boolean", int: "integer", float: "number"}.get(type(value), "object")


def main() -> int:
    contract = load_contract()
    spec = app.openapi()
    components = spec["components"]["schemas"]
    print(f"เทียบ contract v{contract['contract_version']} กับ OpenAPI {spec['openapi']}\n")

    for endpoint in contract["endpoints"]:
        label = f"{endpoint['method']} {endpoint['path']}"
        print(label)
        path_item = spec["paths"].get(endpoint["path"], {})
        operation = path_item.get(endpoint["method"].lower())
        check("endpoint และ method มีจริง", operation is not None, True)
        if operation is None:
            continue

        # ---- request ----
        request = endpoint.get("request")
        if request:
            content = operation["requestBody"]["content"]
            check("content type", list(content), [request["content_type"]])
            body = components[content[request["content_type"]]["schema"]["$ref"].split("/")[-1]]
            props, required = body["properties"], set(body.get("required", []))
            check("ชื่อ field ของ request", sorted(props), sorted(request["fields"]))
            for name, spec_field in request["fields"].items():
                check(f"field '{name}' บังคับหรือไม่", name in required, spec_field["required"])
                if "default" in spec_field:
                    check(f"field '{name}' ค่าเริ่มต้น", props[name].get("default"), spec_field["default"])
            check("นามสกุลไฟล์ที่รับ", sorted(request["fields"]["file"]["accept"]), sorted(ALLOWED_SUFFIXES))
            check("ขนาดไฟล์สูงสุด (MB)", request["fields"]["file"]["max_mb"], settings.max_upload_mb)
            pre_source = (APP_DIR / "pipeline_service.py").read_text(encoding="utf-8")
            enum_in_code = re.search(r'preprocessing not in \{([^}]*)\}', pre_source)
            check("ค่า method ที่รับ", sorted(METHODS), sorted(request["fields"]["method"]["enum"]))
            check("ค่า preprocessing ที่รับ",
                  sorted(re.findall(r'"(\w+)"', enum_in_code.group(1))) if enum_in_code else None,
                  sorted(request["fields"]["preprocessing"]["enum"]))

        # ---- response 200 ----
        want_ok = endpoint["responses"]["200"]
        schema_200 = operation["responses"]["200"]["content"]["application/json"]["schema"]
        if "$ref" in schema_200:   # endpoint ที่มี response_model (มี schema รายฟิลด์)
            resolved = components[schema_200["$ref"].split("/")[-1]]["properties"]
            got = {k: short_type(v, components) for k, v in resolved.items()}
            check("ชื่อ field ของ response 200", sorted(got), sorted(want_ok))
            check("ชนิดของแต่ละ field", got, want_ok)
        else:            # endpoint ที่ไม่มี response_model (health): เรียกฟังก์ชันตรง ๆ แล้วดูค่าจริง
            actual = health()
            check("ชื่อ field ของ response 200", sorted(actual), sorted(want_ok))
            check("ชนิดของแต่ละ field", {k: python_type(v) for k, v in actual.items()}, want_ok)

        # ---- error code ที่ main.py ส่งเอง ----
        if "errors" in endpoint:
            documented = sorted(int(c) for c in endpoint["errors"] if c.isdigit())
            in_code = sorted(int(c) for c in re.findall(r"status_code=(\d+)", MAIN_SOURCE))
            in_code = sorted(set(in_code) | {422})   # 422 ยังมาจาก except (ValueError, RuntimeError)
            check("รหัส error ที่ระบบส่งเอง", documented, in_code)
        print()

    failed = _results.count(False)
    print(f"ผ่าน {_results.count(True)} · ไม่ผ่าน {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
