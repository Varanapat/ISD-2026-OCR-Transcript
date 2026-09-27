import json
import re
import sys
from pathlib import Path
from .augmentation import AUGMENTATION_KINDS
from .engines.paddle_engine import PaddleOCREngine
from .evaluation import compare_fields
from .preprocessing import read_image
from .schemas import OCRLine, OCRPageResult
from .transcript_extraction import extract_transcript_fields
from .utils.io import save_json

LANG_BY_PREFIX = {"71": "th", "72": "en"}


def _guess_language(stem: str, ground_truth_dir: Path) -> str:
    if (ground_truth_dir / f"Json_{stem}_th.json").exists():
        return "th"
    if (ground_truth_dir / f"Json_{stem}_en.json").exists():
        return "en"
    return LANG_BY_PREFIX.get(stem[:2], "th")


def _load_ground_truth(stem: str, lang: str, ground_truth_dir: Path) -> dict | None:
    path = ground_truth_dir / f"Json_{stem}_{lang}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ocr_image_to_fields(engine: PaddleOCREngine, image_path: Path) -> dict:
    image = read_image(image_path)
    lines: list[OCRLine] = engine.recognize(image, page=1)
    text = "\n".join(line.text for line in lines if line.text.strip())
    page = OCRPageResult(page=1, text=text, lines=lines, image_path=str(image_path))
    return extract_transcript_fields([page])


def _discover_variants(doc_dir: Path) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    source_dir = doc_dir / "_source_pages"
    if source_dir.exists():
        originals = sorted(source_dir.glob("*_page_001.jpg"))
        if originals:
            variants.append(("original", originals[0]))
    for kind in AUGMENTATION_KINDS:
        candidate = doc_dir / f"page_001_{kind}.jpg"
        if candidate.exists():
            variants.append((kind, candidate))
    return variants


def run_augmented_dataset(
    augmented_dir: str | Path,
    ground_truth_dir: str | Path,
    output_path: str | Path,
    limit_documents: int | None = None,
    progress: bool = True,
) -> dict:
    augmented_dir = Path(augmented_dir)
    ground_truth_dir = Path(ground_truth_dir)

    doc_dirs = sorted(d for d in augmented_dir.iterdir() if d.is_dir())
    if limit_documents:
        doc_dirs = doc_dirs[:limit_documents]

    engines: dict[str, PaddleOCREngine] = {}
    page_records: list[dict] = []
    field_failures: dict[str, dict] = {}

    total = len(doc_dirs)
    for doc_idx, doc_dir in enumerate(doc_dirs, start=1):
        stem = doc_dir.name
        lang = _guess_language(stem, ground_truth_dir)
        ground_truth = _load_ground_truth(stem, lang, ground_truth_dir)
        if ground_truth is None:
            if progress:
                print(f"[{doc_idx}/{total}] {stem}: skipped (no ground truth)", file=sys.stderr)
            continue

        if lang not in engines:
            engines[lang] = PaddleOCREngine(lang=lang)
        engine = engines[lang]

        variants = _discover_variants(doc_dir)
        for kind, image_path in variants:
            try:
                fields = _ocr_image_to_fields(engine, image_path)
                result = compare_fields(ground_truth, fields)
            except Exception as exc:
                if progress:
                    print(f"[{doc_idx}/{total}] {stem} [{lang}] {kind}: FAILED ({exc})", file=sys.stderr)
                page_records.append({
                    "document": stem, "language": lang, "category": kind, "image": str(image_path),
                    "total_fields": None, "matched_fields": None, "accuracy": None,
                    "strict_accuracy": None,
                    "non_null_accuracy": None, "precision": None, "f1": None,
                    "num_extra_fields": None, "error": str(exc),
                })
                continue
            page_records.append({
                "document": stem,
                "language": lang,
                "category": kind,
                "image": str(image_path),
                "total_fields": result["total_fields"],
                "matched_fields": result["matched_fields"],
                "accuracy": result["accuracy"],
                "strict_accuracy": result["strict_accuracy"],
                "non_null_accuracy": result["non_null_accuracy"],
                "precision": result["precision"],
                "f1": result["f1"],
                "num_extra_fields": len(result["extra_fields"]),
            })
            for field_result in result["fields"]:
                # collapse list indices so all subjects/semesters group under one key
                key = re.sub(r"\[[^\]]*\]", "[]", field_result["field"])
                bucket = field_failures.setdefault(key, {"field": key, "fail_count": 0, "total_count": 0})
                bucket["total_count"] += 1
                if not field_result["match"]:
                    bucket["fail_count"] += 1
            if progress:
                print(
                    f"[{doc_idx}/{total}] {stem} [{lang}] {kind}: "
                    f"{result['matched_fields']}/{result['total_fields']} "
                    f"({result['accuracy']:.1%}, non-null {result['non_null_accuracy']:.1%}, "
                    f"F1 {result['f1']:.1%}, +{len(result['extra_fields'])} extra)",
                    file=sys.stderr,
                )

    field_level = sorted(field_failures.values(), key=lambda x: -x["fail_count"])
    for entry in field_level:
        entry["fail_rate"] = round(entry["fail_count"] / entry["total_count"], 4) if entry["total_count"] else 0.0

    category_level: dict[tuple[str, str], list[dict]] = {}
    for rec in page_records:
        if rec["accuracy"] is None:
            continue
        category_level.setdefault((rec["category"], rec["language"]), []).append(rec)
        category_level.setdefault((rec["category"], "all"), []).append(rec)

    def _avg(rows: list[dict], key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4)

    category_summary = [
        {
            "category": cat,
            "language": lang,
            "num_pages": len(rows),
            "avg_accuracy": _avg(rows, "accuracy"),
            "avg_strict_accuracy": _avg(rows, "strict_accuracy"),
            "avg_non_null_accuracy": _avg(rows, "non_null_accuracy"),
            "avg_precision": _avg(rows, "precision"),
            "avg_f1": _avg(rows, "f1"),
            "avg_extra_fields": _avg(rows, "num_extra_fields"),
        }
        for (cat, lang), rows in sorted(category_level.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]

    report = {
        "num_documents": len({r["document"] for r in page_records}),
        "num_pages_evaluated": len(page_records),
        "page_level": page_records,
        "category_level": category_summary,
        "field_level": field_level,
    }
    save_json(report, output_path)
    return report
