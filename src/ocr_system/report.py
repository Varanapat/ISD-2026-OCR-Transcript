"""Evaluate benchmark results at field, page and category level, across several datasets.

Reads the OCR results a benchmark run saved (<benchmark dir>/<engine>/<doc>_ocr.json),
extracts the transcript again with the current transcript_extraction.py (so code changes
are measured without re-running OCR) and compares it with the ground truth field by field.

Levels
- field:    accuracy per field type, e.g. header.name, subject.grade_earn, semester.GPA
- page:     one row per document page: fields correct, accuracy, fully correct or not
- category: accuracy per field group (header / semester / subject / summary / footer)
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .benchmark import document_language, document_variant, find_ground_truth
from .evaluation import _flatten, _normalize_field
from .transcript_extraction import extract_transcript_from_file

CATEGORIES = ["header", "semester", "subject", "summary", "footer"]

# Runs used when none are given: (dataset label, benchmark output dir, ground truth dir).
DEFAULT_RUNS = [
    ("set1", "outputs/benchmark", "data/ground_truth"),
    ("setG", "outputs/benchmark_G", "data/ground_truth_G"),
    ("aug_set1", "outputs/benchmark_aug", "data/Augmentation_input/ground_truth"),
    ("aug_setG", "outputs/benchmark_aug_G", "data/Augmentation_input_G/ground_truth"),
]


def field_name(path: str) -> tuple[str, str]:
    """("category", "field") of a ground truth path.

    transcript_detail.semesters[2].subject[5].grade_earn -> ("subject", "subject.grade_earn")
    transcript_detail.semesters[2].GPA                   -> ("semester", "semester.GPA")
    transcript_detail.total_credits_earned               -> ("summary", "summary.total_credits_earned")
    header_detail.name / footer_detail.by.by_reg         -> ("header", "header.name") / ("footer", "footer.by.by_reg")
    """
    plain = re.sub(r"\[\d+\]", "", path)
    if ".subject." in plain:
        return "subject", "subject." + plain.rsplit(".", 1)[-1]
    if ".semesters." in plain:
        return "semester", "semester." + plain.rsplit(".", 1)[-1]
    section, rest = plain.split(".", 1)
    category = {"header_detail": "header", "footer_detail": "footer", "transcript_detail": "summary"}[section]
    return category, f"{category}.{rest}"


def compare(prediction: dict, ground_truth: dict) -> list[dict]:
    """One record per non-null ground truth field: category, field, correct, missing."""
    predicted = _flatten(prediction)
    records = []
    for path, expected in _flatten(ground_truth).items():
        if expected is None:
            continue
        got = predicted.get(path)
        category, name = field_name(path)
        records.append({
            "category": category,
            "field": name,
            "correct": _normalize_field(got) == _normalize_field(expected),
            "missing": got is None,
        })
    return records


def evaluate_runs(runs: list[tuple[str, Path, Path]]) -> tuple[list[dict], list[dict]]:
    """Returns (page rows, field records with page context)."""
    pages, records = [], []
    for dataset, bench_dir, gt_dir in runs:
        if not bench_dir.exists():
            print(f"skip {dataset}: {bench_dir} not found (run the benchmark first)")
            continue
        engines = sorted(d.name for d in bench_dir.iterdir() if d.is_dir() and any(d.glob("*_ocr.json")))
        for engine in engines:
            for ocr_json in sorted((bench_dir / engine).glob("*_ocr.json")):
                stem = ocr_json.stem.removesuffix("_ocr")
                gt_path = find_ground_truth(gt_dir, stem)
                if gt_path is None:
                    continue
                context = {
                    "dataset": dataset,
                    "engine": engine,
                    "document": stem,
                    "lang": document_language(gt_path),
                    "variant": document_variant(Path(stem)),
                }
                try:
                    prediction = extract_transcript_from_file(ocr_json)
                except Exception as exc:  # keep going; the page is reported with its error
                    pages.append({**context, "error": str(exc)})
                    continue
                doc_records = compare(prediction, json.loads(gt_path.read_text(encoding="utf-8")))
                records += [{**context, **r} for r in doc_records]
                pages.append({**context, **_page_row(doc_records)})
    return pages, records


def _page_row(records: list[dict]) -> dict:
    correct = sum(r["correct"] for r in records)
    row = {
        "fields_correct": correct,
        "fields_total": len(records),
        "fields_missing": sum(r["missing"] for r in records),
        "page_accuracy": round(correct / len(records), 4) if records else None,
        "all_fields_correct": correct == len(records),
    }
    for category in CATEGORIES:
        mine = [r for r in records if r["category"] == category]
        row[f"{category}_accuracy"] = round(sum(r["correct"] for r in mine) / len(mine), 4) if mine else None
    return row


def _groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """all / th / en and, for augmented datasets, one group per augmentation."""
    groups = [("all", rows)]
    for lang in sorted({r["lang"] for r in rows} - {"-"}):
        groups.append((lang, [r for r in rows if r["lang"] == lang]))
    for variant in sorted({r["variant"] for r in rows} - {"-"}, key=lambda v: (v != "original", v)):
        groups.append((f"aug:{variant}", [r for r in rows if r["variant"] == variant]))
    return groups


def _by_run(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    runs = defaultdict(list)
    for r in rows:
        runs[(r["dataset"], r["engine"])].append(r)
    return runs


def field_level(records: list[dict]) -> list[dict]:
    out = []
    for (dataset, engine), mine in _by_run(records).items():
        for group, rows in _groups(mine):
            by_field = defaultdict(list)
            for r in rows:
                by_field[(r["category"], r["field"])].append(r)
            for (category, field), rs in sorted(by_field.items(), key=lambda x: (CATEGORIES.index(x[0][0]), x[0][1])):
                correct = sum(r["correct"] for r in rs)
                missing = sum(r["missing"] for r in rs)
                out.append({
                    "dataset": dataset, "engine": engine, "group": group, "category": category, "field": field,
                    "correct": correct, "missing": missing, "wrong": len(rs) - correct - missing,
                    "total": len(rs), "accuracy": round(correct / len(rs), 4),
                })
    return out


def category_level(records: list[dict]) -> list[dict]:
    out = []
    for (dataset, engine), mine in _by_run(records).items():
        for group, rows in _groups(mine):
            for category in CATEGORIES + ["overall"]:
                rs = rows if category == "overall" else [r for r in rows if r["category"] == category]
                if not rs:
                    continue
                correct = sum(r["correct"] for r in rs)
                out.append({
                    "dataset": dataset, "engine": engine, "group": group, "category": category,
                    "correct": correct, "missing": sum(r["missing"] for r in rs), "total": len(rs),
                    "accuracy": round(correct / len(rs), 4),
                })
    return out


def page_summary(pages: list[dict]) -> list[dict]:
    out = []
    for (dataset, engine), mine in _by_run(pages).items():
        for group, rows in _groups(mine):
            ok = [r for r in rows if r.get("page_accuracy") is not None]
            if not ok:
                continue
            accuracies = sorted(r["page_accuracy"] for r in ok)
            out.append({
                "dataset": dataset, "engine": engine, "group": group, "pages": len(ok),
                "errors": len(rows) - len(ok),
                "mean_page_accuracy": round(sum(accuracies) / len(accuracies), 4),
                "median_page_accuracy": accuracies[len(accuracies) // 2],
                "min_page_accuracy": accuracies[0],
                "pages_>=90%": sum(a >= 0.9 for a in accuracies),
                "pages_100%": sum(r["all_fields_correct"] for r in ok),
            })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Thai shows correctly in Excel
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns))
    Console().print(table)


def build_report(runs: list[tuple[str, Path, Path]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages, records = evaluate_runs(runs)
    if not records:
        raise SystemExit("Nothing to evaluate: no benchmark results with ground truth were found.")

    fields = field_level(records)
    categories = category_level(records)
    summary = page_summary(pages)
    write_csv(fields, output_dir / "field_level.csv")
    write_csv(pages, output_dir / "page_level.csv")
    write_csv(summary, output_dir / "page_summary.csv")
    write_csv(categories, output_dir / "category_level.csv")

    # Console overview: category accuracy (all documents) per dataset and engine.
    overview = defaultdict(dict)
    for row in categories:
        if row["group"] == "all":
            overview[(row["dataset"], row["engine"])][row["category"]] = row["accuracy"]
    _print_table(
        "Category level (all documents)",
        [{"dataset": d, "engine": e, **{c: v.get(c, "-") for c in CATEGORIES + ["overall"]}} for (d, e), v in overview.items()],
        ["dataset", "engine", *CATEGORIES, "overall"],
    )
    _print_table(
        "Page level",
        [row for row in summary if row["group"] == "all" or row["group"].startswith("aug:")],
        ["dataset", "engine", "group", "pages", "mean_page_accuracy", "min_page_accuracy", "pages_>=90%", "pages_100%"],
    )
    worst = sorted((r for r in fields if r["group"] == "all"), key=lambda r: r["accuracy"])[:10]
    _print_table("Field level: 10 weakest fields", worst, ["dataset", "engine", "field", "accuracy", "correct", "missing", "wrong", "total"])
    print(f"Report written to {output_dir}: field_level.csv, page_level.csv, page_summary.csv, category_level.csv")
