"""Run several OCR engines over a folder of documents and compare them against ground truth."""
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console
from .config import OCRConfig
from .document_loader import is_image, is_pdf
from .engine_factory import ENSEMBLE_MEMBERS, build_engine
from .engines.ensemble_engine import EnsembleOCREngine
from .evaluation import evaluate_prediction
from .pipeline import build_result, prepare_pages
from .utils.io import ensure_dir

RESULT_COLUMNS = ["engine", "file", "lang", "field_recall", "cer", "field_found", "field_total", "seconds", "error"]


def find_ground_truth(ground_truth_dir: Path, stem: str) -> Path | None:
    """Match 71010001.pdf to Json_71010001_th.json / Json_71010001_en.json (or 71010001.json)."""
    matches = sorted(ground_truth_dir.glob(f"Json_{stem}_*.json")) or sorted(ground_truth_dir.glob(f"{stem}.json"))
    return matches[0] if matches else None


def document_language(gt_path: Path) -> str:
    """Json_71010001_th.json -> "th"; anything without a _th/_en suffix -> "-"."""
    suffix = gt_path.stem.rsplit("_", 1)[-1]
    return suffix if suffix in ("th", "en") else "-"


def collect_documents(input_dir: Path, ground_truth_dir: Path, limit: int | None) -> list[tuple[Path, Path]]:
    docs = []
    for path in sorted(input_dir.iterdir()):
        if not (is_pdf(path) or is_image(path)):
            continue
        gt = find_ground_truth(ground_truth_dir, path.stem)
        if gt is None:
            print(f"[yellow]skip[/yellow] {path.name}: no ground truth in {ground_truth_dir}")
            continue
        docs.append((path, gt))
    return docs[:limit] if limit else docs


def run_benchmark(
    base_config: OCRConfig,
    input_dir: Path,
    ground_truth_dir: Path,
    engines: list[str],
    limit: int | None = None,
) -> list[dict]:
    output_dir = ensure_dir(base_config.output_dir)
    docs = collect_documents(input_dir, ground_truth_dir, limit)
    if not docs:
        raise ValueError(f"No documents with ground truth found in {input_dir}")

    # Ensemble reuses the member engines' results instead of running them a second time.
    base_names = [e for e in engines if e != "ensemble"]
    if "ensemble" in engines:
        base_names += [m for m in ENSEMBLE_MEMBERS if m not in base_names]

    loaded = {}
    for name in base_names:
        print(f"Loading engine [cyan]{name}[/cyan] ...")
        loaded[name] = build_engine(base_config, name)

    rows: list[dict] = []
    results_path = output_dir / "results.csv"
    for doc_no, (doc_path, gt_path) in enumerate(docs, start=1):
        print(f"[bold]({doc_no}/{len(docs)})[/bold] {doc_path.name}")
        lang = document_language(gt_path)
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        doc_config = replace(base_config, input_path=doc_path, save_debug_images=False)
        pages = prepare_pages(doc_config)

        engine_lines: dict[str, list] = {}
        engine_seconds: dict[str, float] = {}
        for name, engine in loaded.items():
            start = time.perf_counter()
            try:
                engine_lines[name] = [(p, img_path, engine.recognize(img, page=p)) for p, img_path, img in pages]
            except Exception as exc:  # keep benchmarking the other engines/documents
                print(f"  [red]{name} failed[/red]: {exc}")
                if name in engines:
                    rows.append({"engine": name, "file": doc_path.name, "lang": lang, "error": str(exc)})
                continue
            engine_seconds[name] = time.perf_counter() - start

        if "ensemble" in engines and all(m in engine_lines for m in ENSEMBLE_MEMBERS):
            engine_lines["ensemble"] = [
                (p, img_path, EnsembleOCREngine.merge(*(engine_lines[m][i][2] for m in ENSEMBLE_MEMBERS)))
                for i, (p, img_path, _) in enumerate(pages)
            ]
            engine_seconds["ensemble"] = sum(engine_seconds[m] for m in ENSEMBLE_MEMBERS)

        for name in engines:
            if name not in engine_lines:
                continue
            result = build_result(replace(doc_config, output_dir=output_dir / name), name, engine_lines[name])
            metrics = evaluate_prediction(ground_truth, result.text, doc_path.name)
            row = {"engine": name, "file": doc_path.name, "lang": lang, "seconds": round(engine_seconds[name], 2)}
            row.update({k: metrics.get(k) for k in ("field_recall", "cer", "field_found", "field_total")})
            rows.append(row)
            print(f"  {name:<10} recall={_fmt(row['field_recall'])} cer={_fmt(row['cer'])} time={row['seconds']}s")

        # Write after every document so partial results survive an interrupted run.
        write_csv(rows, results_path, RESULT_COLUMNS)

    summary = summarize(rows, engines)
    write_csv(summary, output_dir / "summary.csv", list(summary[0].keys()))
    print_summary(summary)
    print(f"[green]Benchmark done[/green]: {results_path} and {output_dir / 'summary.csv'}")
    return rows


def summarize(rows: list[dict], engines: list[str]) -> list[dict]:
    """One row per engine over all documents, plus one per engine and language (th/en)."""
    langs = sorted({r["lang"] for r in rows if r["lang"] != "-"})
    groups = [("all", lambda r: True)]
    if len(langs) > 1:
        groups += [(lang, lambda r, lang=lang: r["lang"] == lang) for lang in langs]

    summary = []
    for lang, match in groups:
        group = [s for name in engines if (s := _summarize_engine(rows, name, lang, match))]
        group.sort(key=lambda s: s["field_recall"] or 0, reverse=True)
        summary.extend(group)
    return summary


def _summarize_engine(rows: list[dict], name: str, lang: str, match) -> dict | None:
    mine = [r for r in rows if r["engine"] == name and match(r)]
    if not mine:
        return None
    ok = [r for r in mine if not r.get("error")]
    found = sum(r["field_found"] or 0 for r in ok)
    total = sum(r["field_total"] or 0 for r in ok)
    return {
        "lang": lang,
        "engine": name,
        "documents": len(ok),
        "errors": len(mine) - len(ok),
        "field_recall": round(found / total, 4) if total else None,
        "avg_cer": _avg([r["cer"] for r in ok]),
        "avg_seconds_per_doc": _avg([r["seconds"] for r in ok], ndigits=2),
    }


def print_summary(summary: list[dict]) -> None:
    table = Table(title="OCR benchmark (per language, sorted by field_recall, higher is better)")
    for col in summary[0]:
        table.add_column(col)
    for i, s in enumerate(summary):
        end_section = i + 1 < len(summary) and summary[i + 1]["lang"] != s["lang"]
        table.add_row(*(_fmt(v) for v in s.values()), end_section=end_section)
    Console().print(table)


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel shows Thai correctly
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _avg(values: list, ndigits: int = 4) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), ndigits) if values else None


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-" if value is None else str(value)
