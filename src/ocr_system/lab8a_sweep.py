"""Lab 8A part 2 -- the sweep.

For every (noise level x cleaning method) pair: clean the page, OCR it, turn the text
into transcript JSON, and score it against the ground truth. The point is not any single
number but the shape of the grid -- which cleaning method stops helping, and where.
"""
import json
import sys
import time
from pathlib import Path

from .evaluation import char_error_rate, compare_fields
from .engines.paddle_engine import PaddleOCREngine
from .noise_levels import NOISE_LEVELS
from .preprocessing import clean_image, read_image
from .schemas import OCRPageResult
from .transcript_extraction import extract_transcript_fields
from .utils.io import ensure_dir, save_json

CLEAN_METHODS = ["none", "light", "heavy"]


def _reference_text(flat: dict) -> str:
    """Join the non-null ground-truth values into one string for the CER measure.

    The ground truth is a field tree, not a page of prose, so CER is defined over the
    values that actually carry content. Null fields are excluded: they contribute no
    characters and would otherwise dilute the rate toward zero.
    """
    return "\n".join(str(v) for v in flat.values() if v is not None)


def _flat_values(tree: dict) -> dict:
    from .evaluation import _flatten
    return _flatten(tree)


def evaluate_one(
    engine: PaddleOCREngine,
    image_path: str | Path,
    ground_truth: dict,
    method: str,
) -> tuple[dict, dict]:
    """Run one cell of the grid. Returns (metrics, extracted fields)."""
    started = time.time()
    image = clean_image(read_image(image_path), method=method)
    lines = engine.recognize(image, page=1)
    text = "\n".join(l.text for l in lines if l.text.strip())
    fields = extract_transcript_fields([OCRPageResult(1, text, lines, str(image_path))])
    elapsed = time.time() - started

    result = compare_fields(ground_truth, fields)
    gt_flat = _flat_values(ground_truth)
    pred_flat = _flat_values(fields)
    # CER compares like with like: only the keys the ground truth actually fills.
    hyp = "\n".join(
        str(pred_flat.get(k)) for k, v in gt_flat.items()
        if v is not None and pred_flat.get(k) is not None
    )
    metrics = {
        "accuracy": round(result["accuracy"], 4),
        "strict_accuracy": round(result["strict_accuracy"], 4),
        "non_null_accuracy": round(result["non_null_accuracy"], 4),
        "cer": round(char_error_rate(_reference_text(gt_flat), hyp), 4),
        "hall": len(result["hallucinated_fields"]),
        "extra": len(result["extra_fields"]),
        "matched_fields": result["matched_fields"],
        "total_fields": result["total_fields"],
        "lines": len(lines),
        "seconds": round(elapsed, 1),
    }
    return metrics, fields


def _page_for_level(noisy_dir: Path, level: str) -> Path | None:
    level_dir = noisy_dir / level
    if not level_dir.is_dir():
        return None
    pages = sorted(level_dir.glob("page_*.png")) or sorted(level_dir.glob("page_*.jpg"))
    return pages[0] if pages else None


def run_sweep(
    noisy_dir: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    levels: list[str] | None = None,
    methods: list[str] | None = None,
    lang: str = "th",
    tile: bool = False,
    progress: bool = True,
) -> dict:
    noisy_dir = Path(noisy_dir)
    output_dir = ensure_dir(output_dir)
    levels = levels or list(NOISE_LEVELS)
    methods = methods or CLEAN_METHODS

    with Path(ground_truth_path).open(encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Merge into any previous sweep so running one more --levels does not discard the
    # combos already measured; a re-run of the same tag replaces it.
    sweep_path = output_dir / "sweep.json"
    combos: dict[str, dict] = {}
    if sweep_path.exists():
        with sweep_path.open(encoding="utf-8") as f:
            combos = json.load(f).get("combos", {})

    engine = PaddleOCREngine(lang=lang, tile=tile)
    total = len(levels) * len(methods)
    done = 0
    for level in levels:
        image_path = _page_for_level(noisy_dir, level)
        if image_path is None:
            if progress:
                print(f"  {level}: ข้าม (ไม่พบภาพใน {noisy_dir / level})", file=sys.stderr)
            continue
        for method in methods:
            done += 1
            tag = f"{level}__{method}"
            try:
                metrics, fields = evaluate_one(engine, image_path, ground_truth, method)
            except Exception as exc:
                combos[tag] = {"level": level, "method": method, "error": str(exc)}
                if progress:
                    print(f"[{done}/{total}] {tag}: FAILED ({exc})", file=sys.stderr)
                continue
            save_json(fields, output_dir / tag / "extracted.json")
            combos[tag] = {"level": level, "method": method, "image": str(image_path), **metrics}
            if progress:
                print(
                    f"[{done}/{total}] {tag:<26} acc {metrics['accuracy']:.1%}  "
                    f"CER {metrics['cer']:.3f}  hall {metrics['hall']}  {metrics['seconds']}s",
                    file=sys.stderr,
                )

    report = {"ground_truth": str(ground_truth_path), "noisy_dir": str(noisy_dir),
              "combos": combos}
    save_json(report, sweep_path)
    return report


# Metrics where a smaller number is the better result.
LOWER_IS_BETTER = {"cer", "hall", "extra", "seconds"}


def format_grid(report: dict, value_key: str = "accuracy") -> str:
    """The §5.4 table: levels down the side, cleaning methods across, best marked."""
    better = min if value_key in LOWER_IS_BETTER else max
    combos = report["combos"]
    levels = [l for l in NOISE_LEVELS if any(c["level"] == l for c in combos.values())]
    methods = [m for m in CLEAN_METHODS if any(c["method"] == m for c in combos.values())]
    head = f"{'ระดับ noise':<16}" + "".join(f"{m:>10}" for m in methods)
    rows = [head, "-" * len(head)]
    for level in levels:
        cells = []
        vals = {m: combos.get(f"{level}__{m}", {}).get(value_key) for m in methods}
        best = better((v for v in vals.values() if v is not None), default=None)
        for m in methods:
            v = vals[m]
            cells.append("         -" if v is None else f"{v:>9.3f}{'*' if v == best else ' '}")
        rows.append(f"{level:<16}" + "".join(cells))
    rows.append("* = ดีที่สุดของแถวนั้น")
    return "\n".join(rows)
