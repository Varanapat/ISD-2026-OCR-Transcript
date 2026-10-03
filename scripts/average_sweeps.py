"""Average several Lab 8A sweep runs (same noisy images, repeated because the VLM output varies).

    python scripts/average_sweeps.py outputs/lab8a/out outputs/lab8a/out_run2 outputs/lab8a/out_run3 -o outputs/lab8a/out_avg

Writes <out>/sweep_avg.csv and <out>/sweep_avg_plot.png and prints mean ± SD tables.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

LEVELS = ["L0_clean", "L1_light", "L2_watermark", "L3_tilted_copy", "L4_rescan"]
METHODS = ["none", "light", "heavy"]
METRICS = [  # key, title, best, decimals
    ("accuracy", "field accuracy (higher is better)", max, 3),
    ("cer", "CER (lower is better)", min, 3),
    ("hall", "hall: hallucinated fields (lower is better)", min, 1),
    ("subject_f1", "subject F1 (higher is better)", max, 3),
    ("seconds", "seconds, MEDIAN (one run had a 6,000 s stall)", min, 0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="sweep output folders (each has sweep.json)")
    ap.add_argument("-o", "--out", default="outputs/lab8a/out_avg")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [json.loads((Path(r) / "sweep.json").read_text(encoding="utf-8"))["results"] for r in args.runs]
    rows = []
    for lv in LEVELS:
        for m in METHODS:
            values = [r[f"{lv}__{m}"] for r in runs if f"{lv}__{m}" in r and not r[f"{lv}__{m}"].get("error")]
            if not values:
                continue
            row = {"level": lv, "method": m, "runs": len(values)}
            for key, *_ in METRICS:
                xs = [v[key] for v in values if v.get(key) is not None]
                center = statistics.median(xs) if key == "seconds" else statistics.mean(xs)
                row[f"{key}_mean"] = round(center, 4)
                row[f"{key}_sd"] = round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0
                row[f"{key}_min"], row[f"{key}_max"] = min(xs), max(xs)
                row[f"{key}_runs"] = " / ".join(str(x) for x in xs)
            rows.append(row)

    with open(out / "sweep_avg.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by = {(r["level"], r["method"]): r for r in rows}
    print(f"\n  {len(runs)} runs: {', '.join(args.runs)}")
    for key, title, best, dec in METRICS:
        print(f"\n  {title}   mean ± SD")
        print(f"  {'noise level':<16}" + "".join(f"{m:>18}" for m in METHODS))
        for lv in LEVELS:
            means = {m: by[(lv, m)][f"{key}_mean"] for m in METHODS if (lv, m) in by}
            b = best(means.values()) if means else None
            cells = []
            for m in METHODS:
                if (lv, m) not in by:
                    cells.append(f"{'—':>18}")
                    continue
                r = by[(lv, m)]
                txt = f"{r[f'{key}_mean']:.{dec}f} ± {r[f'{key}_sd']:.{dec}f}" + (" *" if r[f"{key}_mean"] == b else "  ")
                cells.append(f"{txt:>18}")
            print(f"  {lv:<16}" + "".join(cells))
    print("  * = best of the row (by mean)")
    print(f"\n  CSV (every run's value too): {out / 'sweep_avg.csv'}")
    plot(by, out / "sweep_avg_plot.png", len(runs))


def plot(by: dict, path: Path, n_runs: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"none": "#7a7a7a", "light": "#1f77b4", "heavy": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, label) in zip(axes, [("accuracy", "field accuracy"), ("cer", "CER"), ("hall", "hallucinated fields")]):
        for i, m in enumerate(METHODS):
            xs = [j + (i - 1) * 0.06 for j, lv in enumerate(LEVELS) if (lv, m) in by]
            pts = [by[(lv, m)] for lv in LEVELS if (lv, m) in by]
            ys = [p[f"{key}_mean"] for p in pts]
            lo = [y - p[f"{key}_min"] for y, p in zip(ys, pts)]
            hi = [p[f"{key}_max"] - y for y, p in zip(ys, pts)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=m, color=colors[m])
        ax.set_xticks(range(len(LEVELS)), [lv.split("_")[0] for lv in LEVELS])
        ax.set_title(f"{label} (mean, bar = min-max)")
        ax.grid(alpha=0.3)
    axes[0].legend(title="cleaning")
    fig.suptitle(f"Lab 8A — noise level vs. extraction quality, average of {n_runs} runs")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot: {path}")


if __name__ == "__main__":
    main()
