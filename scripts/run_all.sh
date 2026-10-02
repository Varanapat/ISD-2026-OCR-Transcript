#!/usr/bin/env bash
# Run the benchmark on every dataset, then build the field / page / category report.
#
#   bash scripts/run_all.sh                         # tesseract + paddle (about 2.5 hours)
#   ENGINES=tesseract bash scripts/run_all.sh       # tesseract only (about 20 minutes)
#   ENGINES=tesseract,paddle,easyocr,doctr,ensemble bash scripts/run_all.sh   # many hours
#
# Results:  outputs/benchmark*/  (per run)   and   outputs/report/  (all runs together)
# Stopped halfway? Run again: finished runs can be skipped with SKIP_DONE=1.
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINES="${ENGINES:-tesseract,paddle}"
OPTIONS="${OPTIONS:---deskew-only}"
PYTHON="${PYTHON:-.venv/bin/python}"

# name | input images/PDFs | ground truth | benchmark output
RUNS=(
  "set1|data/input|data/ground_truth|outputs/benchmark"
  "setG|data/input_G|data/ground_truth_G|outputs/benchmark_G"
  "aug_set1|data/Augmentation_input/images|data/Augmentation_input/ground_truth|outputs/benchmark_aug"
  "aug_setG|data/Augmentation_input_G/images|data/Augmentation_input_G/ground_truth|outputs/benchmark_aug_G"
)

if [ ! -d data/Augmentation_input ] || [ ! -d data/Augmentation_input_G ]; then
  echo "== creating augmented datasets"
  "$PYTHON" -m ocr_system.cli augment
  "$PYTHON" -m ocr_system.cli augment --input-dir data/input_G --ground-truth-dir data/ground_truth_G --output-dir data/Augmentation_input_G
fi

REPORT_ARGS=()
for run in "${RUNS[@]}"; do
  IFS="|" read -r name input gt out <<< "$run"
  if [ "${SKIP_DONE:-0}" = "1" ] && [ -f "$out/summary.csv" ]; then
    echo "== $name: already done ($out/summary.csv), skipping"
  else
    echo "== $name: benchmark ($ENGINES) -> $out"
    "$PYTHON" -m ocr_system.cli benchmark --input-dir "$input" --ground-truth-dir "$gt" \
      --output-dir "$out" --engines "$ENGINES" $OPTIONS
  fi
  REPORT_ARGS+=(--run "$name" "$out" "$gt")
done

echo "== report"
"$PYTHON" -m ocr_system.cli report "${REPORT_ARGS[@]}" --output-dir outputs/report
