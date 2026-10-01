import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
from jiwer import wer
import Levenshtein


@dataclass
class EvaluationResult:
    file: str
    cer: float
    wer: float
    exact_match: bool
    reference_chars: int
    prediction_chars: int


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def char_error_rate(reference: str, prediction: str) -> float:
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(prediction).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def evaluate_text(reference: str, prediction: str, file_name: str = "") -> EvaluationResult:
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    return EvaluationResult(
        file=file_name,
        cer=char_error_rate(ref, hyp),
        wer=wer(ref, hyp) if ref else (0.0 if not hyp else 1.0),
        exact_match=ref == hyp,
        reference_chars=len(ref),
        prediction_chars=len(hyp),
    )


def flatten_values(data: Any) -> list[str]:
    """Collect every non-null leaf value of a nested JSON object as a string."""
    if isinstance(data, dict):
        return [v for value in data.values() for v in flatten_values(value)]
    if isinstance(data, list):
        return [v for value in data for v in flatten_values(value)]
    if data is None:
        return []
    return [str(data)]


def _compact(text: str) -> str:
    # Structured ground truth stores values without spaces, so compare without whitespace.
    text = re.sub(r"--- Page \d+ ---", "", text)
    return re.sub(r"\s+", "", text).lower()


def evaluate_structured(ground_truth: dict, prediction_text: str, file_name: str = "") -> dict:
    """Evaluate against a structured transcript ground truth (header/transcript/footer JSON).

    - field_recall: share of ground-truth values found somewhere in the OCR text.
    - cer: character error rate of all values joined vs. OCR text. Approximate, because
      the ground-truth value order may differ from the reading order on the page.
    """
    values = [v for v in flatten_values(ground_truth) if len(_compact(v)) >= 2]
    hyp = _compact(prediction_text)
    missing = [v for v in values if _compact(v) not in hyp]
    found = len(values) - len(missing)

    ref = "".join(_compact(v) for v in values)
    return {
        "file": file_name,
        "mode": "structured",
        "field_total": len(values),
        "field_found": found,
        "field_recall": found / len(values) if values else 0.0,
        "cer": Levenshtein.distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0),
        "reference_chars": len(ref),
        "prediction_chars": len(hyp),
        "missing_values": missing,
    }


def evaluate_from_files(ground_truth_json: str | Path, prediction_json: str | Path) -> dict:
    with Path(ground_truth_json).open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with Path(prediction_json).open("r", encoding="utf-8") as f:
        prediction = json.load(f)

    return evaluate_prediction(ground_truth, prediction["text"], Path(prediction["source_path"]).name)


def evaluate_prediction(ground_truth: dict, prediction_text: str, source_name: str) -> dict:
    reference = ground_truth.get(source_name) or ground_truth.get(Path(source_name).stem)
    if isinstance(reference, str):
        result = evaluate_text(reference, prediction_text, file_name=source_name)
        return asdict(result)

    if "header_detail" in ground_truth:
        return evaluate_structured(ground_truth, prediction_text, file_name=source_name)

    raise KeyError(f"No ground truth found for {source_name}")
