import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
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


def evaluate_from_files(ground_truth_json: str | Path, prediction_json: str | Path) -> dict:
    with Path(ground_truth_json).open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with Path(prediction_json).open("r", encoding="utf-8") as f:
        prediction = json.load(f)

    source_name = Path(prediction["source_path"]).name
    reference = ground_truth.get(source_name) or ground_truth.get(Path(source_name).stem)
    if reference is None:
        raise KeyError(f"No ground truth found for {source_name}")

    result = evaluate_text(reference, prediction["text"], file_name=source_name)
    return asdict(result)


# Lists whose items carry a stable identity. Flattening these by position lets one
# missed item shift every later index, so a near-perfect read scores near zero;
# keying by identity keeps the items that did survive comparable.
LIST_ID_FIELDS = {
    "subject": ("subject_id",),
    "semesters": ("year", "sem_num"),
}



def _list_item_keys(items: list, list_name: str) -> list[str]:
    id_fields = LIST_ID_FIELDS.get(list_name)
    keys: list[str] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(items):
        key = None
        if id_fields and isinstance(item, dict):
            values = [item.get(f) for f in id_fields]
            if all(v is not None and v != "" for v in values):
                key = "|".join(str(v) for v in values)
        if key is None:
            # No usable identity (e.g. OCR lost the subject id) -- fall back to position.
            key = f"#{index}"
        count = seen.get(key, 0)
        seen[key] = count + 1
        keys.append(key if count == 0 else f"{key}#{count}")
    return keys


def _flatten(value, prefix: str = "") -> dict:
    flat: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            flat.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(value, list):
        list_name = prefix.rsplit(".", 1)[-1].split("[", 1)[0]
        for key, item in zip(_list_item_keys(value, list_name), value):
            flat.update(_flatten(item, f"{prefix}[{key}]"))
    else:
        flat[prefix] = value
    return flat


_MISSING = object()

# Tone marks (U+0E48-U+0E4B) and thanthakhat (U+0E4C) ride above the base character and
# are the first thing OCR loses on a Thai scan. Comparing without them stops a reading
# that got every letter right from scoring zero on the whole field -- "เกล้า" read as
# "เกลา" is a recognisable win, not a failure. Vowels are left alone: "อำ" read as "อา"
# is a different vowel, not a dropped diacritic, and folding those would hide real errors.
_THAI_DIACRITIC_RE = re.compile(r"[\u0e48-\u0e4c]")


def strip_diacritics(value):
    """Remove Thai tone marks from a string; other values pass through untouched."""
    return _THAI_DIACRITIC_RE.sub("", value) if isinstance(value, str) else value


def compare_fields(ground_truth: dict, prediction: dict, normalize: bool = True) -> dict:
    gt_flat = _flatten(ground_truth)
    pred_flat = _flatten(prediction)
    mismatches = []
    fields = []
    matched = 0
    strict_matched = 0
    non_null_total = 0
    non_null_matched = 0
    for key, gt_val in gt_flat.items():
        pred_val = pred_flat.get(key, _MISSING)
        # A key the prediction never produced is a miss even when the ground truth
        # value is null -- otherwise a subject OCR dropped entirely still scores on
        # every null field it would have carried.
        present = pred_val is not _MISSING
        strict_match = present and gt_val == pred_val
        is_match = (
            present and strip_diacritics(gt_val) == strip_diacritics(pred_val)
            if normalize
            else strict_match
        )
        strict_matched += strict_match
        fields.append({"field": key, "match": is_match})
        if gt_val is not None:
            non_null_total += 1
            non_null_matched += is_match
        if is_match:
            matched += 1
        else:
            entry = {
                "field": key,
                "ground_truth": gt_val,
                "prediction": None if pred_val is _MISSING else pred_val,
            }
            if not present:
                entry["missing"] = True
            mismatches.append(entry)

    # Lab 8A's "hall": the ground truth leaves the box blank but the pipeline filled it
    # in. Distinct from extra_fields, which is a key the ground truth does not have at
    # all; this is a key it has and deliberately left null.
    hallucinated = [
        {"field": key, "prediction": pred_flat[key]}
        for key, gt_val in gt_flat.items()
        if gt_val is None and pred_flat.get(key) is not None
    ]

    # Fields the prediction invented. Recall alone never penalises these.
    extra_fields = [
        {"field": key, "prediction": value}
        for key, value in pred_flat.items()
        if key not in gt_flat
    ]

    total = len(gt_flat)
    predicted = len(pred_flat)
    recall = matched / total if total else 0.0
    precision = matched / predicted if predicted else 0.0
    return {
        "total_fields": total,
        "matched_fields": matched,
        "fields": fields,
        # Recall over ground-truth fields, ignoring Thai tone marks by default.
        "accuracy": recall,
        # The same count with tone marks required, so a report can state exactly how
        # much of the score the relaxation is worth.
        "strict_accuracy": strict_matched / total if total else 0.0,
        "strict_matched_fields": strict_matched,
        # ~20% of ground-truth fields are null, and the extractor hardcodes several
        # of them to None, so they match for free. This excludes them.
        "non_null_accuracy": non_null_matched / non_null_total if non_null_total else 0.0,
        "predicted_fields": predicted,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mismatches": mismatches,
        "extra_fields": extra_fields,
        "hallucinated_fields": hallucinated,
    }


def evaluate_fields_from_files(ground_truth_json: str | Path, prediction_json: str | Path) -> dict:
    with Path(ground_truth_json).open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with Path(prediction_json).open("r", encoding="utf-8") as f:
        prediction = json.load(f)
    return compare_fields(ground_truth, prediction)
