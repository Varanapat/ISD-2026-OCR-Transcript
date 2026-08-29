import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import Levenshtein


@dataclass
class FieldStat:
    name: str = ""
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total > 0 else 0.0

    def add(
        self, 
        reference: Any, 
        hypothesis: Any, 
        field_name: str = "", 
        mode: str = "soft", 
        track_wer: bool = False
    ):
        """สะสมผลการประเมินค่า reference กับ hypothesis"""
        if not isinstance(self.total, int):
            self.total = int(self.total) if str(self.total).isdigit() else 0

        # จัดการกรณีเกรดที่มีเครื่องหมายลบ (-) ปนมา ให้ลบออก
        if field_name == "grade" or "grade" in field_name:
            if isinstance(hypothesis, str):
                hypothesis = re.sub(r"-", "", hypothesis)
            if isinstance(reference, str):
                reference = re.sub(r"-", "", reference)

        ref_norm = normalize(str(reference) if reference is not None else None, mode=mode)
        hyp_norm = normalize(str(hypothesis) if hypothesis is not None else None, mode=mode)
        
        self.total += 1
        if ref_norm == hyp_norm and (ref_norm != "" or reference == hypothesis):
            self.correct += 1


@dataclass
class AlignResult:
    matched: list = field(default_factory=list)
    missed: list = field(default_factory=list)
    spurious: list = field(default_factory=list)

    @property
    def precision(self) -> float:
        total_pred = len(self.matched) + len(self.spurious)
        return len(self.matched) / total_pred if total_pred > 0 else 0.0

    @property
    def recall(self) -> float:
        total_gt = len(self.matched) + len(self.missed)
        return len(self.matched) / total_gt if total_gt > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def align_by_key(gt_list: list, pred_list: list, key_fn) -> AlignResult:
    """จับคู่ข้อมูลจาก Ground Truth และ Prediction ด้วย Key จาก key_fn"""
    result = AlignResult()
    pred_map = {}
    for p in pred_list:
        k = key_fn(p)
        if k:
            pred_map[k] = p
    
    for gt in gt_list:
        k = key_fn(gt)
        if k in pred_map:
            result.matched.append((gt, pred_map.pop(k)))
        else:
            result.missed.append(gt)
            
    result.spurious.extend(pred_map.values())
    return result


def normalize(text: Optional[Any], mode: str = "soft") -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if mode == "strict":
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE).lower()
    return re.sub(r"\s+", " ", text).lower()


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference, "soft")
    hyp = normalize(hypothesis, "soft")
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def print_table(stats: Dict[str, FieldStat], title: str = "") -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")
    print(f"  {'Field Description':<30} | {'Correct/Total':<15} | {'Accuracy':<10}")
    print(f"  {'-' * 71}")
    for k, stat in stats.items():
        disp_name = stat.name if stat.name else k
        acc = f"{stat.accuracy * 100:.2f}%"
        counts = f"{stat.correct}/{stat.total}"
        print(f"  {disp_name:<30} | {counts:<15} | {acc:<10}")
    print(f"{'=' * 75}\n")


def print_errors(stats: Dict[str, FieldStat], limit: int = 5) -> None:
    pass


def stats_to_dict(stats: Dict[str, FieldStat]) -> dict:
    return {
        k: {
            "name": stat.name,
            "total": stat.total,
            "correct": stat.correct,
            "accuracy": round(stat.accuracy, 4),
        }
        for k, stat in stats.items()
    }


def save_csv(stats: Dict[str, FieldStat], filepath: str, extra: Optional[dict] = None) -> None:
    import csv
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "description", "total", "correct", "accuracy"])
        for k, stat in stats.items():
            writer.writerow([k, stat.name, stat.total, stat.correct, round(stat.accuracy, 4)])