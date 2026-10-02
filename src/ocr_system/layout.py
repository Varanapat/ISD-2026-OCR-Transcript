"""Rule-based layout analysis: split a page into named regions before/after OCR.

1. Line-based morphology finds the ruled table: long vertical lines give the column
   boundaries, long horizontal lines give the table frame and the header row.
2. Projection profiles split the area above/below the table: empty rows separate
   blocks, and a wide empty gap in the middle splits a block into left/right columns.

Resulting regions (in reading order), e.g. for a transcript page:
    header, info_left, info_right, table_header, table_col1 ... table_col6, footer_left, footer_right
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
from .schemas import OCRLine


@dataclass
class Region:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        # numpy ints -> plain ints so regions can be saved to JSON
        self.x0, self.y0, self.x1, self.y1 = int(self.x0), int(self.y0), int(self.x1), int(self.y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def distance(self, x: float, y: float) -> float:
        dx = max(self.x0 - x, 0, x - self.x1)
        dy = max(self.y0 - y, 0, y - self.y1)
        return (dx * dx + dy * dy) ** 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask where ink (text/lines) = 255, paper = 0."""
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where values is True."""
    padded = np.concatenate([[False], values, [False]]).astype(int)
    diff = np.diff(padded)
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def _line_positions(profile: np.ndarray, min_length: float) -> list[int]:
    """Centers of ruled lines from a projection of a line mask."""
    return [int((s + e) / 2) for s, e in _runs(profile >= min_length)]


# ---------------------------------------------------------------- 1. table (morphology)

def detect_table(ink: np.ndarray) -> list[Region]:
    h, w = ink.shape
    # Kernels relative to page size: much longer than any character stroke (~40-50px at 300 dpi).
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30)))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1)))

    contours, _ = cv2.findContours(vertical | horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    x, y, tw, th = cv2.boundingRect(max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3]))
    if tw < 0.4 * w or th < 0.1 * h:
        return []
    x1, y1 = x + tw, y + th

    # Column boundaries: vertical lines spanning at least half the table height.
    xs = _line_positions(vertical[y:y1, x:x1].sum(axis=0) / 255, 0.5 * th)
    xs = sorted({x, x1, *(x + v for v in xs)})
    xs = [v for i, v in enumerate(xs) if i == 0 or v - xs[i - 1] > 10]
    # Header row: first full-width horizontal line just below the top border
    # (ignores short underlines such as semester titles).
    ys = [y + v for v in _line_positions(horizontal[y:y1, x:x1].sum(axis=1) / 255, 0.8 * tw)]
    header_bottom = next((v for v in ys if y + 10 < v < y + 0.2 * th), None)

    inset = 6  # keep the ruled lines out of crops
    regions = []
    body_top = y
    if header_bottom is not None:
        regions.append(Region("table_header", x + inset, y + inset, x1 - inset, header_bottom - inset))
        body_top = header_bottom
    for i, (left, right) in enumerate(zip(xs, xs[1:]), start=1):
        regions.append(Region(f"table_col{i}", left + inset, body_top + inset, right - inset, y1 - inset))
    return regions


# ---------------------------------------------------------------- 2. blocks (projection profile)

def _text_lines(block: np.ndarray, max_gap: int = 10) -> list[tuple[int, int]]:
    """Row ranges of text lines. Thai vowels/tone marks above or below a line are separated
    from it by a few empty rows, so ink rows closer than max_gap belong to the same line."""
    lines: list[list[int]] = []
    for s, e in _runs(block.sum(axis=1) > 0):
        if lines and s - lines[-1][1] <= max_gap:
            lines[-1][1] = e
        else:
            lines.append([s, e])
    return [(s, e) for s, e in lines]


def _best_gap(lines_with_ink: np.ndarray, tolerance: int, x0: int, w: int) -> int | None:
    best, split = 0, None
    for s, e in _runs(lines_with_ink <= tolerance):
        if s == 0 or e == len(lines_with_ink):  # a column gap needs text on both sides
            continue
        # Only gaps around the middle of the page count; margins are not column gaps.
        overlap = min(e + x0, int(0.7 * w)) - max(s + x0, int(0.3 * w))
        if overlap < 0.04 * w or overlap <= best:
            continue
        best = overlap
        # Split inside the part of the gap that no line crosses, so no text is cut.
        empty = _runs(lines_with_ink[s:e] == 0)
        a, b = max(empty, key=lambda r: r[1] - r[0]) if empty else (0, e - s)
        split = x0 + s + (a + b) // 2
    return split


def _split_columns(ink: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int | None:
    """x where a block splits into left/right columns, or None if it is a single column."""
    w = ink.shape[1]
    block = ink[y0:y1, x0:x1]
    lines = _text_lines(block)
    # How many text lines have ink in each pixel column.
    lines_with_ink = sum((block[s:e].sum(axis=0) > 0).astype(int) for s, e in lines)
    split = _best_gap(lines_with_ink, 0, x0, w)
    if split is None and len(lines) >= 3:
        # Allow one line to run into the gap (e.g. a long "Program : ..." line on the left).
        split = _best_gap(lines_with_ink, 1, x0, w)
    return split


def detect_blocks(ink: np.ndarray, y_start: int, y_end: int, base: str, after_columns: str) -> list[Region]:
    """Split rows [y_start, y_end) into text blocks separated by empty bands.

    Single-column blocks are named `base`; once a two-column block appears, it and
    everything after it is named `after_columns` (+ _left/_right).
    """
    h, w = ink.shape
    if y_end - y_start < 5:
        return []
    has_ink = ink[y_start:y_end].sum(axis=1) > 0
    min_gap = int(0.01 * h)  # ~35px at 300 dpi: larger than the gap between lines of one block

    # Merge ink rows separated by less than min_gap into blocks.
    blocks: list[list[int]] = []
    for s, e in _runs(has_ink):
        s, e = s + y_start, e + y_start
        if blocks and s - blocks[-1][1] < min_gap:
            blocks[-1][1] = e
        else:
            blocks.append([s, e])

    regions: dict[str, Region] = {}
    name = base
    last_split = None
    pad = 10
    for s, e in blocks:
        cols = np.where(ink[s:e].sum(axis=0) > 0)[0]
        left, right = int(cols.min()), int(cols.max()) + 1
        split = _split_columns(ink, left, s, right, e)
        if split is not None:
            name, last_split = after_columns, split
            parts = [(f"{name}_left", left, split), (f"{name}_right", split, right)]
        elif last_split is not None and left >= last_split:  # e.g. signature under the right column
            parts = [(f"{name}_right", left, right)]
        elif last_split is not None and right <= last_split:
            parts = [(f"{name}_left", left, right)]
        else:
            parts = [(name, left, right)]
        for part, px0, px1 in parts:
            # Pad only the outer edges so left/right columns do not overlap at the split.
            bx0 = px0 if px0 == split else max(px0 - pad, 0)
            bx1 = px1 if px1 == split else min(px1 + pad, w)
            box = Region(part, bx0, max(s - pad, 0), bx1, min(e + pad, h))
            if part in regions:  # same name again (e.g. several header lines): grow the region
                r = regions[part]
                box = Region(part, min(r.x0, box.x0), min(r.y0, box.y0), max(r.x1, box.x1), max(r.y1, box.y1))
            regions[part] = box
    return list(regions.values())


# ---------------------------------------------------------------- page

def detect_regions(image: np.ndarray) -> list[Region]:
    """Named regions of one page, in reading order."""
    ink = _ink_mask(image)
    h = ink.shape[0]
    table = detect_table(ink)
    if not table:
        return detect_blocks(ink, 0, h, "header", "body")

    top = min(r.y0 for r in table) - 6
    bottom = max(r.y1 for r in table) + 6
    above = detect_blocks(ink, 0, top, "header", "info")
    below = detect_blocks(ink, bottom + 1, h, "footer", "footer")
    return above + table + below


def assign_regions(lines: list[OCRLine], regions: list[Region]) -> list[OCRLine]:
    """Method B: label each OCR line with the region that contains the center of its box."""
    for line in lines:
        if not line.box or not regions:
            continue
        cx = sum(p[0] for p in line.box) / len(line.box)
        cy = sum(p[1] for p in line.box) / len(line.box)
        inside = [r for r in regions if r.contains(cx, cy)]
        line.region = (inside[0] if inside else min(regions, key=lambda r: r.distance(cx, cy))).name
    return lines


def recognize_by_crop(engine, image: np.ndarray, regions: list[Region], page: int | None = None) -> list[OCRLine]:
    """Method A: OCR each region separately, then shift boxes back to page coordinates."""
    lines: list[OCRLine] = []
    for region in regions:
        if region.x1 - region.x0 < 10 or region.y1 - region.y0 < 10:
            continue
        crop = image[region.y0:region.y1, region.x0:region.x1]
        if _ink_mask(crop).sum() == 0:  # empty cell, e.g. unused right half of the table
            continue
        for line in engine.recognize(crop, page=page):
            if line.box:
                line.box = [[float(px) + region.x0, float(py) + region.y0] for px, py in line.box]
            line.region = region.name
            lines.append(line)
    return lines


def group_text_by_region(lines: list[OCRLine], regions: list[Region]) -> str:
    """Page text with one "[region]" heading per region, in reading order."""
    order = [r.name for r in regions]
    order += sorted({l.region for l in lines if l.region and l.region not in order})
    parts = []
    for name in order:
        texts = [l.text for l in lines if l.region == name and l.text.strip()]
        if texts:
            parts.append(f"[{name}]\n" + "\n".join(texts))
    return "\n\n".join(parts)


def save_layout_image(image: np.ndarray, regions: list[Region], path: str | Path) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    for i, r in enumerate(regions):
        color = colors[i % len(colors)]
        cv2.rectangle(canvas, (r.x0, r.y0), (r.x1, r.y1), color, 4)
        cv2.putText(canvas, r.name, (r.x0 + 6, r.y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
