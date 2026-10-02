"""Create an augmented, labeled copy of a document dataset to test OCR robustness.

Every page gets one image per augmentation (one distortion at a time, so the benchmark
shows which distortion hurts which engine), plus the clean "original" render as control.
The distortions do not change the document's text, so each image is labeled with a copy
of its source ground truth. Random values are written to manifest.csv.
"""
import csv
import json
import random
import shutil
import zlib
from pathlib import Path
import cv2
import numpy as np
from rich import print
from .benchmark import find_ground_truth
from .document_loader import is_image, is_pdf
from .utils.io import ensure_dir

WHITE = 255


def _ink_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of all ink on the page."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rotation(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Rotate by a random small angle (scanned page placed slightly crooked).
    The canvas grows so no corner of the page is cut off."""
    angle = rng.choice([-1, 1]) * rng.uniform(0.5, 3.0)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    out = cv2.warpAffine(img, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(WHITE, WHITE, WHITE))
    return out, {"angle_deg": round(angle, 2)}


def crop(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Randomly trim each side of the paper margin, never into the text."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _ink_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    keep = 10  # px of white kept around the ink
    left = rng.randint(0, max(x0 - keep, 0))
    top = rng.randint(0, max(y0 - keep, 0))
    right = rng.randint(0, max(w - x1 - keep, 0))
    bottom = rng.randint(0, max(h - y1 - keep, 0))
    out = img[top : h - bottom, left : w - right]
    return out, {"left": left, "top": top, "right": right, "bottom": bottom}


def translation(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Shift the page by up to 3% of its size (not further than the margin, so no text is lost)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _ink_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    max_dx, max_dy = int(0.03 * w), int(0.03 * h)
    dx = rng.randint(-min(max_dx, x0), min(max_dx, w - x1))
    dy = rng.randint(-min(max_dy, y0), min(max_dy, h - y1))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, matrix, (w, h), borderValue=(WHITE, WHITE, WHITE))
    return out, {"dx": dx, "dy": dy}


def brightness_contrast(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Random contrast (x0.75-1.25 around mid-gray) and brightness (+-25% of mid-gray)."""
    contrast = rng.uniform(0.75, 1.25)
    brightness = rng.uniform(-0.25, 0.25) * 128
    out = (img.astype(np.float32) - 128) * contrast + 128 + brightness
    return np.clip(out, 0, 255).astype(np.uint8), {"contrast": round(contrast, 3), "brightness": round(brightness, 1)}


def noise(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Gaussian sensor/scan noise, light to medium."""
    sigma = rng.uniform(5, 20)
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    gray_noise = noise_rng.normal(0, sigma, img.shape[:2]).astype(np.float32)[..., None]  # same on all channels
    out = img.astype(np.float32) + gray_noise
    return np.clip(out, 0, 255).astype(np.uint8), {"sigma": round(sigma, 2)}


def dropout(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Faded print: a few % of pixels and many tiny patches turned white."""
    h, w = img.shape[:2]
    out = img.copy()
    rate = rng.uniform(0.01, 0.03)
    np_rng = np.random.default_rng(rng.randrange(2**32))
    out[np_rng.random((h, w)) < rate] = WHITE
    patches = rng.randint(50, 150)
    for _ in range(patches):
        size = rng.randint(4, 12)
        x, y = rng.randrange(0, w - size), rng.randrange(0, h - size)
        out[y : y + size, x : x + size] = WHITE
    return out, {"pixel_rate": round(rate, 4), "patches": patches}


AUGMENTATIONS = {
    "rotation": rotation,
    "crop": crop,
    "translation": translation,
    "brightness_contrast": brightness_contrast,
    "noise": noise,
    "dropout": dropout,
}


def _load_page(path: Path, dpi: int) -> np.ndarray:
    if is_pdf(path):
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path), dpi=dpi)
        if len(pages) != 1:
            raise ValueError(f"{path.name} has {len(pages)} pages; only single-page documents are supported")
        return cv2.cvtColor(np.array(pages[0].convert("RGB")), cv2.COLOR_RGB2BGR)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def augment_dataset(
    input_dir: Path,
    ground_truth_dir: Path,
    output_dir: Path,
    augmentations: list[str],
    seed: int = 42,
    dpi: int = 300,
    include_original: bool = True,
) -> Path:
    """Write <output_dir>/images, <output_dir>/ground_truth and <output_dir>/manifest.csv."""
    image_dir = ensure_dir(output_dir / "images")
    label_dir = ensure_dir(output_dir / "ground_truth")
    variants = (["original"] if include_original else []) + augmentations

    rows = []
    docs = sorted(p for p in input_dir.iterdir() if is_pdf(p) or is_image(p))
    for doc_no, doc in enumerate(docs, start=1):
        gt = find_ground_truth(ground_truth_dir, doc.stem)
        if gt is None:
            print(f"[yellow]skip[/yellow] {doc.name}: no ground truth in {ground_truth_dir}")
            continue
        lang_suffix = gt.stem.rsplit("_", 1)[-1]
        page = _load_page(doc, dpi)
        print(f"({doc_no}/{len(docs)}) {doc.name}")
        for variant in variants:
            # Seed per file and augmentation: re-running gives the same images.
            rng = random.Random(seed * 1_000_003 + zlib.crc32(f"{doc.stem}:{variant}".encode()))
            image, params = (page, {}) if variant == "original" else AUGMENTATIONS[variant](page, rng)
            name = f"{doc.stem}_{variant}"
            cv2.imwrite(str(image_dir / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label = label_dir / f"Json_{name}_{lang_suffix}.json"
            shutil.copyfile(gt, label)
            rows.append({
                "image": f"images/{name}.jpg",
                "ground_truth": f"ground_truth/{label.name}",
                "source": str(doc),
                "source_ground_truth": str(gt),
                "augmentation": variant,
                "params": json.dumps(params),
                "width": image.shape[1],
                "height": image.shape[0],
                "seed": seed,
            })

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[green]Augmentation done[/green]: {len(rows)} images in {image_dir}, labels in {label_dir}, {manifest}")
    return manifest
