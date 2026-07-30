import random
from pathlib import Path
import cv2
import numpy as np
from .document_loader import load_document_pages
from .preprocessing import read_image, rotate_bound
from .utils.io import ensure_dir, save_json

# Each entry generates one augmented variant. Ranges are kept mild so the
# document content stays fully legible and in-frame -- only the visual
# quality/geometry is perturbed (simulates a rotated, skewed, blurry, or
# noisy scan), never a crop or distortion that would alter the content.
AUGMENTATION_KINDS = ["rotate", "skew", "blur", "noise", "combined"]


def _rotate(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    angle = rng.uniform(1.5, 6.0) * rng.choice([-1, 1])
    return rotate_bound(image, angle), {"angle_deg": round(angle, 2)}


def _skew(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    shift = rng.uniform(0.04, 0.08)
    dx, dy = shift * w, shift * h
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), h - rng.uniform(0, dy)],
        [rng.uniform(0, dx), h - rng.uniform(0, dy)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped, {"max_corner_shift_ratio": round(shift, 3)}


def _blur(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    ksize = rng.choice([9, 11, 13])
    sigma = rng.uniform(3.0, 5.0)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return blurred, {"kernel_size": ksize, "sigma": round(sigma, 2)}


def _noise(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    sigma = rng.uniform(8, 18)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, {"gaussian_sigma": round(sigma, 2)}


def _combined(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    rotated, rot_params = _rotate(image, rng)
    blurred, blur_params = _blur(rotated, rng)
    noisy, noise_params = _noise(blurred, rng)
    return noisy, {"rotate": rot_params, "blur": blur_params, "noise": noise_params}


_AUGMENTERS = {
    "rotate": _rotate,
    "skew": _skew,
    "blur": _blur,
    "noise": _noise,
    "combined": _combined,
}


def generate_variants(image: np.ndarray, seed: int | None = None) -> list[tuple[str, np.ndarray, dict]]:
    rng = random.Random(seed)
    return [(kind, *_AUGMENTERS[kind](image, rng)) for kind in AUGMENTATION_KINDS]


def augment_document(
    input_path: str | Path,
    output_dir: str | Path,
    ground_truth_dir: str | Path | None = None,
    dpi: int = 300,
    seed: int | None = None,
) -> dict:
    input_path = Path(input_path)
    output_dir = ensure_dir(Path(output_dir) / input_path.stem)
    render_dir = ensure_dir(output_dir / "_source_pages")

    page_paths = load_document_pages(input_path, render_dir, dpi=dpi)

    ground_truth_path = None
    if ground_truth_dir is not None:
        candidate = Path(ground_truth_dir) / f"Json_{input_path.stem}_th.json"
        if candidate.exists():
            ground_truth_path = str(candidate)

    manifest = {"source": str(input_path), "ground_truth": ground_truth_path, "pages": []}

    for page_no, page_path in enumerate(page_paths, start=1):
        image = read_image(page_path)
        page_entry = {"page": page_no, "original": str(page_path), "variants": []}
        for kind, variant_image, params in generate_variants(image, seed=seed):
            out_path = output_dir / f"page_{page_no:03d}_{kind}.jpg"
            cv2.imwrite(str(out_path), variant_image)
            page_entry["variants"].append({"kind": kind, "file": str(out_path), "params": params})
        manifest["pages"].append(page_entry)

    save_json(manifest, output_dir / "manifest.json")
    return manifest
