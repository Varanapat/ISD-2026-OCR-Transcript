"""Lab 8A part 1 -- controlled degradation.

Five fixed levels simulating how a transcript decays between a digital original and a
third-generation office photocopy. Every parameter is a constant: no RNG anywhere, so
the same input always yields byte-identical output and results are comparable between
machines, which is what makes the sweep in part 2 worth reading.
"""
from pathlib import Path
import cv2
import numpy as np
from .document_loader import load_document_pages
from .preprocessing import read_image, rotate_bound
from .utils.io import ensure_dir, save_json

# angle_deg / scale follow the table in the lab sheet; the remaining flags pick which
# artefacts each level adds on top.
NOISE_LEVELS: dict[str, dict] = {
    "L0_clean":       {"angle_deg": 0.0, "scale": 1.00, "simulates": "digital original"},
    "L1_light":       {"angle_deg": 0.4, "scale": 1.00, "simulates": "good scanner, flat paper",
                       "ink_bleed": 1, "jpeg_quality": 82},
    "L2_watermark":   {"angle_deg": 1.1, "scale": 0.85, "simulates": "phone photo with a seal",
                       "ink_bleed": 1, "jpeg_quality": 78, "seal": True},
    "L3_titled_copy": {"angle_deg": 2.3, "scale": 0.65, "simulates": "repeatedly re-copied",
                       "ink_bleed": 2, "jpeg_quality": 70, "copy_text": True},
    "L4_rescan":      {"angle_deg": 3.8, "scale": 0.50, "simulates": "office copier rescan",
                       "ink_bleed": 2, "jpeg_quality": 62, "seal": True, "copy_text": True,
                       "darken": 0.72},
}


def _ink_bleed(image: np.ndarray, strength: int) -> np.ndarray:
    """Let dark strokes spread into the paper, the way wet toner does.

    Dilating on an inverted image thickens the glyphs; blending the result back at a
    fraction keeps the edges soft instead of simply making the text bolder.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strength * 2 + 1,) * 2)
    spread = cv2.erode(image, kernel)
    softened = cv2.GaussianBlur(spread, (3, 3), 0)
    return cv2.addWeighted(image, 0.65, softened, 0.35, 0)


def _jpeg_artifact(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _seal(image: np.ndarray) -> np.ndarray:
    """A faint round institutional stamp, drawn from circles and radial ticks only.

    Deliberately font-free: rendering Thai text needs a font file that may not exist on
    the marker's machine, and a missing glyph would silently change the image.
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    centre = (int(w * 0.72), int(h * 0.80))
    radius = int(min(h, w) * 0.11)
    colour = (120, 90, 90)
    cv2.circle(overlay, centre, radius, colour, 6)
    cv2.circle(overlay, centre, int(radius * 0.80), colour, 3)
    for deg in range(0, 360, 15):
        rad = np.deg2rad(deg)
        p1 = (int(centre[0] + np.cos(rad) * radius * 0.82),
              int(centre[1] + np.sin(rad) * radius * 0.82))
        p2 = (int(centre[0] + np.cos(rad) * radius * 0.96),
              int(centre[1] + np.sin(rad) * radius * 0.96))
        cv2.line(overlay, p1, p2, colour, 2)
    return cv2.addWeighted(image, 0.82, overlay, 0.18, 0)


def _copy_text(image: np.ndarray) -> np.ndarray:
    """Large diagonal COPY across the page. ASCII only, so cv2's built-in font suffices."""
    overlay = image.copy()
    h, w = image.shape[:2]
    scale = w / 260.0
    text = "COPY"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, int(scale * 3))
    canvas = np.zeros_like(image)
    cv2.putText(canvas, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (128, 128, 128), int(scale * 3), cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (w, h))
    mask = canvas.any(axis=2)
    overlay[mask] = canvas[mask]
    return cv2.addWeighted(image, 0.88, overlay, 0.12, 0)


def apply_level(image: np.ndarray, level: str) -> tuple[np.ndarray, dict]:
    """Apply one named level. Step order matters and mirrors physical reality:
    the page is marked, then skewed and shrunk by the capture device, then the file
    format degrades it last."""
    if level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level: {level}. Known: {', '.join(NOISE_LEVELS)}")
    spec = NOISE_LEVELS[level]
    out = image
    source_width = image.shape[1]

    if spec.get("ink_bleed"):
        out = _ink_bleed(out, spec["ink_bleed"])
    if spec.get("seal"):
        out = _seal(out)
    if spec.get("copy_text"):
        out = _copy_text(out)
    if spec["angle_deg"]:
        out = rotate_bound(out, spec["angle_deg"])
    if out.shape[1] != round(source_width * spec["scale"]):
        # Scale is defined against the ORIGINAL page width, not the rotated canvas:
        # rotate_bound grows the frame to fit the corners, so scaling by the raw factor
        # would leave L4_rescan at 55% of the source instead of the 50% the table states.
        factor = (source_width * spec["scale"]) / out.shape[1]
        out = cv2.resize(out, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    if spec.get("darken"):
        out = np.clip(out.astype(np.float32) * spec["darken"], 0, 255).astype(np.uint8)
    if spec.get("jpeg_quality"):
        out = _jpeg_artifact(out, spec["jpeg_quality"])
    return out, dict(spec)


def generate_noisy_document(
    input_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    levels: list[str] | None = None,
) -> dict:
    """Render a document and write one image per level under <output_dir>/<level>/."""
    input_path = Path(input_path)
    output_dir = ensure_dir(output_dir)
    levels = levels or list(NOISE_LEVELS)

    page_paths = load_document_pages(input_path, ensure_dir(output_dir / "_source_pages"), dpi=dpi)
    manifest = {"source": str(input_path), "dpi": dpi, "levels": {}}

    for level in levels:
        level_dir = ensure_dir(output_dir / level)
        entries = []
        for page_no, page_path in enumerate(page_paths, start=1):
            out_image, params = apply_level(read_image(page_path), level)
            out_path = level_dir / f"page_{page_no:02d}.png"
            cv2.imwrite(str(out_path), out_image)
            entries.append({"page": page_no, "file": str(out_path),
                            "size": [out_image.shape[1], out_image.shape[0]]})
        manifest["levels"][level] = {"params": params, "pages": entries}

    save_json(manifest, output_dir / "manifest.json")
    return manifest
