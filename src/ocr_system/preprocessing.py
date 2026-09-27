from pathlib import Path
import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def resize_if_small(image: np.ndarray, min_width: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    if w >= min_width:
        return image
    scale = min_width / w
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def estimate_skew_angle(gray: np.ndarray) -> float:
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 15:
        return 0.0
    return float(angle)


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_shadow(gray: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Flatten uneven lighting by background division.

    Closing with a kernel much wider than a glyph swallows the text, leaving an
    estimate of the page illumination; dividing the original by that estimate pulls
    shadowed regions back up to the brightness of the rest of the page.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(gray, background, scale=255)


def remove_speckles(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Drop isolated specks from a binarised page.

    Opening erases small *foreground* blobs, and adaptiveThreshold leaves text dark
    on a light page, so the image is inverted first -- text becomes the foreground
    and the stray dark dots are what gets removed. The kernel stays at 2px because
    Thai vowels and tone marks are only a few pixels tall and a 3px kernel eats them.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    inverted = cv2.bitwise_not(binary)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opened)


def clean_image(image: np.ndarray, method: str = "none", deskew: bool = True) -> np.ndarray:
    """Lab 8A cleaning methods.

    none  -- untouched, the baseline every other method is measured against
    light -- deskew, flatten lighting, denoise; keeps the grey levels
    heavy -- light plus binarise and despeckle; pure black and white

    The step order is fixed and must not be reshuffled: deskew comes first because
    every later step assumes text sits on horizontal rows, shadow removal comes
    before denoising so the denoiser is not fighting a brightness gradient, and
    binarisation comes last once the page is evenly lit.
    """
    if method == "none":
        return image
    if method not in ("light", "heavy"):
        raise ValueError(f"Unknown clean method: {method}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if deskew:
        gray = rotate_bound(gray, estimate_skew_angle(gray))
    gray = remove_shadow(gray)
    # fastNlMeansDenoising, not GaussianBlur: a Gaussian blurs the glyph edges too,
    # and those edges are exactly what has to survive.
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    if method == "light":
        return gray

    # adaptiveThreshold, not Otsu: Otsu picks one cut-off for the whole page, so a
    # page that is darker on one side comes out half solid black.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    return remove_speckles(binary)


def save_debug_image(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
