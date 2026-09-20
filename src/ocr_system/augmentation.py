"""
augmentation.py  (วางที่ src/ocr_system/augmentation.py)
--------------------------------------------------------
ฟังก์ชัน augment ภาพเอกสารล้วนๆ ไม่พึ่ง OCR/pipeline
เป็นโมดูลให้ dataset_builder.py และ cli.py import ไปใช้

หลักการ: augment เปลี่ยนแค่หน้าตาภาพ เนื้อหาไม่เปลี่ยน => label เดิมใช้ได้
"""

import random
import cv2
import numpy as np


def aug_rotate(img, max_deg=4.0):
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def aug_perspective(img, strength=0.02):
    h, w = img.shape[:2]
    d = strength * min(h, w)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[random.uniform(-d, d), random.uniform(-d, d)] for _ in range(4)])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def aug_brightness_contrast(img):
    return cv2.convertScaleAbs(img, alpha=random.uniform(0.85, 1.15),
                               beta=random.uniform(-25, 25))


def aug_gaussian_noise(img, sigma_range=(3, 12)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_blur(img):
    k = random.choice([3, 3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_jpeg(img, quality_range=(35, 75)):
    q = random.randint(*quality_range)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def aug_shadow(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
    mask[:, x1:x2] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.1)
    return np.clip(img.astype(np.float32) * (1.0 - 0.25 * mask[:, :, None]), 0, 255).astype(np.uint8)


AUGS = [aug_rotate, aug_perspective, aug_brightness_contrast,
        aug_gaussian_noise, aug_blur, aug_jpeg, aug_shadow]


def augment_once(img: np.ndarray) -> np.ndarray:
    """สุ่มหยิบ 2-4 เทคนิคมาต่อกันเป็นหนึ่งเวอร์ชัน"""
    out = img.copy()
    for fn in random.sample(AUGS, random.randint(2, 4)):
        out = fn(out)
    return out