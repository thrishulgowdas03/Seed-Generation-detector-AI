"""
Seed and shoot detection via HSV color thresholding.

Tuned and validated against a real bh_maize paper-towel-roll sample:
- Seeds are red/orange (fungicide-treated kernels) -> easy, reliable color signal
- Shoots (coleoptiles) are yellow-green -> easy, reliable color signal
- Roots are white/cream, nearly the same tone as the paper towel -> NOT
  reliably color-detectable, deliberately not attempted here. Root
  presence is instead inferred indirectly (see classify_pipeline.py).

If your tray/lighting looks different from the reference sample, re-run
the HSV tuning cell-by-cell (see README "Step 1: tune detection") before
trusting these defaults blindly.
"""

import cv2
import numpy as np

# --- Seed (red/orange) HSV range ---
# Red wraps around 0/180 in OpenCV's HSV, so two ranges are OR'd together.
SEED_LOWER_RED_1 = np.array([0, 100, 80])
SEED_UPPER_RED_1 = np.array([15, 255, 255])
SEED_LOWER_RED_2 = np.array([165, 100, 80])
SEED_UPPER_RED_2 = np.array([180, 255, 255])

# --- Shoot (yellow-green) HSV range ---
SHOOT_LOWER = np.array([25, 40, 60])
SHOOT_UPPER = np.array([55, 255, 255])

SEED_MIN_AREA_FRAC = 0.00008   # as a fraction of image area, scale-independent
SHOOT_MIN_AREA_FRAC = 0.000015


def get_seed_mask(hsv: np.ndarray) -> np.ndarray:
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, SEED_LOWER_RED_1, SEED_UPPER_RED_1),
        cv2.inRange(hsv, SEED_LOWER_RED_2, SEED_UPPER_RED_2),
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def get_shoot_mask(hsv: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(hsv, SHOOT_LOWER, SHOOT_UPPER)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def detect_seeds(image_bgr: np.ndarray):
    """Returns list of dicts: {cx, cy, w, h, area}"""
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = img_area * SEED_MIN_AREA_FRAC

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = get_seed_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seeds = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        seeds.append({"cx": x + w // 2, "cy": y + h // 2, "w": w, "h": h, "area": area})
    return seeds


def get_clean_shoot_mask(image_bgr: np.ndarray, min_area_frac: float = SHOOT_MIN_AREA_FRAC):
    """Shoot mask with small noise specks removed (kept as filled blobs,
    ready for skeletonization)."""
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = img_area * min_area_frac

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = get_shoot_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean
