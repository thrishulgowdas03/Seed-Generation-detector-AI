import cv2
import numpy as np

# Ragi 3-class growth classifier.
# Tuned for the supplied Ragi paper images and the OpenCV seed centers.
# The seed body is detected first; this stage looks locally around that
# center for visible green shoot / radicle evidence.

GREEN_H_MIN, GREEN_H_MAX = 25, 100
GREEN_S_MIN, GREEN_V_MIN = 45, 55
ROOT_S_MIN, ROOT_V_MAX, ROOT_DARKNESS = 18, 185, 22
MIN_COMPONENT_AREA = 3

# More selective than the earlier V9 thresholds to avoid treating the
# paper texture in the wider crop as germination evidence.
GERMI_GREEN_PIXELS = 5
GERMI_LENGTH = 16.0
SEMI_GREEN_PIXELS = 1
SEMI_LENGTH = 10.0
PATCH_RADIUS = 32


def extract_patch(image, cx, cy, radius=PATCH_RADIUS):
    h, w = image.shape[:2]
    x1, y1 = max(0, int(cx)-radius), max(0, int(cy)-radius)
    x2, y2 = min(w, int(cx)+radius+1), min(h, int(cy)+radius+1)
    return image[y1:y2, x1:x2]


def detect_growth(patch):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    green_mask = ((h >= GREEN_H_MIN) & (h <= GREEN_H_MAX) &
                  (s >= GREEN_S_MIN) & (v >= GREEN_V_MIN))

    background = cv2.GaussianBlur(gray, (21, 21), 0)
    local_dark = background.astype(np.float32) - gray.astype(np.float32)
    root_mask = ((local_dark >= ROOT_DARKNESS) &
                 (s >= ROOT_S_MIN) & (v <= ROOT_V_MAX))

    growth = (green_mask | root_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    growth = cv2.morphologyEx(growth, cv2.MORPH_OPEN, kernel)
    growth = cv2.morphologyEx(growth, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(growth, 8)
    clean = np.zeros_like(growth)
    lengths = []
    total = 0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if max(w, h) <= 3 and area < 6:
            continue
        clean[labels == label] = 255
        total += area
        lengths.append(float(max(w, h)))

    green_pixels = int(np.count_nonzero(green_mask & (clean > 0)))
    root_pixels = int(np.count_nonzero(root_mask & (clean > 0)))
    max_length = max(lengths) if lengths else 0.0
    return clean, total, max_length, green_pixels, root_pixels


def classify_one(image, cx, cy):
    patch = extract_patch(image, cx, cy)
    _, pixels, length, green, root = detect_growth(patch)

    if green >= GERMI_GREEN_PIXELS or length >= GERMI_LENGTH:
        cls = "GERMI"
    elif green >= SEMI_GREEN_PIXELS or length >= SEMI_LENGTH:
        cls = "SEMI GERMI"
    else:
        cls = "NON GERMI"

    return cls, {
        "growth_pixels": int(pixels),
        "max_growth_length": round(float(length), 1),
        "green_pixels": int(green),
        "root_pixels": int(root),
        "patch_radius": PATCH_RADIUS,
    }
