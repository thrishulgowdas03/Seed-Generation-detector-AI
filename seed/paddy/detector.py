
import cv2
import numpy as np

# Paddy seed detector:
# - OpenCV only
# - tuned for the supplied 1536x2048 white-paper images
# - detects compact dark/brown paddy seed bodies, not roots/shoots
# - no manual review required

def _roi_polygon(h, w):
    # Conservative quadrilateral around the paper.  It removes most of the
    # surrounding foil/table while retaining seeds near the paper edges.
    return np.array([
        [int(0.035*w), int(0.045*h)],
        [int(0.950*w), int(0.030*h)],
        [int(0.965*w), int(0.950*h)],
        [int(0.045*w), int(0.975*h)]
    ], dtype=np.int32)

def detect_paddy_seeds(image_bgr):
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [_roi_polygon(h, w)], 255)

    # Dark blobs are the seed-body signal.  The color test below rejects
    # most gray/black mold and paper marks.
    masked = gray.copy()
    masked[roi == 0] = 255

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 450

    params.filterByCircularity = True
    params.minCircularity = 0.15

    params.filterByConvexity = True
    params.minConvexity = 0.55

    params.filterByInertia = True
    params.minInertiaRatio = 0.12

    params.filterByColor = True
    params.blobColor = 0

    params.minThreshold = 30
    params.maxThreshold = 180
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(masked)

    candidates = []
    for kp in keypoints:
        x, y = kp.pt
        ix, iy = int(round(x)), int(round(y))

        if ix < 5 or iy < 5 or ix >= w-5 or iy >= h-5:
            continue

        patch = hsv[iy-4:iy+5, ix-4:ix+5]
        if patch.size == 0:
            continue

        H, S, V = np.median(patch.reshape(-1, 3), axis=0)

        # Brown/orange colored seed OR very dark seed with some saturation.
        # Mold is generally less seed-like in shape and/or color.
        color_ok = (
            (S >= 35 and H <= 35 and V <= 220) or
            (S >= 20 and V <= 90)
        )
        if not color_ok:
            continue

        candidates.append({
            "x": float(x),
            "y": float(y),
            "size": float(kp.size)
        })

    # Remove duplicate detections that arise from nearby threshold levels.
    candidates.sort(key=lambda z: -z["size"])
    kept = []
    min_dist = 10.0
    for c in candidates:
        if all((c["x"]-q["x"])**2 + (c["y"]-q["y"])**2 > min_dist**2 for q in kept):
            kept.append(c)

    # Stable numbering: top-to-bottom, then left-to-right.
    kept.sort(key=lambda z: (z["y"], z["x"]))
    for i, c in enumerate(kept, 1):
        c["id"] = i

    return kept

def annotate_paddy(image_bgr, predictions):
    out = image_bgr.copy()
    colors = {
        "GERMI": (0, 180, 0),
        "SEMI GERMI": (0, 165, 255),
        "NON GERMI": (0, 0, 220),
        "UNKNOWN": (180, 180, 180)
    }

    for p in predictions:
        x, y = int(round(p["x"])), int(round(p["y"]))
        label = p.get("class", "UNKNOWN")
        color = colors.get(label, colors["UNKNOWN"])
        r = max(8, int(round(p.get("size", 16) / 2)))
        cv2.circle(out, (x, y), r, color, 2)
        cv2.putText(
            out, str(p["id"]), (x + r + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA
        )
    return out
