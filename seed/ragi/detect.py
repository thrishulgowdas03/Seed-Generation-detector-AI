import cv2
import numpy as np
import json
from pathlib import Path
import argparse

# ============================================================
# RAGI OPEN-CV FINAL DETECTOR
# Designed for tiny brown/red Ragi seed bodies on white paper.
# No ML, no internet, no manual review.
# ============================================================

# Paper region used by the supplied Ragi photo setup.
PAPER_X1 = 0.095
PAPER_X2 = 0.930
PAPER_Y1 = 0.030
PAPER_Y2 = 0.920

# Blob detection: seeds are small compact reddish/brown blobs.
MIN_BLOB_AREA = 6
MAX_BLOB_AREA = 80
MIN_CIRCULARITY = 0.25
MIN_CONVEXITY = 0.70
MIN_INERTIA = 0.30

# Seed colour verification.
MIN_SATURATION = 60
MAX_BRIGHTNESS = 180
MIN_RED_GREEN = 8
MIN_RED_BLUE = 6

# Prevent duplicate detections that can occur very close together.
MIN_CENTER_DISTANCE = 5.0


def paper_crop(img):
    h, w = img.shape[:2]
    x1 = int(PAPER_X1 * w)
    x2 = int(PAPER_X2 * w)
    y1 = int(PAPER_Y1 * h)
    y2 = int(PAPER_Y2 * h)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def make_blob_image(paper):
    b, g, r = cv2.split(paper)
    redness = r.astype(np.int16) - (
        (g.astype(np.int16) + b.astype(np.int16)) // 2
    )

    # Positive red/brown chroma becomes dark in the blob-detector image.
    score = np.clip(redness, 0, 80).astype(np.uint8)
    return 255 - score


def blob_keypoints(blob_image):
    p = cv2.SimpleBlobDetector_Params()

    p.minThreshold = 175
    p.maxThreshold = 255
    p.thresholdStep = 5

    p.filterByArea = True
    p.minArea = MIN_BLOB_AREA
    p.maxArea = MAX_BLOB_AREA

    p.filterByCircularity = True
    p.minCircularity = MIN_CIRCULARITY

    p.filterByConvexity = True
    p.minConvexity = MIN_CONVEXITY

    p.filterByInertia = True
    p.minInertiaRatio = MIN_INERTIA

    p.filterByColor = True
    p.blobColor = 0

    detector = cv2.SimpleBlobDetector_create(p)
    return detector.detect(blob_image)


def colour_ok(paper, hsv, kp):
    x = int(round(kp.pt[0]))
    y = int(round(kp.pt[1]))

    r = 3
    y0, y1 = max(0, y-r), min(paper.shape[0], y+r+1)
    x0, x1 = max(0, x-r), min(paper.shape[1], x+r+1)

    patch = paper[y0:y1, x0:x1]
    hp = hsv[y0:y1, x0:x1]

    if patch.size == 0:
        return False

    mean_s = float(np.mean(hp[:, :, 1]))
    mean_v = float(np.mean(hp[:, :, 2]))

    mean_b = float(np.mean(patch[:, :, 0]))
    mean_g = float(np.mean(patch[:, :, 1]))
    mean_r = float(np.mean(patch[:, :, 2]))

    rg = mean_r - mean_g
    rb = mean_r - mean_b

    return (
        mean_s >= MIN_SATURATION
        and mean_v <= MAX_BRIGHTNESS
        and rg >= MIN_RED_GREEN
        and rb >= MIN_RED_BLUE
    )


def deduplicate(points):
    # Keep stronger/larger blob first, then suppress points that are
    # effectively the same seed.
    points = sorted(
        points,
        key=lambda z: (z["response"], z["size"]),
        reverse=True
    )

    kept = []
    min_d2 = MIN_CENTER_DISTANCE ** 2

    for p in points:
        ok = True
        for q in kept:
            dx = p["x"] - q["x"]
            dy = p["y"] - q["y"]
            if dx * dx + dy * dy < min_d2:
                ok = False
                break
        if ok:
            kept.append(p)

    kept.sort(key=lambda z: (z["y"], z["x"]))

    for i, p in enumerate(kept, 1):
        p["id"] = i

    return kept


def detect(img):
    paper, (ox, oy, ox2, oy2) = paper_crop(img)
    hsv = cv2.cvtColor(paper, cv2.COLOR_BGR2HSV)
    blob_img = make_blob_image(paper)
    keypoints = blob_keypoints(blob_img)

    candidates = []

    for kp in keypoints:
        if not colour_ok(paper, hsv, kp):
            continue

        x = float(kp.pt[0])
        y = float(kp.pt[1])
        size = float(kp.size)

        # Convert keypoint size into a small visualization box.
        half = max(2.0, min(7.0, size * 0.42))

        candidates.append({
            "x": round(x + ox, 2),
            "y": round(y + oy, 2),
            "size": round(size, 2),
            "response": round(float(kp.response), 4)
        })

    candidates = deduplicate(candidates)
    return candidates, (ox, oy, ox2, oy2)


def annotate(img, candidates):
    out = img.copy()

    for c in candidates:
        x = int(round(c["x"]))
        y = int(round(c["y"]))

        cv2.circle(out, (x, y), 5, (255, 0, 0), 1)
        cv2.putText(
            out,
            str(c["id"]),
            (x + 5, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return out


def process(path, out_dir):
    img = cv2.imread(str(path))
    if img is None:
        print(f"ERROR: Could not read {path}")
        return

    candidates, roi = detect(img)
    annotated = annotate(img, candidates)

    out_img = out_dir / f"{path.stem}_opencv_final.jpg"
    out_json = out_dir / f"{path.stem}_opencv_final.json"

    cv2.imwrite(str(out_img), annotated)

    data = {
        "image": path.name,
        "detector": "Ragi OpenCV Final",
        "automatic_seed_count": len(candidates),
        "paper_roi": {
            "x1": roi[0],
            "y1": roi[1],
            "x2": roi[2],
            "y2": roi[3]
        },
        "candidates": candidates,
        "manual_review": False
    }

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{path.name}: {len(candidates)} seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=".",
        help="Image file or folder containing Ragi JPG/JPEG/PNG images"
    )
    ap.add_argument(
        "--output",
        default="opencv_results",
        help="Output folder"
    )
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if inp.is_file():
        files = [inp]
    else:
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(inp.glob(ext))
        files = sorted(set(files))

    if not files:
        print("No images found.")
        return

    for f in files:
        process(f, out_dir)

    print(f"\nResults saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
