"""
FINAL PIPELINE for the real bh_maize paper-towel-roll images.

Approach (in order):
1. Detect seeds via red/orange color (reliable).
2. Detect shoots via yellow-green color (reliable), skeletonize into
   centerlines to handle heavy tangling between neighboring shoots.
3. For each seed, find its nearest skeleton endpoint (candidate shoot
   tip/base). If close enough (gap_multiplier * seed size), the seed
   is classified GERMINATED -- this handles the majority of cases
   using pure geometry, no model calls needed.
4. For seeds with no nearby shoot tip, fall back to CLIP zero-shot on
   a wider crop to decide SEMI_GERMINATED vs NON_GERMINATED (this is
   the harder case: distinguishing a short white root from bare towel).

Known limitation (state this explicitly in the pitch): root color is
too close to the paper towel to detect directly, and heavy tangling
means shoot-to-seed attribution is a geometric approximation, not
perfect tracing. This is a deliberate, documented scoping decision
given hackathon time constraints.

Usage:
    python infer_final.py --image ../../sample/sample_tray.jpg
"""

import argparse
import json
import os
import sys
import cv2

sys.path.append(os.path.dirname(__file__))
from detect_seeds_shoots import detect_seeds, get_clean_shoot_mask
from skeleton_graph import build_skeleton_graph, match_seeds_to_shoots
from tie_break_classify import TieBreakClassifier, extract_crop

CLASS_COLORS = {
    "germinated": (0, 200, 0),
    "semi_germinated": (0, 200, 255),
    "non_germinated": (0, 0, 200),
}


def run(image_path: str, out_dir: str = "output", gap_multiplier: float = 2.0,
        downscale_width: int = 1400):
    os.makedirs(out_dir, exist_ok=True)

    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Downscale for detection speed; classification crops are taken
    # from this same working resolution for consistency.
    h, w = orig.shape[:2]
    scale = min(1.0, downscale_width / w)
    img = cv2.resize(orig, (int(w * scale), int(h * scale))) if scale < 1.0 else orig.copy()

    print("Detecting seeds and shoots...")
    seeds = detect_seeds(img)
    shoot_mask = get_clean_shoot_mask(img)
    skel, G, degrees, endpoints = build_skeleton_graph(shoot_mask)
    matches = match_seeds_to_shoots(seeds, endpoints, G, degrees, gap_multiplier)

    print(f"Detected {len(seeds)} seeds. Loading CLIP for tie-break cases...")
    tie_breaker = TieBreakClassifier()

    predictions = []
    annotated = img.copy()
    counts = {"germinated": 0, "semi_germinated": 0, "non_germinated": 0}

    for seed, match in zip(seeds, matches):
        if match["shoot_found"]:
            cls_name = "germinated"
            confidence = None  # geometric decision, not a probability
            detail = {"gap_to_endpoint": match["gap_to_endpoint"],
                       "shoot_path_length": match.get("shoot_path_length"),
                       "termination": match.get("termination")}
        else:
            crop = extract_crop(img, seed)
            cls_name, confidence = tie_breaker.classify_crop(crop)
            detail = {"gap_to_endpoint": match.get("gap_to_endpoint"), "method": "clip_tie_break"}

        counts[cls_name] += 1
        predictions.append({
            "class_name": cls_name,
            "confidence": confidence,
            "bbox_xywh": [seed["cx"] - seed["w"] // 2, seed["cy"] - seed["h"] // 2,
                          seed["w"], seed["h"]],
            "detail": detail,
        })

        color = CLASS_COLORS[cls_name]
        x, y, bw, bh = predictions[-1]["bbox_xywh"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        label = cls_name[:4] if confidence is None else f"{cls_name[:4]} {confidence:.2f}"
        cv2.putText(annotated, label, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    total = sum(counts.values())
    semi_weight = 0.5
    germination_rate = (
        (counts["germinated"] + semi_weight * counts["semi_germinated"]) / total
        if total else 0.0
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(out_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    output = {
        "image": image_path,
        "annotated_image": annotated_path,
        "predictions": predictions,
        "analytics": {
            "counts": counts,
            "total_seeds": total,
            "germination_rate": round(germination_rate, 4),
            "semi_weight_used": semi_weight,
        },
        "known_limitations": (
            "Root color is visually similar to the paper towel and is not "
            "directly detected. Shoot-to-seed attribution under heavy "
            "tangling is a nearest-endpoint geometric approximation, not "
            "exact tracing. Ambiguous cases are resolved via CLIP zero-shot."
        ),
    }

    json_path = os.path.join(out_dir, f"{base_name}_result.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output["analytics"], indent=2))
    print(f"\nAnnotated image: {annotated_path}")
    print(f"JSON result: {json_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--gap_multiplier", type=float, default=2.0,
                         help="how close a shoot tip must be (x seed size) to count as this seed's own shoot")
    parser.add_argument("--downscale_width", type=int, default=1400,
                         help="working resolution; raise if small seeds are missed, lower for speed")
    args = parser.parse_args()

    run(args.image, args.out_dir, args.gap_multiplier, args.downscale_width)
