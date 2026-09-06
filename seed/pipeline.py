from pathlib import Path
import cv2
from functools import lru_cache


# ================================================================
# AUTOMATIC 3-CROP DETECTION
#   Maize -> proven Maize detector
#   Ragi  -> OpenCV Ragi detector
#   Paddy -> OpenCV Paddy detector
#
# The detectors have very different signatures, so crop detection is
# based on their candidate counts rather than forcing one detector to
# identify every crop.
# ================================================================

def detect_crop(image):
    from maize.detect_seeds_shoots import detect_seeds, get_clean_shoot_mask
    from maize.skeleton_graph import build_skeleton_graph, match_seeds_to_shoots
    from ragi.detect import detect as detect_ragi
    from paddy.detector import detect_paddy_seeds

    maize_candidates = detect_seeds(image)
    ragi_candidates, _ = detect_ragi(image)
    paddy_candidates = detect_paddy_seeds(image)

    maize_n = len(maize_candidates)
    ragi_n = len(ragi_candidates)
    paddy_n = len(paddy_candidates)

    # A Maize image normally has many seed/shoot associations. The
    # Maize detector alone can also find brown Paddy objects, so use the
    # proven skeleton-graph association as the Maize-specific signature.
    maize_matched = 0
    maize_match_ratio = 0.0
    if maize_n:
        shoot_mask = get_clean_shoot_mask(image)
        _, graph, degrees, endpoints = build_skeleton_graph(shoot_mask)
        matches = match_seeds_to_shoots(
            maize_candidates, endpoints, graph, degrees, 2.0
        )
        maize_matched = sum(
            bool(m.get('shoot_found')) for m in matches
        )
        maize_match_ratio = maize_matched / maize_n

    if maize_n >= 15 and maize_match_ratio >= 0.65:
        crop = 'Maize'
    else:
        # On the supplied raw tray images, Paddy produces substantially
        # more Paddy-style candidates than Ragi. The observed separation
        # is roughly >=0.84 for Paddy and <=0.50 for Ragi.
        paddy_to_ragi = paddy_n / max(ragi_n, 1)
        crop = 'Paddy' if paddy_to_ragi >= 0.65 else 'Ragi'

    return crop, {
        'maize_candidates': maize_n,
        'maize_shoot_matches': maize_matched,
        'maize_shoot_match_ratio': round(maize_match_ratio, 3),
        'ragi_candidates': ragi_n,
        'paddy_candidates': paddy_n,
        'paddy_to_ragi_ratio': round(paddy_n / max(ragi_n, 1), 3),
    }


@lru_cache(maxsize=1)
def get_ragi_clip():
    from ragi.clip_classify import RagiCLIPClassifier
    return RagiCLIPClassifier(crop_radius=42, upscale=4)


def run_ragi(image):
    from ragi.detect import detect
    candidates, roi = detect(image)

    classifier = get_ragi_clip()
    centers = [(c['x'], c['y']) for c in candidates]
    clip_results, patch_boxes = classifier.classify(image, centers) if centers else ([], [])

    annotated = image.copy()
    counts = {'GERMI': 0, 'SEMI GERMI': 0, 'NON GERMI': 0}
    predictions = []
    colors = {
        'GERMI': (0, 180, 0),
        'SEMI GERMI': (0, 180, 255),
        'NON GERMI': (0, 0, 200),
    }

    for c, (cls, conf, scores), patch_box in zip(candidates, clip_results, patch_boxes):
        counts[cls] += 1
        predictions.append({
            'id': c['id'],
            'class_name': cls,
            'confidence': conf,
            'center': [c['x'], c['y']],
            'size': c['size'],
            'detail': {
                'method': 'clip_3class',
                'class_scores': scores,
                'patch_box': {
                    'x1': patch_box[0], 'y1': patch_box[1],
                    'x2': patch_box[2], 'y2': patch_box[3],
                },
            },
        })
        x, y = int(c['x']), int(c['y'])
        cv2.circle(annotated, (x, y), 7, colors[cls], 2)
        cv2.putText(
            annotated, f'{c["id"]}:{cls[:4]} {conf:.2f}',
            (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
            0.34, colors[cls], 1, cv2.LINE_AA,
        )

    total = len(predictions)
    germinated = counts['GERMI'] + counts['SEMI GERMI']
    rate = germinated / total if total else 0.0
    weighted = (counts['GERMI'] + 0.5 * counts['SEMI GERMI']) / total if total else 0.0

    return annotated, {
        'crop': 'Ragi',
        'total_seeds': total,
        'counts': counts,
        'germinated_total': germinated,
        'germination_rate': round(rate, 4),
        'weighted_germination_rate': round(weighted, 4),
        'paper_roi': dict(zip(['x1', 'y1', 'x2', 'y2'], roi)),
        'predictions': predictions,
        'method': 'OpenCV Ragi seed-body detection + CLIP 3-class germination classification',
        'classifier': 'openai/clip-vit-base-patch32',
        'manual_review': False,
    }


def run_paddy(image, image_path, out_dir):
    from paddy.detector import detect_paddy_seeds, annotate_paddy
    from paddy.clip_classify import classify_paddy

    candidates = detect_paddy_seeds(image)
    predictions = classify_paddy(image, candidates)

    counts = {'GERMI': 0, 'SEMI GERMI': 0, 'NON GERMI': 0}
    for p in predictions:
        counts[p['class']] += 1

    total = len(predictions)
    germinated = counts['GERMI'] + counts['SEMI GERMI']
    strict_rate = germinated / total if total else 0.0
    weighted_rate = (counts['GERMI'] + 0.5 * counts['SEMI GERMI']) / total if total else 0.0

    annotated = annotate_paddy(image, predictions)
    out_dir = Path(out_dir)
    paddy_dir = out_dir / 'paddy'
    paddy_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    annotated_path = paddy_dir / f'{stem}_paddy_annotated.jpg'

    cv2.imwrite(str(annotated_path), annotated)

    # Keep the dashboard schema consistent across all three crops.
    return annotated, {
        'crop': 'Paddy',
        'image': Path(image_path).name,
        'annotated_image': str(annotated_path),
        'method': 'OpenCV seed detection + Paddy-tailored CLIP 3-class germination classification',
        'counts': counts,
        'total_seeds': total,
        'germinated_total': germinated,
        'germination_rate': round(strict_rate, 4),
        'germination_rate_strict_percent': round(strict_rate * 100, 2),
        'weighted_germination_rate': round(weighted_rate, 4),
        'weighted_germination_rate_percent': round(weighted_rate * 100, 2),
        'predictions': predictions,
        'manual_review': False,
    }


def run_maize(image_path, out_dir):
    from maize.infer_dashboard import run

    result = run(str(image_path), out_dir=str(Path(out_dir) / 'maize'))
    annotated = cv2.imread(result['annotated_image'])

    # Normalize both the newer top-level Maize schema and the older
    # nested analytics schema.
    analytics = result.get('analytics', {})
    counts = result.get('counts', analytics.get('counts', {}))
    total = result.get('total_seeds', analytics.get('total_seeds', sum(counts.values())))
    germinated_total = result.get(
        'germinated_total',
        analytics.get(
            'germinated_total',
            counts.get('germinated', 0) + counts.get('semi_germinated', 0),
        ),
    )
    rate = result.get('germination_rate', analytics.get('germination_rate'))
    if rate is None:
        rate = germinated_total / total if total else 0.0

    weighted = result.get(
        'weighted_germination_rate',
        analytics.get('weighted_germination_rate'),
    )

    # Convert Maize's class names to the common dashboard vocabulary.
    common_counts = {
        'GERMI': counts.get('germinated', counts.get('GERMI', 0)),
        'SEMI GERMI': counts.get('semi_germinated', counts.get('SEMI GERMI', 0)),
        'NON GERMI': counts.get('non_germinated', counts.get('NON GERMI', 0)),
    }

    return annotated, {
        'crop': 'Maize',
        'image': result.get('image', str(image_path)),
        'annotated_image': result.get('annotated_image'),
        'counts': common_counts,
        'total_seeds': total,
        'germinated_total': germinated_total,
        'germination_rate': rate,
        'weighted_germination_rate': weighted,
        'predictions': result.get('predictions', []),
        'method': 'HSV seed/shoot detection + skeleton graph + CLIP tie-break',
        'manual_review': False,
    }


def analyze(image_path, out_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop, meta = detect_crop(image)

    if crop == 'Maize':
        annotated, result = run_maize(Path(image_path), out_dir)
    elif crop == 'Paddy':
        annotated, result = run_paddy(image, Path(image_path), out_dir)
    else:
        annotated, result = run_ragi(image)

    result['auto_crop_detection'] = meta
    return annotated, result
