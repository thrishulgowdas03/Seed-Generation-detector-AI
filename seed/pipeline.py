from pathlib import Path
import cv2
from functools import lru_cache


def detect_crop(image):
    from maize.detect_seeds_shoots import detect_seeds
    from ragi.detect import detect as detect_ragi
    maize_seeds = detect_seeds(image)
    ragi_seeds, _ = detect_ragi(image)
    if len(maize_seeds) >= 3:
        return 'Maize', {'maize_candidates': len(maize_seeds), 'ragi_candidates': len(ragi_seeds)}
    return 'Ragi', {'maize_candidates': len(maize_seeds), 'ragi_candidates': len(ragi_seeds)}


@lru_cache(maxsize=1)
def get_ragi_clip():
    from ragi.clip_classify import RagiCLIPClassifier
    return RagiCLIPClassifier(crop_radius=42, upscale=4)


def run_ragi(image):
    from ragi.detect import detect
    candidates, roi = detect(image)

    # Use CLIP for every detected Ragi seed, with the same enlarged-neighborhood
    # idea that made the earlier Ragi V9 crops useful.
    classifier = get_ragi_clip()
    centers = [(c['x'], c['y']) for c in candidates]
    clip_results, patch_boxes = classifier.classify(image, centers) if centers else ([], [])

    annotated = image.copy()
    counts = {'GERMI': 0, 'SEMI GERMI': 0, 'NON GERMI': 0}
    predictions = []
    colors = {'GERMI': (0, 180, 0), 'SEMI GERMI': (0, 180, 255), 'NON GERMI': (0, 0, 200)}

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
                'patch_box': {'x1': patch_box[0], 'y1': patch_box[1], 'x2': patch_box[2], 'y2': patch_box[3]},
            }
        })
        x, y = int(c['x']), int(c['y'])
        cv2.circle(annotated, (x, y), 7, colors[cls], 2)
        cv2.putText(annotated, f'{c["id"]}:{cls[:4]} {conf:.2f}', (x + 5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colors[cls], 1, cv2.LINE_AA)

    total = len(predictions)
    germinated = counts['GERMI'] + counts['SEMI GERMI']
    rate = germinated / total if total else 0.0
    output = {
        'crop': 'Ragi',
        'total_seeds': total,
        'counts': counts,
        'germinated_total': germinated,
        'germination_rate': round(rate, 4),
        'paper_roi': dict(zip(['x1', 'y1', 'x2', 'y2'], roi)),
        'predictions': predictions,
        'method': 'OpenCV Ragi seed-body detection + CLIP 3-class germination classification',
        'classifier': 'openai/clip-vit-base-patch32',
        'clip_crop_radius': 42,
        'clip_upscale': 4,
        'manual_review': False,
    }
    return annotated, output


def run_maize(image_path, out_dir):
    from maize.infer_dashboard import run
    result = run(str(image_path), out_dir=str(out_dir / 'maize'))
    annotated = cv2.imread(result['annotated_image'])
    # The dashboard Maize runner returns counts/rates at the top level.
    # Older versions of the Maize pipeline returned them under `analytics`.
    # Normalize both formats so the dashboard cannot fail with KeyError.
    analytics = result.get('analytics', {})
    counts = result.get('counts', analytics.get('counts', {}))
    total = result.get('total_seeds', analytics.get('total_seeds', sum(counts.values())))
    germinated_total = result.get(
        'germinated_total',
        analytics.get('germinated_total', counts.get('germinated', 0) + counts.get('semi_germinated', 0))
    )
    rate = result.get('germination_rate', analytics.get('germination_rate'))
    if rate is None:
        rate = germinated_total / total if total else 0.0

    return annotated, {
        'crop': 'Maize',
        'image': result.get('image', str(image_path)),
        'annotated_image': result.get('annotated_image'),
        'counts': counts,
        'total_seeds': total,
        'germinated_total': germinated_total,
        'germination_rate': rate,
        'weighted_germination_rate': result.get('weighted_germination_rate', analytics.get('weighted_germination_rate')),
        'predictions': result.get('predictions', []),
        'method': 'HSV seed/shoot detection + skeleton graph + CLIP tie-break'
    }


def analyze(image_path, out_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')

    crop, meta = detect_crop(image)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if crop == 'Ragi':
        annotated, result = run_ragi(image)
    else:
        annotated, result = run_maize(Path(image_path), out_dir)

    result['auto_crop_detection'] = meta
    return annotated, result
