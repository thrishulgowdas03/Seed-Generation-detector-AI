"""
PADDY CLIP V2 — MAIZE-STYLE + PADDY TAILORING

Goal:
- Keep the existing OpenCV seed detector unchanged.
- Use the same pretrained CLIP model as Maize.
- Classify every detected Paddy seed using Paddy-specific prompts.
- Use visible yellow/green shoot evidence as an additional signal,
  NOT as an automatic GERMI override.
- This makes CLIP the primary classifier while OpenCV provides
  biologically useful growth evidence.

Classes:
    GERMI       = clear/developed germination
    SEMI GERMI  = early/partial germination
    NON GERMI   = no visible germination
"""

import cv2
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

LABELS = ["GERMI", "SEMI GERMI", "NON GERMI"]

# Paddy-specific prompts. Multiple descriptions reduce dependence on
# one particular wording.
PROMPTS = {
    "GERMI": [
        "a close-up paddy rice seed with a clearly developed root and visible green shoot",
        "a clearly germinated rice seed with obvious root growth and a developed green seedling",
        "a paddy seedling with a long visible root emerging from the rice seed",
        "a germinated paddy rice seed showing strong visible root or shoot growth",
        "a rice seed with clear advanced germination and visible seedling growth",
    ],
    "SEMI GERMI": [
        "a paddy rice seed with a very short root just beginning to emerge",
        "an early germinating rice seed with a small short sprout",
        "a partially germinated paddy seed with limited root growth",
        "a rice seed showing early germination but only a small emerging root or shoot",
        "a paddy seed with slight visible germination and a short emerging root",
    ],
    "NON GERMI": [
        "an intact dry paddy rice seed with no root and no shoot",
        "an ungerminated rice seed with no visible germination",
        "a paddy seed showing only the seed body with no emerging growth",
        "a dry paddy rice seed with no visible root or seedling",
        "an ungerminated rice seed lying on paper with no root or shoot",
    ],
}

# Yellow/green shoot detection. Kept deliberately conservative.
SHOOT_H_MIN = 20
SHOOT_H_MAX = 78
SHOOT_S_MIN = 45
SHOOT_V_MIN = 60
SHOOT_MIN_AREA = 8

# The shoot must be close to the detected seed.
SHOOT_RADIUS_MIN = 18
SHOOT_RADIUS_MAX = 65

# Fusion strength. CLIP remains primary; shoot evidence is a bonus.
SHOOT_BONUS_CLEAR = 0.22
SHOOT_BONUS_PARTIAL = 0.08


@lru_cache(maxsize=1)
def load_clip():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return processor, model, device


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _get_text_features(model, inputs):
    out = model.text_model(**inputs)
    pooled = out.pooler_output
    return _normalize(model.text_projection(pooled))


def _get_image_features(model, inputs):
    out = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )
    pooled = out.pooler_output
    return _normalize(model.visual_projection(pooled))


def _make_shoot_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array(
            [SHOOT_H_MIN, SHOOT_S_MIN, SHOOT_V_MIN],
            dtype=np.uint8,
        ),
        np.array(
            [SHOOT_H_MAX, 255, 255],
            dtype=np.uint8,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    clean = np.zeros_like(mask)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= SHOOT_MIN_AREA:
            clean[labels == i] = 255

    return clean


def _shoot_evidence(image_bgr, seed, shoot_mask):
    """
    Estimate local shoot evidence around a detected seed.

    Returns:
        0.0 = none
        0.5 = partial/weak
        1.0 = clear
    """
    h, w = shoot_mask.shape[:2]

    x = int(round(seed["x"]))
    y = int(round(seed["y"]))

    if not (0 <= x < w and 0 <= y < h):
        return 0.0, 0.0

    size = max(float(seed.get("size", 12.0)), 6.0)

    radius = int(
        np.clip(size * 4.0, SHOOT_RADIUS_MIN, SHOOT_RADIUS_MAX)
    )

    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)

    roi = shoot_mask[y0:y1, x0:x1]

    if roi.size == 0:
        return 0.0, 0.0

    pixels = int(np.count_nonzero(roi))
    area = roi.shape[0] * roi.shape[1]
    density = pixels / max(area, 1)

    # Check whether green/yellow pixels are immediately around the seed.
    seed_radius = int(np.clip(size * 1.8, 8, 22))
    sx0 = max(0, x - seed_radius)
    sy0 = max(0, y - seed_radius)
    sx1 = min(w, x + seed_radius + 1)
    sy1 = min(h, y + seed_radius + 1)

    near = shoot_mask[sy0:sy1, sx0:sx1]
    near_pixels = int(np.count_nonzero(near))

    # Very close colored growth is stronger evidence than distant growth.
    if near_pixels >= 5 and density >= 0.003:
        return 1.0, density

    if pixels >= 8 and density >= 0.0015:
        return 0.5, density

    return 0.0, density


def _make_crop(image_bgr, seed, radius):
    h, w = image_bgr.shape[:2]

    x = int(round(seed["x"]))
    y = int(round(seed["y"]))

    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)

    crop = image_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        crop = np.zeros((128, 128, 3), dtype=np.uint8)

    crop = cv2.resize(
        crop,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    return Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )


def _build_class_features(processor, model, device):
    texts = []
    owners = []

    for label in LABELS:
        for prompt in PROMPTS[label]:
            texts.append(prompt)
            owners.append(label)

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = _get_text_features(model, inputs)

    class_features = []

    for label in LABELS:
        idx = [
            i for i, owner in enumerate(owners)
            if owner == label
        ]

        f = features[idx].mean(dim=0, keepdim=True)
        class_features.append(_normalize(f))

    return torch.cat(class_features, dim=0)


def _clip_classify(crops, processor, model, device, class_features):
    inputs = processor(
        images=crops,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = _get_image_features(
            model, inputs
        )

        similarities = image_features @ class_features.T

        # CLIP's standard high-temperature scaling.
        probabilities = torch.softmax(
            similarities * 100.0,
            dim=-1,
        ).cpu().numpy()

    return probabilities


def classify_paddy(image_bgr, candidates, batch_size=16):
    """
    Classify detected Paddy seeds.

    CLIP is the primary classifier for all candidates.
    OpenCV shoot evidence is fused as a modest GERMI bonus.
    """

    if not candidates:
        return []

    processor, model, device = load_clip()
    class_features = _build_class_features(
        processor,
        model,
        device,
    )

    shoot_mask = _make_shoot_mask(image_bgr)

    results = []

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]

        # Three scales: tight seed, normal context, wider growth context.
        # The medium view is weighted most heavily.
        crop_sets = [
            [
                _make_crop(image_bgr, seed, 34)
                for seed in batch
            ],
            [
                _make_crop(image_bgr, seed, 52)
                for seed in batch
            ],
            [
                _make_crop(image_bgr, seed, 72)
                for seed in batch
            ],
        ]

        probs_by_view = [
            _clip_classify(
                crops,
                processor,
                model,
                device,
                class_features,
            )
            for crops in crop_sets
        ]

        # Tight seed / normal context / wider growth context.
        probs = (
            0.25 * probs_by_view[0]
            + 0.50 * probs_by_view[1]
            + 0.25 * probs_by_view[2]
        )

        for i, seed in enumerate(batch):
            shoot_level, shoot_density = _shoot_evidence(
                image_bgr,
                seed,
                shoot_mask,
            )

            fused = probs[i].copy()

            # Add modest biological evidence for GERMI.
            # Never force a GERMI label from color alone.
            if shoot_level == 1.0:
                fused[0] += SHOOT_BONUS_CLEAR
            elif shoot_level == 0.5:
                fused[0] += SHOOT_BONUS_PARTIAL

            fused = fused / fused.sum()

            idx = int(np.argmax(fused))

            q = dict(seed)
            q["class"] = LABELS[idx]
            q["confidence"] = round(
                float(fused[idx]),
                4,
            )
            q["shoot_evidence"] = (
                "clear"
                if shoot_level == 1.0
                else "partial"
                if shoot_level == 0.5
                else "none"
            )
            q["shoot_density"] = round(
                float(shoot_density),
                5,
            )
            q["classification_method"] = (
                "Paddy CLIP + OpenCV growth evidence"
            )
            q["scores"] = {
                LABELS[k]: round(
                    float(fused[k]),
                    4,
                )
                for k in range(len(LABELS))
            }

            results.append(q)

    return results
