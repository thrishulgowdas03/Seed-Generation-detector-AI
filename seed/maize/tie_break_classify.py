"""
For seeds where the skeleton-tracing step found no nearby shoot tip
(shoot_found=False), we still need to decide: is there a short,
partially-emerged root visible (semi_germinated), or truly nothing
(non_germinated)? Root color is too close to the paper towel to detect
reliably with plain thresholding, so this narrow, 2-way decision is
handed to CLIP zero-shot on a moderate crop around the seed.

This keeps CLIP calls limited to only the ambiguous seeds (typically a
handful per image), which keeps this fast even on CPU.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

TIE_BREAK_PROMPTS = {
    "semi_germinated": [
        "a maize seed with a short partial root just starting to emerge, no leaf",
        "a corn kernel showing an early stub of a root, incomplete germination",
    ],
    "non_germinated": [
        "an intact dry maize seed with no visible root or sprout at all",
        "a maize kernel showing no signs of germination, smooth and unbroken",
    ],
}


class TieBreakClassifier:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        all_prompts, self.prompt_to_class = [], []
        for cls, prompts in TIE_BREAK_PROMPTS.items():
            for p in prompts:
                all_prompts.append(p)
                self.prompt_to_class.append(cls)

        text_inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

    def classify_crop(self, crop_bgr: np.ndarray):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        image_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ self.text_features.T).squeeze(0)
        probs = torch.softmax(sims * 100, dim=0)

        class_scores = {"semi_germinated": 0.0, "non_germinated": 0.0}
        for prob, cls in zip(probs.tolist(), self.prompt_to_class):
            class_scores[cls] += prob

        best = max(class_scores, key=class_scores.get)
        return best, round(class_scores[best], 4)


def extract_crop(image_bgr: np.ndarray, seed: dict, size_multiplier: float = 5.0):
    """Crop centered on the seed, sized relative to seed dimensions."""
    seed_size = max(seed["w"], seed["h"])
    r = int(seed_size * size_multiplier)
    x0 = max(0, seed["cx"] - r)
    y0 = max(0, seed["cy"] - r)
    x1 = min(image_bgr.shape[1], seed["cx"] + r)
    y1 = min(image_bgr.shape[0], seed["cy"] + r)
    return image_bgr[y0:y1, x0:x1]
