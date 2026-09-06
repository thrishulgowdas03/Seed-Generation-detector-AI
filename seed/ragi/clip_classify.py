import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = 'openai/clip-vit-base-patch32'

# Ragi-specific 3-way zero-shot prompts. Multiple prompts per class
# reduce dependence on one wording.
PROMPTS = {
    'GERMI': [
        'a close-up photograph of a germinated ragi seed with a clearly developed root or shoot',
        'a finger millet seed showing clear advanced germination with visible root and/or green shoot',
        'a ragi seed with a long visible emerging root or developed green sprout',
    ],
    'SEMI GERMI': [
        'a close-up photograph of a partially germinated ragi seed with a short root or tiny sprout just emerging',
        'a finger millet seed in early germination with a small newly emerged root or shoot',
        'a ragi seed showing only a short early germination growth, not fully developed',
    ],
    'NON GERMI': [
        'a close-up photograph of an ungerminated ragi seed with no visible root or sprout',
        'a dry finger millet seed showing no signs of germination',
        'a ragi seed with an intact seed body and no emerging root or shoot',
    ],
}

CLASS_ORDER = ['GERMI', 'SEMI GERMI', 'NON GERMI']


class RagiCLIPClassifier:
    def __init__(self, device=None, crop_radius=42, upscale=4):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.crop_radius = crop_radius
        self.upscale = upscale
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()

        prompts = []
        self.prompt_classes = []
        for cls in CLASS_ORDER:
            for p in PROMPTS[cls]:
                prompts.append(p)
                self.prompt_classes.append(cls)

        text_inputs = self.processor(text=prompts, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    def extract_crop(self, image, cx, cy):
        h, w = image.shape[:2]
        r = int(self.crop_radius)
        x0 = max(0, int(cx) - r)
        y0 = max(0, int(cy) - r)
        x1 = min(w, int(cx) + r + 1)
        y1 = min(h, int(cy) + r + 1)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            raise ValueError('Empty Ragi seed crop')
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if self.upscale > 1:
            crop = cv2.resize(crop, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(crop), (x0, y0, x1, y1)

    def classify(self, image, centers):
        pil_images = []
        boxes = []
        for cx, cy in centers:
            im, box = self.extract_crop(image, cx, cy)
            pil_images.append(im)
            boxes.append(box)

        image_inputs = self.processor(images=pil_images, return_tensors='pt').to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ self.text_features.T
        probs = torch.softmax(sims * 100, dim=1).cpu().numpy()

        results = []
        for row in probs:
            scores = {cls: 0.0 for cls in CLASS_ORDER}
            for j, cls in enumerate(self.prompt_classes):
                scores[cls] += float(row[j])
            total = sum(scores.values()) or 1.0
            scores = {k: v / total for k, v in scores.items()}
            best = max(scores, key=scores.get)
            results.append((best, round(scores[best], 4), {k: round(v, 4) for k, v in scores.items()}))
        return results, boxes
