"""Zero-shot frame classification using a pretrained CLIP model.

No training/fine-tuning is performed: class membership is decided purely by
cosine similarity between a frame's image embedding and text embeddings of
natural-language class descriptions (prompts.py). This is what makes the
detector "zero-shot" — it works on day one for the "inside a vehicle" class
without any labeled video dataset.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .prompts import (
    CLASS_NON_VEHICLE,
    CLASS_VEHICLE,
    NON_VEHICLE_PROMPTS,
    PROMPT_TEMPLATES,
    VEHICLE_PROMPTS,
)

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


def _as_tensor(features) -> torch.Tensor:
    """Unwrap the embedding tensor regardless of transformers version.

    Older transformers releases have `get_text_features`/`get_image_features`
    return the projected embedding tensor directly; newer ones (>=4.5x style
    CLIP refactor) return a `BaseModelOutputWithPooling` whose `pooler_output`
    holds that same projected embedding.
    """
    if isinstance(features, torch.Tensor):
        return features
    return features.pooler_output


@dataclass
class FrameScore:
    vehicle_score: float
    label: str


class ClipZeroShotClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self._vehicle_text_emb = self._embed_class_prompts(VEHICLE_PROMPTS)
        self._non_vehicle_text_emb = self._embed_class_prompts(NON_VEHICLE_PROMPTS)

    @torch.no_grad()
    def _embed_class_prompts(self, base_prompts: list[str]) -> torch.Tensor:
        """Average the embeddings of every (template, prompt) pair for a class.

        Averaging several phrasings per class ("prompt ensembling") is a
        standard CLIP zero-shot technique that reduces variance from any
        single prompt's exact wording.
        """
        expanded = [tmpl.format(p) for p in base_prompts for tmpl in PROMPT_TEMPLATES]
        inputs = self.processor(text=expanded, return_tensors="pt", padding=True).to(self.device)
        text_features = _as_tensor(self.model.get_text_features(**inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def score_frames(self, images_rgb: list[np.ndarray]) -> list[FrameScore]:
        """Return a vehicle-class probability and label for each frame.

        The two class embeddings are each a single averaged vector, so per
        frame this reduces to a 2-way softmax over cosine similarities
        (scaled by CLIP's learned logit_scale), rather than an N-prompt
        softmax — the prompt ensembling already happened at embed time.
        """
        pil_images = [Image.fromarray(img) for img in images_rgb]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        image_features = _as_tensor(self.model.get_image_features(**inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        class_text_emb = torch.cat([self._vehicle_text_emb, self._non_vehicle_text_emb], dim=0)
        logits = logit_scale * image_features @ class_text_emb.T
        probs = logits.softmax(dim=-1)

        results = []
        for vehicle_prob, non_vehicle_prob in probs.tolist():
            label = CLASS_VEHICLE if vehicle_prob >= non_vehicle_prob else CLASS_NON_VEHICLE
            results.append(FrameScore(vehicle_score=vehicle_prob, label=label))
        return results
