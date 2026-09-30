import sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def _pad_square(c):
    """
    Pads a crop to a square with its median colour so CLIP's resize +
    center-crop doesn't cut off the ends of long, thin fixtures (LED strips).
    """
    h, w = c.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    fill = tuple(int(v) for v in np.median(c.reshape(-1, c.shape[2]), axis=0))
    return cv2.copyMakeBorder(
        c, top, s - h - top, left, s - w - left,
        cv2.BORDER_CONSTANT, value=fill
    )


class VLMClassifier:
    def __init__(self, model_id="openai/clip-vit-large-patch14-336"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "System CPU"

        print("\n" + "=" * 60)
        print(f"[VLM INIT] Initializing zero-shot verifier...")
        print(f"  • Model Checkpoint : {model_id}")
        print(f"  • Compute Target   : {self.device.upper()} ({device_name})")
        print("=" * 60)

        print("[1/3] Loading CLIP processor...", end=" ", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print("Done.")

        print("[2/3] Loading Vision Transformer weights into memory...", end=" ", flush=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        print("Done.")

        print("[3/3] Pre-computing text embeddings for target classes...", end=" ", flush=True)
        self.candidate_labels = [
            # [0] Target Luminaire (outdoor / garden / patio fixtures)
            "a close-up photo of an active glowing light fixture, outdoor garden pathway light, hanging patio lamp, landscape spotlight, or illuminated luminaire",

            # [1] Target Luminaire (indoor ceiling fixtures)  <-- NEW positive class
            "a photo of a switched-on ceiling light: recessed downlight, LED linear strip light, tube light, panel light, or pendant lamp glowing bright white",

            # [2] Environmental Specular Reflections, Plant Glare & Gloss
            "a photo of specular reflection on glossy plant leaves, wet brick pathway glare, shiny chrome appliance glare, wet floor reflection, or lens flare",

            # [3] Human Skin, Face & Body Highlights
            "a photo of human skin, a person's face, forehead highlight, oily skin reflection, nose, cheeks, facial hair, or human body",

            # [4] Natural Daylight, Windows & Blinds
            "a photo of bright outdoor daylight, a sunlit window, glass louvers, horizontal window blinds, window slats, or outdoor skylight",

            # [5] Architectural Surfaces, Foliage & Inert Hardware
            "a photo of a blank white ceiling, plain drywall, dark night sky, tree branches, unlit foliage, door frame, or electrical switchboard",

            # [6] Foliage highlights
            "a photo of dark green leaves and bushes with small bright highlights, garden foliage at night",

            # [7] Lit walls / windows / interiors
            "a photo of a brightly lit wall, a window or doorway showing an indoor room, or a white painted surface",
        ]

        # Positive classes are summed; everything else is a negative.
        self.pos_idx = [0, 1]
        self.neg_idx = [i for i in range(len(self.candidate_labels)) if i not in self.pos_idx]

        text_inputs = self.processor(
            text=self.candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_outputs = self.model.text_model(**text_inputs)
            text_feats = self.model.text_projection(text_outputs.pooler_output)
            self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
        print("Done.")
        print("[STATUS] VLM Engine is active and ready for inference.\n")

    def verify_crops_batch(self, crops_bgr, scene_t=0.0):
        """
        Processes ALL candidate crops simultaneously with competitive scoring.
        scene_t: 0.0 = night, 1.0 = bright indoor/day. Relaxes the confidence
        requirement slightly in bright scenes; night behaviour is unchanged.
        """
        if not crops_bgr:
            return []

        total = len(crops_bgr)
        print(f"[VLM INFERENCE] Batch-evaluating {total} candidate crop(s) on {self.device.upper()}...", end=" ", flush=True)

        pil_images = [Image.fromarray(_pad_square(c)[:, :, ::-1]) for c in crops_bgr]

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)

        if self.device == "cuda":
            inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_feats = self.model.visual_projection(vision_outputs.pooler_output)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits = (image_feats @ self.text_features.T) * logit_scale
            probs = logits.softmax(dim=-1).tolist()

        print("Done.")

        results = []
        for i, p in enumerate(probs):
            bulb_conf = sum(p[j] for j in self.pos_idx)
            best_negative = max(p[j] for j in self.neg_idx)

            h, w = crops_bgr[i].shape[:2]
            is_small = (h * w) < 2500
            required_conf = (0.45 if is_small else 0.55) - 0.08 * scene_t

            is_bulb = (bulb_conf >= required_conf) and (bulb_conf > best_negative)

            pred_label = "bulb" if is_bulb else "artifact"
            results.append((is_bulb, bulb_conf, pred_label))

        return results