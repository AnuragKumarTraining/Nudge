import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
            # [0] Target Luminaire (Physical active lighting fixtures)
            "a close-up photo of an active glowing light fixture, outdoor garden pathway light, hanging patio lamp, landscape spotlight, or illuminated luminaire",

            # [1] Environmental Specular Reflections, Plant Glare & Gloss
            "a photo of specular reflection on glossy plant leaves, wet brick pathway glare, shiny chrome appliance glare, wet floor reflection, or lens flare",

            # [2] Human Skin, Face & Body Highlights (Eliminates forehead/face false positives)
            "a photo of human skin, a person's face, forehead highlight, oily skin reflection, nose, cheeks, facial hair, or human body",

            # [3] Natural Daylight, Windows & Blinds (Eliminates louver/window false positives)
            "a photo of bright outdoor daylight, a sunlit window, glass louvers, horizontal window blinds, window slats, or outdoor skylight",

            # [4] Architectural Surfaces, Foliage & Inert Hardware
            "a photo of a blank white ceiling, plain drywall, dark night sky, tree branches, unlit foliage, door frame, or electrical switchboard"
        ]

        text_inputs = self.processor(
            text=self.candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            # Process through the text tower and apply the projection layer manually
            text_outputs = self.model.text_model(**text_inputs)
            text_feats = self.model.text_projection(text_outputs.pooler_output)
            self.text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
        print("Done.")
        print("[STATUS] VLM Engine is active and ready for inference.\n")

    def verify_crops_batch(self, crops_bgr):
        """
        Processes ALL candidate crops simultaneously with competitive scoring to reject leaf glares.
        """
        if not crops_bgr:
            return []

        total = len(crops_bgr)
        print(f"[VLM INFERENCE] Batch-evaluating {total} candidate crop(s) on {self.device.upper()}...", end=" ", flush=True)

        pil_images = [Image.fromarray(c[:, :, ::-1]) for c in crops_bgr]

        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)

        if self.device == "cuda":
            inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            # Process through the vision tower and apply the projection layer manually
            vision_outputs = self.model.vision_model(**inputs)
            image_feats = self.model.visual_projection(vision_outputs.pooler_output)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits = (image_feats @ self.text_features.T) * logit_scale
            probs = logits.softmax(dim=-1).tolist()

        print("Done.")

        results = []
        for i, p in enumerate(probs):
            bulb_conf = p[0]
            reflection_conf = p[1]
            daylight_conf = p[3]

            h, w = crops_bgr[i].shape[:2]
            is_small = (h * w) < 2500
            required_conf = 0.40 if is_small else 0.55

            # Competitive Selection: Bulb score must explicitly beat the reflection/leaf glare score
            is_bulb = (
                bulb_conf > reflection_conf 
                and bulb_conf > daylight_conf 
                and bulb_conf >= required_conf
            )
            
            pred_label = "bulb" if is_bulb else "artifact"
            results.append((is_bulb, bulb_conf, pred_label))

        return results