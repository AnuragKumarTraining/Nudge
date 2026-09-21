import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingService:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GROUNDING] Loading Grounding DINO ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.is_available = True

    def detect_objects(self, image_bgr: np.ndarray, candidate_classes: list[str],
                       box_threshold: float = 0.30, text_threshold: float = 0.25) -> list[dict]:
        if not candidate_classes:
            return []

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        # Grounding DINO expects classes formatted as a single dot-separated string
        text_prompt = ". ".join(candidate_classes) + "."

        inputs = self.processor(images=pil_img, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Common kwargs across transformers versions
        post_kwargs = {
            "outputs": outputs,
            "input_ids": inputs.input_ids,
            "target_sizes": [(h, w)]
        }

        # Handle API variation across different `transformers` library versions
        try:
            results = self.processor.post_process_grounded_object_detection(
                **post_kwargs,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )[0]
        except TypeError:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    **post_kwargs,
                    threshold=box_threshold,
                    text_threshold=text_threshold
                )[0]
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    **post_kwargs,
                    text_threshold=text_threshold
                )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("labels", results.get("phrases", []))

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            # Enforce box_threshold filtering manually in Python
            if float(score) < box_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            
            # Normalize coordinates (0.0 to 1.0)
            norm_box = [
                round(x1 / w, 4),
                round(y1 / h, 4),
                round(x2 / w, 4),
                round(y2 / h, 4)
            ]

            clean_label = str(label).strip().lower()

            detections.append({
                "class": clean_label,
                "confidence": float(round(score, 3)),
                "source": "vlm_grounded",
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_normalized": norm_box,
                "centroid": [
                    round((norm_box[0] + norm_box[2]) / 2, 4),
                    round((norm_box[1] + norm_box[3]) / 2, 4)
                ]
            })

        return detections