import os
import cv2
import numpy as np
from Bulbs.state_analyzer import detect_bulb_candidates
from Bulbs.vlm_classifier import VLMClassifier

class BulbDetector:
    def __init__(self, model_id="openai/clip-vit-large-patch14-336"):
        self.vlm = VLMClassifier(model_id=model_id)

    def detect_bulbs(self, frame, threshold_value=225, min_area=30, pad_pixels=35):
        """
        Executes candidate extraction + zero-shot CLIP classification.
        Returns:
            dict: {
                "detected": bool,
                "count": int,
                "detections": list of confirmed bulb dicts,
                "annotated_frame": frame with drawn bounding boxes
            }
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        candidates = detect_bulb_candidates(
            frame,
            threshold_value=threshold_value,
            min_area=min_area,
            pad_pixels=pad_pixels
        )

        annotated_frame = frame.copy()

        if not candidates:
            return {
                "detected": False,
                "count": 0,
                "detections": [],
                "annotated_frame": annotated_frame
            }

        crops = [frame[c["bbox"][1]:c["bbox"][3], c["bbox"][0]:c["bbox"][2]] for c in candidates]
        vlm_results = self.vlm.verify_crops_batch(crops)

        confirmed_bulbs = []
        for idx, (cand, (is_bulb, conf, label)) in enumerate(zip(candidates, vlm_results), start=1):
            x1, y1, x2, y2 = cand["bbox"]
            if is_bulb:
                confirmed_bulbs.append({
                    "id": idx,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"Bulb ({conf * 100:.0f}%)",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        return {
            "detected": len(confirmed_bulbs) > 0,
            "count": len(confirmed_bulbs),
            "detections": confirmed_bulbs,
            "annotated_frame": annotated_frame
        }