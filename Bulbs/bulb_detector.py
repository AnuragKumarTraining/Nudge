import os
import cv2
import numpy as np
from Bulbs.state_analyzer_2 import detect_bulb_candidates
from Bulbs.vlm_classifier_2 import VLMClassifier

class BulbDetector:
    def __init__(self, model_id="openai/clip-vit-large-patch14-336"):
        """Initializes the local CLIP VLM verifier once."""
        self.vlm = VLMClassifier(model_id=model_id)

    def detect_bulbs(self, frame, threshold_value=225, min_area=30, pad_pixels=35):
        """
        Executes two-stage detection on a single image/frame.
        
        Args:
            frame (np.ndarray): Input image in BGR format.
            threshold_value (int): Luminosity threshold for candidate glowing spots.
            min_area (int): Minimum pixel area to discard high-frequency glints.
            pad_pixels (int): Context padding around candidate crops.
            
        Returns:
            dict: {
                "detected": bool,
                "count": int,
                "detections": list of dicts with bbox and confidence,
                "annotated_frame": np.ndarray
            }
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]

        # Stage 1: OpenCV candidate extraction
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

        # Stage 2: Batch VLM evaluation
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

                # Draw green bounding box & label on frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"Bulb ({conf * 100:.0f}%)"
                cv2.putText(annotated_frame, tag, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        return {
            "detected": len(confirmed_bulbs) > 0,
            "count": len(confirmed_bulbs),
            "detections": confirmed_bulbs,
            "annotated_frame": annotated_frame
        }