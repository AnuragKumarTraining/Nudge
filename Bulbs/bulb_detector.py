import cv2
import numpy as np
from Bulbs.state_analyzer import detect_bulb_candidates, get_luminosity, scene_brightness
from Bulbs.vlm_classifier import VLMClassifier


class BulbDetector:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14-336"):
        self.vlm = VLMClassifier(model_id=model_id)

    def detect_bulbs(
        self,
        frame: np.ndarray,
        threshold_value: int | None = None,
        min_area: int | None = None,
        pad_pixels: int = 45,
        debug: bool = False,
    ) -> dict:
        """
        Executes candidate extraction + zero-shot CLIP classification.
        Adapts to night / day scenes via a continuous brightness factor t.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame passed to BulbDetector is None or empty.")

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        safe_pad = 45 if pad_pixels is None else int(pad_pixels)

        # Scene brightness: 0 = night, 1 = bright indoor/day
        t = scene_brightness(get_luminosity(frame))
        # Less context padding in bright scenes (45 night -> ~20 day),
        # otherwise a small downlight crop is mostly plain ceiling.
        safe_pad = int(round(safe_pad * (1 - 0.55 * t)))

        # 1. Candidate extraction
        candidates = detect_bulb_candidates(
            frame,
            threshold_value=threshold_value,
            min_area=min_area,
            pad_pixels=safe_pad,
            debug=debug,
        )

        annotated_frame = frame.copy()

        if not candidates:
            return {
                "detected": False,
                "count": 0,
                "detections": [],
                "annotated_frame": annotated_frame,
            }

        # 2. Extract valid image crops (Using padded bbox for VLM)
        MIN_CROP = 80   # minimum context window in pixels
        valid_candidates, crops = [], []
        for cand in candidates:
            x1, y1, x2, y2 = [int(v) for v in cand["bbox"]]

            # grow small boxes symmetrically around their centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_w = max((x2 - x1) // 2, MIN_CROP // 2)
            half_h = max((y2 - y1) // 2, MIN_CROP // 2)
            x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
            y1, y2 = max(0, cy - half_h), min(h, cy + half_h)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] > 2 and crop.shape[1] > 2:
                cand["bbox"] = [x1, y1, x2, y2]
                valid_candidates.append(cand)
                crops.append(crop)

        if not crops:
            return {
                "detected": False,
                "count": 0,
                "detections": [],
                "annotated_frame": annotated_frame,
            }

        # 3. Batch VLM inference (scene-adaptive confidence)
        vlm_results = self.vlm.verify_crops_batch(crops, scene_t=t)

        # 4. Filter confirmed detections and annotate frame
        confirmed_bulbs = []
        bulb_idx = 1
        for cand, (is_bulb, conf, label) in zip(valid_candidates, vlm_results):
            print(
                f"t={t:.2f} bulb conf={conf:.2f} "
                f"core={cand.get('core_frac', 0):.2f} "
                f"contrast={cand.get('contrast', 0):.0f} "
                f"area={cand.get('area', 0):.0f}"
            )
            if not is_bulb:
                continue

            # Retrieve the TIGHT bounding box for drawing
            dx1, dy1, dx2, dy2 = cand.get("draw_bbox", cand["bbox"])

            confirmed_bulbs.append(
                {
                    "id": bulb_idx,
                    "confidence": round(float(conf), 4),
                    "label": str(label),
                    "bbox": [dx1, dy1, dx2, dy2],  # Store tight box in output
                    "contour_area": cand.get("area", 0.0),
                }
            )

            # Draw tight bounding box
            cv2.rectangle(annotated_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

            # Text label badge
            text = f"Bulb {conf * 100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            # Align text to the tight drawing box
            text_y = dy1 - 6 if dy1 - 6 > text_h + 4 else dy1 + text_h + 6

            # Background rectangle behind text
            cv2.rectangle(
                annotated_frame,
                (dx1, text_y - text_h - 2),
                (dx1 + text_w + 2, text_y + baseline - 1),
                (0, 180, 0),
                -1,
            )
            cv2.putText(
                annotated_frame,
                text,
                (dx1 + 1, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            bulb_idx += 1

        return {
            "detected": len(confirmed_bulbs) > 0,
            "count": len(confirmed_bulbs),
            "detections": confirmed_bulbs,
            "annotated_frame": annotated_frame,
        }