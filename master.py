import os
import json
import cv2
from utils.model import load_yolo
from utils.features import extract_features
from config.master_config import *
from Bulbs.bulb_detector import BulbDetector


def generate_all_baselines():
    os.makedirs(BASELINES_DIR, exist_ok=True)
    images = [f for f in os.listdir(MASTER_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    model = load_yolo()
    bulb_engine = BulbDetector()  # <-- 1. Initialize detector once

    for img_name in images:
        room_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(MASTER_DIR, img_name)
        
        json_out = os.path.join(BASELINES_DIR, f"{room_name}_baseline.json")
        ref_out = os.path.join(BASELINES_DIR, f"{room_name}_ref.jpg")

        data, img = extract_features(img_path, model)

        # <-- 2. Detect and store baseline bulbs
        bulb_res = bulb_engine.detect_bulbs(img)
        data["bulb_count"] = bulb_res["count"]
        data["bulbs"] = bulb_res["detections"]

        cv2.imwrite(ref_out, img)
        with open(json_out, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Baseline created for {room_name} (Active Bulbs: {bulb_res['count']})")


if __name__ == "__main__":
    generate_all_baselines()