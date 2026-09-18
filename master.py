<<<<<<< Updated upstream
=======
import os
import json
import cv2
from utils.model import load_yolo
from utils.features import extract_features
from config.master_config import*


def generate_all_baselines():
    os.makedirs(BASELINES_DIR, exist_ok=True)
    images = [f for f in os.listdir(MASTER_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    model = load_yolo()

    for img_name in images:
        room_name = os.path.splitext(img_name)[0]
        img_path = os.path.join("images", "master.jpg")
        
        json_out = os.path.join(BASELINES_DIR, f"{room_name}_baseline.json")
        ref_out = os.path.join(BASELINES_DIR, f"{room_name}_ref.jpg")

        data, img = extract_features(img_path, model)

        cv2.imwrite(ref_out, img)
        with open(json_out, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[SUCCESS] Baseline created for {room_name}")
    
generate_all_baselines()
>>>>>>> Stashed changes
