<<<<<<< Updated upstream
=======
import os
import cv2
import json
import matplotlib.pyplot as plt

# Import functions from your utils package[cite: 1]
from utils.feature_extraction import extract_features
from utils.model import load_yolo

def main():
    # 1. Define paths according to your folder structure[cite: 1]
    image_path = os.path.join("images", "daily_2.jpg")
    input_metadata_path = os.path.join("metadata", "daily_image.json")
    output_json_path = os.path.join("metadata", "stage_3.json")

    # 2. Load the JSON metadata
    try:
        with open(input_metadata_path, 'r') as file:
            stage_metadata = json.load(file)
            print("[INFO] Metadata loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find metadata at {input_metadata_path}")
        print("[INFO] Using fallback mock metadata...")
        stage_metadata = {
            "capture_id": "cap_892347_daily_2",
            "environment": {"ambient_lux": 350}
        }

    # 3. Load the YOLO model
    print("[INFO] Loading YOLO model...")
    yolo_model = load_yolo()

    # 4. Run the feature extraction pipeline
    print(f"[INFO] Extracting features for {image_path}...")
    try:
        stage_3_json, processed_img = extract_features(
            image_path=image_path,
            model=yolo_model,
            stage_metadata=stage_metadata
        )

        # 5. Write Stage 3 output to metadata/stage_3.json
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as out_file:
            json.dump(stage_3_json, out_file, indent=4)
        print(f"[SUCCESS] Saved Stage 3 JSON to {output_json_path}")

        # 6. Visualize the image (Convert BGR to RGB for Matplotlib)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Extracted Image (Post-Lighting Normalization)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
