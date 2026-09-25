import os
import glob
import cv2
from Bulbs.bulb_detector import BulbDetector

# Setup directories
BASE_DIR = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge"
IMAGE_DIR = os.path.join(BASE_DIR, "Cafe_Lights")  # Or your input images path
OUTPUT_DIR = os.path.join(BASE_DIR, "Cafe_Lights_Testing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Bulb Engine
print("[INIT] Loading Bulb Detector and VLM Classifier...")
bulb_engine = BulbDetector()

# Gather images
supported_exts = ("*.jpg", "*.jpeg", "*.png")
image_files = []
for ext in supported_exts:
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

print(f"[INFO] Found {len(image_files)} image(s) to process.\n")

for img_path in image_files:
    filename = os.path.basename(img_path)
    stem, ext = os.path.splitext(filename)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Could not load image: {filename}")
        continue

    # Execute bulb detection + VLM verification
    bulb_res = bulb_engine.detect_bulbs(img)
    annotated = bulb_res.get("annotated_frame")
    active_count = bulb_res.get("count", 0)

    # Save to Testing directory
    if annotated is not None:
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_annotated{ext}")
        cv2.imwrite(out_path, annotated)
        print(f"[DONE] {filename} -> {out_path} (Active Bulbs: {active_count})")

print(f"\nAll images processed. Annotated outputs saved in: {OUTPUT_DIR}")