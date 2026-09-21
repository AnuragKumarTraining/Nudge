import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from deepface import DeepFace
from ultralytics import YOLO

# 1. Paths configuration
output_dir = "cropped_faces"
os.makedirs(output_dir, exist_ok=True)

model_path = (
    r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\best (1).pt"
)
image_path = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\Anji_current.jpg"
reference_path = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\Anurag_image.jpg"

# 2. Load trained YOLO model & input image
model = YOLO(model_path)
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Target image not found: {image_path}")

# 3. Detect faces using YOLO
results = model.predict(source=image_path, conf=0.5, save=False)

# 4. Extract, save, and verify against reference
person_found = False

for r in results:
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # Ensure bounding coordinates stay inside the image frame with margin
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop face
        cropped_face = img[y1:y2, x1:x2]
        if cropped_face.size == 0:
            continue

        filename = os.path.join(output_dir, f"face_crop_{i + 1}.jpg")
        cv2.imwrite(filename, cropped_face)

        try:
            verification = DeepFace.verify(
                img1_path=filename,
                img2_path=reference_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                distance_metric="cosine",
                align=True,
                enforce_detection=False,
            )

            if verification["verified"]:
                person_found = True
                break  # Stop checking further faces once a match is found

        except Exception as e:
            pass

    if person_found:
        break

# Final verdict
if person_found:
    print("Match: Same Person")
else:
    print("Match: Different Person")