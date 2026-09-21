import os
import cv2
from ultralytics import YOLO

# Create folder if it doesn't exist
output_dir = 'cropped_faces'
os.makedirs(output_dir, exist_ok=True)

# 1. Load your trained weights
model = YOLO(r'C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\best (1).pt')

# 2. Path to the image you want to test
image_path = r'C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\FacePredictions.jpg'

# 3. Read image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# 4. Run prediction
results = model.predict(source=image_path, conf=0.5, save=False)

# 5. Extract, save, and show each cropped face
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # Bounding box coordinates [x1, y1, x2, y2]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        # Crop the face using array slicing
        cropped_face = img[y1:y2, x1:x2]
        
        # Save each cropped face to disk
        filename = os.path.join(output_dir, f'face_crop_{i + 1}.jpg')
        cv2.imwrite(filename, cropped_face)
        
        # Draw bounding box on the original image canvas
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Face {i + 1}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the crop
        #cv2.imshow(f"Cropped Face {i + 1}", cropped_face)
        cv2.waitKey(0)

# Display the whole image with all detected face boxes
cv2.imshow("All Detected Faces", img)
cv2.waitKey(0)

cv2.destroyAllWindows()