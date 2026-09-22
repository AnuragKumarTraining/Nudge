import os
import cv2
from ultralytics import YOLO

# 1. Setup output directory
output_dir = 'cropped_faces'
os.makedirs(output_dir, exist_ok=True)

# 2. Load model weights
model_path = r'C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\Face\FaceDetectionModel.pt'
model = YOLO(model_path)

# 3. Choose source (0 for webcam, or a string path for an image)
source = 0
# source = r'C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\FacePredictions.jpg'

is_webcam = isinstance(source, int)

# Use DirectShow backend on Windows to avoid MSMF errors
if is_webcam:
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {source}")

# 4. Button definitions & mouse callback
btn_capture = (20, 20, 180, 60)   # (x1, y1, x2, y2)
btn_quit = (195, 20, 305, 60)      # (x1, y1, x2, y2)

capture_requested = False
quit_requested = False

def on_mouse_click(event, x, y, flags, param):
    global capture_requested, quit_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if "Capture Face" was clicked
        cx1, cy1, cx2, cy2 = btn_capture
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            capture_requested = True

        # Check if "Quit" was clicked
        qx1, qy1, qx2, qy2 = btn_quit
        if qx1 <= x <= qx2 and qy1 <= y <= qy2:
            quit_requested = True

window_name = "Detection Feed"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse_click)

frame_count = 0

# 5. Main Processing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        if is_webcam:
            print("Failed to grab frame from camera.")
        break

    frame_count += 1
    h, w, _ = frame.shape

    # Check keyboard inputs ('q' to quit, 'c' to capture)
    delay = 1 if is_webcam else 0
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q') or quit_requested:
        break
    elif key == ord('c'):
        capture_requested = True

    # Run YOLO inference
    results = model.predict(source=frame, conf=0.5, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_face = frame[y1:y2, x1:x2]

            # Save cropped face only when capture is clicked/requested
            if capture_requested and cropped_face.size > 0:
                filename = os.path.join(output_dir, f'crop_{frame_count}_face_{i + 1}.jpg')
                cv2.imwrite(filename, cropped_face)
                print(f"Saved: {filename}")

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i + 1}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Reset capture flag for the next frame
    if capture_requested:
        capture_requested = False

    # Draw "Capture Face" button (Orange)
    cx1, cy1, cx2, cy2 = btn_capture
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (40, 140, 255), -1)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
    cv2.putText(frame, "Capture Face", (cx1 + 12, cy1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw "Quit" button (Red)
    qx1, qy1, qx2, qy2 = btn_quit
    cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (40, 40, 220), -1)
    cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 255, 255), 2)
    cv2.putText(frame, "Quit", (qx1 + 32, qy1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    # If testing a single image file, hold until user quits
    if not is_webcam:
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()