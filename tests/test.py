import cv2
import json
import numpy as np

def draw_detections_on_image(image_path, json_data, output_path="output.jpg"):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # 2. Iterate through each detected object in the JSON list
    for obj in json_data:
        # Extract class name and confidence
        label = f"{obj['class']} ({obj['confidence']:.2f})"
        
        # --- Draw Bounding Box ---
        bbox = obj['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        # Draw green rectangle for the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # --- Draw Mask Polygon ---
        if 'mask_polygon' in obj and 'points_pixel' in obj['mask_polygon']:
            # Convert points list to a numpy array of shape (N, 1, 2) required by OpenCV
            points = np.array(obj['mask_polygon']['points_pixel'], np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Draw blue polygon for the mask (isClosed=True)
            cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)
            
        # --- Draw Centroid ---
        if 'centroid' in obj:
            cx, cy = int(obj['centroid']['x']), int(obj['centroid']['y'])
            cv2.circle(img, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1) # Red dot
            
        # --- Add Text Label ---
        # Put the label just above the bounding box
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 3. Save or display the output
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

# --- Example Usage ---
# Assuming your JSON array is saved in a file or string.
# (Note: I closed the bracket from your snippet for valid JSON)
json_string = """
[
    {
      "object_id": "obj_ov_018_fire_extinguisher",
      "class": "fire extinguisher",
      "confidence": 0.233,
      "bbox": { "x1": 914, "y1": 240, "x2": 936, "y2": 296 },
      "centroid": { "x": 925, "y": 268 },
      "mask_polygon": {
        "points_pixel": [
          [914, 240], [914, 295], [920, 295], [915, 269], 
          [926, 259], [933, 267], [929, 295], [935, 295], [935, 240]
        ]
      }
    }
]
"""

# Load the JSON data
data = json.loads(json_string)

# Call the function (replace 'cafe_input.jpg' with your actual image path)
# draw_detections_on_image('cafe_input.jpg', data)