import cv2
import numpy as np
import os
import json

def get_luminosity(image):
    """Converts image to grayscale to isolate raw pixel intensity."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_haziness(gray_image, gamma=2.5):
    """Reduces haziness (glare) using Gamma Correction (LUT) to darken mid-tones."""
    # Create a lookup table mapping pixel values [0, 255] to their gamma-adjusted values
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply the lookup table using OpenCV
    return cv2.LUT(gray_image, table)

def isolate_active_bulbs(smoothed_image, threshold_value):
    """Applies binary thresholding to isolate the brightest light sources."""
    _, thresh = cv2.threshold(smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def is_round_or_oval(contour, min_circularity=0.10, min_axis_ratio=0.15, min_solidity=0.60):
    """
    Checks if a light source is even remotely circular, oval, or spherical.
    Allows heavily deformed/stretched ovals while filtering out long, sharp linear streaks.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area == 0:
        return False

    # 1. Very relaxed circularity check: 4 * pi * Area / Perimeter^2 (1.0 = perfect circle)
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # 2. Relaxed oval / axis ratio check
    if len(contour) >= 5:
        (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
        axis_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    else:
        _, _, w, h = cv2.boundingRect(contour)
        axis_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

    # 3. Solidity check: how solid/convex the light blob is (circles and ovals are compact)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Accept if it satisfies ANY relaxed sign of roundness/oval shape
    return (circularity >= min_circularity) or (axis_ratio >= min_axis_ratio) or (solidity >= min_solidity)

def count_and_annotate_bulbs(thresh_image, original_image, min_area):
    """Finds contours, filters by area and roundness, and compiles bulb metadata."""
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [
        c for c in contours 
        if cv2.contourArea(c) > min_area and is_round_or_oval(c)
    ]
    
    bulb_data = []
    annotated_image = original_image.copy()

    for idx, c in enumerate(valid_contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        
        # Calculate centroid coordinates
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + (w // 2)
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + (h // 2)
        
        bulb_data.append({
            "id": idx,
            "centroid": {"x": cx, "y": cy},
            "area_pixels": float(cv2.contourArea(c)),
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        })

    cv2.drawContours(annotated_image, valid_contours, -1, (0, 0, 255), 2)
    return bulb_data, annotated_image
def identify_dark_zones(image, dark_threshold ,min_area):
    """
    Identifies and highlights areas with low luminosity.
    Uses THRESH_BINARY_INV so dark pixels (below dark_threshold) become white (detectable).
    """
    # 1. Convert to grayscale for pure luminosity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply a heavy blur to combine broken shadow patches into unified zones
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    
    # 3. Inverted threshold: pixels darker than '60' become 255 (white/active)
    _, dark_mask = cv2.threshold(blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Find contours of the dark areas
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filter out small natural shadows and draw the significant dark zones
    annotated_image = image.copy()
    valid_zones = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Draw the dark zones in Blue (255, 0, 0)
    cv2.drawContours(annotated_image, valid_zones, -1, (255, 0, 0), 3)
    
    return dark_mask, annotated_image

def process_bulb_image(image_path):
    """Executes the bulb detection pipeline and saves outputs with a JSON report."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Image processing steps
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.0)
    thresh = isolate_active_bulbs(smooth, threshold_value=220)
    bulb_data, final_image = count_and_annotate_bulbs(thresh, image, min_area=4)

    # Compile JSON report
    h, w, _ = image.shape
    report = {
        "image_metadata": {
            "file_name": os.path.basename(image_path),
            "resolution": {"width": w, "height": h}
        },
        "parameters_used": {
            "gamma_correction": 2.5,
            "threshold_value": 220,
            "min_area": 4,
            "shape_filters": {
                "min_circularity": 0.10,
                "min_axis_ratio": 0.15,
                "min_solidity": 0.60
            }
        },
        "summary": {
            "total_working_bulbs": len(bulb_data),
            "lighting_status": "OPERATIONAL" if len(bulb_data) > 0 else "NO_ACTIVE_BULBS"
        },
        "detected_bulbs": bulb_data
    }

    # Save output artifacts
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "1_luminosity.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "2_haziness_reduced.png"), smooth)
    cv2.imwrite(os.path.join(output_dir, "3_brightness_threshold.png"), thresh)
    cv2.imwrite(os.path.join(output_dir, "4_detected_bulbs.png"), final_image)

    json_path = os.path.join(output_dir, "bulb_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Total working bulbs detected: {len(bulb_data)}")
    print(f"Results and report saved to '{output_dir}'.")

if __name__ == "__main__":
    # Replace with your actual image path
    process_bulb_image(r'C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\image1.png')