import cv2
import numpy as np
import os

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

def count_and_annotate_bulbs(thresh_image, original_image, min_area):
    """Finds contours, filters out small reflections by area, and draws them."""
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Keep only contours that are larger than the min_area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    annotated_image = original_image.copy()
    cv2.drawContours(annotated_image, valid_contours, -1, (0, 0, 255), 2)
    
    return len(valid_contours), annotated_image

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
    """Main pipeline to run the modular functions and display intermediate steps."""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    # 1. Process the image through the modular steps
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.5)
    thresh = isolate_active_bulbs(smooth, threshold_value=230)
    # Adjust min_area up or down depending on how large the leaf reflection is
    bulb_count, final_image = count_and_annotate_bulbs(thresh, image, min_area=4)

    dark_mask, dark_zones_image = identify_dark_zones(image, dark_threshold=40, min_area=5000)

    print(f"Total working bulbs detected: {bulb_count}")

    # 2. Save the outputs to a folder
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "1_luminosity.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "2_haziness_reduced.png"), smooth)
    cv2.imwrite(os.path.join(output_dir, "3_brightness_threshold.png"), thresh)
    cv2.imwrite(os.path.join(output_dir, "4_detected_bulbs.png"), final_image)
    cv2.imwrite(os.path.join(output_dir, "5_dark_zones_mask.png"), dark_mask)
    cv2.imwrite(os.path.join(output_dir, "6_dark_zones_highlighted.png"), dark_zones_image)

    print(f"Images successfully saved to the '{output_dir}' folder.")

if __name__ == "__main__":
    # Replace with your actual image path
    process_bulb_image('image.png')