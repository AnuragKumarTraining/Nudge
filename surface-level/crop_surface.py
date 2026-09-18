import cv2
import numpy as np


def crop_and_mask_surface(image, bbox, polygon_coords):
    x1, y1, x2, y2 = bbox
    
    # 1. Bounding Box Crop
    crop = image[y1:y2, x1:x2]
    
    # 2. Adjust polygon coordinates relative to the crop origin (x1, y1)
    relative_poly = np.array(
        [[pt[0] - x1, pt[1] - y1] for pt in polygon_coords],
        dtype=np.int32,
    )
    
    # 3. Create local mask
    crop_mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillPoly(crop_mask, [relative_poly], 255)
    
    # 4. Mask out background pixels
    masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
    
    return masked_crop, crop_mask