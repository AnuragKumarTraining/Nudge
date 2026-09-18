import numpy as np

from crop_surface import crop_and_mask_surface
from stain_detection import detect_stains
from texture_detection import extract_texture_baseline


def analyze_single_surface(image, bbox, polygon_coords, material):
    # Step A: Isolate ROI
    masked_crop, crop_mask = crop_and_mask_surface(image, bbox, polygon_coords)
    
    # Total surface pixel area
    surface_area_px = int(np.count_nonzero(crop_mask))
    if surface_area_px == 0:
        return {"error": "Invalid polygon mask area"}
    
    # Step B: Detect Stains
    stain_candidates, total_stain_area = detect_stains(masked_crop, crop_mask)
    
    # Step C: Extract Texture
    lbp_hist, glcm_features = extract_texture_baseline(masked_crop, crop_mask)
    
    # Step D: Compute Surface Cleanliness Score (0.0 = Dirty, 1.0 = Clean)
    stain_coverage_ratio = total_stain_area / surface_area_px
    cleanliness_score = max(0.0, min(1.0, 1.0 - (stain_coverage_ratio * 3.5)))
    
    return {
        "material_hint": material,
        "surface_area_px": surface_area_px,
        "cleanliness_score": round(cleanliness_score, 2),
        "stains_detected_count": len(stain_candidates),
        "stain_coverage_ratio": round(stain_coverage_ratio, 4),
        "stain_candidates": stain_candidates,
        "texture_baseline": {
            "glcm": glcm_features,
            "lbp_histogram": lbp_hist
        }
    }