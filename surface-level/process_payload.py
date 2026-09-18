import json
import cv2
from analyze_surface import analyze_single_surface

def process_surface_payload(payload_json, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded.")
        
    data = json.loads(payload_json) if isinstance(payload_json, str) else payload_json
    results = []

    for obj in data.get("objects", []):
        # Only process objects marked as surface relevant
        if not obj.get("surface_relevant", False):
            continue
            
        object_id = obj["object_id"]
        obj_class = obj["class"]
        material = obj.get("material_hint", "unknown")
        
        # Extract BBox coordinates
        bbox = obj["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Extract pixel coordinates from the polygon payload object.
        mask_polygon = obj.get("mask_polygon", {})
        polygon_coords = mask_polygon.get("points_pixel")
        if not polygon_coords:
            raise ValueError(
                f"Object {object_id} is missing mask_polygon.points_pixel."
            )
        
        # Run extraction pipeline for this specific surface
        surface_analysis = analyze_single_surface(
            image=image, 
            bbox=(x1, y1, x2, y2), 
            polygon_coords=polygon_coords,
            material=material
        )
        
        surface_analysis["object_id"] = object_id
        surface_analysis["class"] = obj_class
        results.append(surface_analysis)
        
    return results