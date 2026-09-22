import os
import cv2
import json
from state_analyzer import detect_bulb_candidates
from vlm_classifier import VLMClassifier

def run_reverse_pipeline(image_path, output_dir="output_results"):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    # Stage 1: OpenCV extracts candidate glowing fixtures
    print("[1/2] Finding glowing light candidates using OpenCV...")
    candidates = detect_bulb_candidates(image, threshold_value=210, min_area=15, pad_pixels=15)
    print(f"OpenCV found {len(candidates)} candidate light sources.")

    # Stage 2: VLM verifies each crop
    print("[2/2] Verifying candidates via VLM zero-shot classifier...")
    vlm = VLMClassifier()

    annotated = image.copy()
    verified_bulbs = []

    for idx, cand in enumerate(candidates, start=1):
        x1, y1, x2, y2 = cand["bbox"]
        crop = image[y1:y2, x1:x2]

        is_bulb, conf, label = vlm.verify_crop(crop)

        status = "CONFIRMED_BULB" if is_bulb else "REJECTED_ARTIFACT"
        color = (0, 255, 0) if is_bulb else (0, 0, 255)

        verified_bulbs.append({
            "id": idx,
            "status": status,
            "vlm_confidence": round(conf, 4),
            "label": label,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

        # Draw box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"Bulb ({conf*100:.0f}%)" if is_bulb else "Glare"
        cv2.putText(annotated, f"#{idx} {tag}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    confirmed_count = sum(1 for b in verified_bulbs if b["status"] == "CONFIRMED_BULB")
    rejected_count = len(verified_bulbs) - confirmed_count

    report = {
        "file": os.path.basename(image_path),
        "total_opencv_candidates": len(candidates),
        "vlm_confirmed_bulbs": confirmed_count,
        "vlm_rejected_artifacts": rejected_count,
        "detections": verified_bulbs
    }

    out_image_path = os.path.join(output_dir, "verified_result.png")
    out_json_path = os.path.join(output_dir, "verification_report.json")

    cv2.imwrite(out_image_path, annotated)
    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Finished: {confirmed_count} confirmed bulbs, {rejected_count} rejected artifacts.")
    print(f"Results saved inside '{output_dir}'.")

if __name__ == "__main__":
    TARGET_IMAGE = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\Media.jpg"
    run_reverse_pipeline(TARGET_IMAGE)