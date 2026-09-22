import os
import cv2
import json
from Nudge.Bulbs.state_analyzer import detect_bulb_candidates
from Nudge.Bulbs.vlm_classifier import VLMClassifier
import numpy as np

def run_reverse_pipeline(image_path, output_dir="output_results"):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    # Ensure clean contiguous array
    image = np.ascontiguousarray(image, dtype=np.uint8)

    # Stage 1: OpenCV extracts candidate glowing fixtures
    print("[1/2] Finding glowing light candidates using OpenCV...")
    # min_area=30 prevents hundreds of single-pixel specular glints from flooding Stage 2
    candidates = detect_bulb_candidates(image, threshold_value=225, min_area=30, pad_pixels=35)
    print(f"OpenCV found {len(candidates)} candidate light sources.")

    if not candidates:
        print("[ALERT] No glowing regions found by OpenCV.")
        return

    # Stage 2: High-res VLM batch verification
    # Stage 2: High-res VLM batch verification
    print("[2/2] Launching VLM verification...")
    vlm = VLMClassifier()

    print(f"[PIPELINE] Cropping {len(candidates)} candidate regions from image...", end=" ", flush=True)
    crops = [image[c["bbox"][1]:c["bbox"][3], c["bbox"][0]:c["bbox"][2]] for c in candidates]
    print("Done.")

    vlm_results = vlm.verify_crops_batch(crops)

    annotated = image.copy()
    verified_bulbs = []

    print("[PIPELINE] Annotating results and parsing verdicts...")
    for idx, (cand, (is_bulb, conf, label)) in enumerate(zip(candidates, vlm_results), start=1):
        x1, y1, x2, y2 = cand["bbox"]
        status = "CONFIRMED_BULB" if is_bulb else "REJECTED_ARTIFACT"
        color = (0, 255, 0) if is_bulb else (0, 0, 255)

        # Log individual candidate verdict in terminal
        tag_icon = "✓ BULB" if is_bulb else "✗ REJECT"
        print(f"  [{idx:02d}/{len(candidates):02d}] {tag_icon} | Conf: {conf*100:5.1f}% | Box: [{x1}, {y1}, {x2}, {y2}]")

        verified_bulbs.append({
            "id": idx,
            "status": status,
            "vlm_confidence": round(conf, 4),
            "label": label,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"Bulb ({conf*100:.0f}%)" if is_bulb else "Glare"
        cv2.putText(annotated, f"#{idx} {tag}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    confirmed_count = sum(1 for b in verified_bulbs if b["status"] == "CONFIRMED_BULB")
    rejected_count = len(verified_bulbs) - confirmed_count

    if confirmed_count == 0:
        print("[ALERT] No active light bulbs were detected in the image.")
        banner_text = "STATUS: NO BULBS DETECTED"
        cv2.rectangle(annotated, (10, 10), (420, 55), (0, 0, 0), -1)
        cv2.putText(annotated, banner_text, (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    report = {
        "file": os.path.basename(image_path),
        "status": "BULBS_FOUND" if confirmed_count > 0 else "NO_BULBS_DETECTED",
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
    TARGET_IMAGE = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\input_images\image1.png"
    run_reverse_pipeline(TARGET_IMAGE)