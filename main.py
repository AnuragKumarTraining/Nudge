import json
import os

from pipeline.run_stage3 import run_stage3_pipeline


def main():
    image_path = os.path.join("images", "daily_2.jpg")
    metadata_path = os.path.join("metadata", "daily_image.json")
    output_path = os.path.join("metadata", "stage_3.json")

    with open(metadata_path) as f:
        capture_metadata = json.load(f)
    print(f"[INFO] Loaded metadata for capture_id={capture_metadata.get('capture_id')}")

    master_image_path = os.path.join("images", "master.jpg")
    master_image_path = master_image_path if os.path.exists(master_image_path) else None

    result = run_stage3_pipeline(image_path, capture_metadata, master_image_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[SUCCESS] Stage 3 output saved to {output_path}")
    print(f"[INFO] status={result.get('status')} objects_found={len(result.get('objects', []))} "
          f"extractor={result.get('extractor_used')}")


if __name__ == "__main__":
    main()
