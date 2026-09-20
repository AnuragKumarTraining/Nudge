import json
import os

from pipeline.run_stage3 import run_stage3_pipeline

def main():
    # Hardcoded to only process the daily_2 image
    image_path = os.path.join("images", "daily_2.jpg")
    
    # Adjusted metadata paths to logically match the daily_2 image
    metadata_path = os.path.join("metadata", "daily_image.json") 
    output_path = os.path.join("metadata", "daily_2_stage_3.json")

    # Safety check: ensure the image actually exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}. Halting execution.")
        return

    # Load metadata safely
    capture_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            capture_metadata = json.load(f)
        print(f"[INFO] Loaded metadata for capture_id={capture_metadata.get('capture_id', 'Unknown')}")
    else:
        print(f"[WARNING] Metadata file not found at {metadata_path}. Proceeding with empty metadata.")

    master_image_path = os.path.join("images", "master.jpg")
    master_image_path = master_image_path if os.path.exists(master_image_path) else None

    # Execute the pipeline
    result = run_stage3_pipeline(image_path, capture_metadata, master_image_path)

    # Save the output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[SUCCESS] Stage 3 output saved to {output_path}")
    print(f"[INFO] status={result.get('status')} objects_found={len(result.get('objects', []))} "
          f"extractor={result.get('extractor_used')}")

if __name__ == "__main__":
    main()