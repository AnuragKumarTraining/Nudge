import os
import json
import argparse
from config.settings import OUTPUT_DIR
from pipeline.run_stage3 import run_stage3_pipeline, run_condition_from_artifact


def _load_metadata(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
        print(f"[INFO] Loaded metadata for capture_id={meta.get('capture_id', 'Unknown')}")
        return meta
    print(f"[WARNING] Metadata file not found at {path}. Proceeding with empty metadata.")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join("images", "master.jpg"))
    ap.add_argument("--metadata", default=os.path.join("metadata", "daily_image.json"))
    ap.add_argument("--out-dir", default=OUTPUT_DIR)
    ap.add_argument("--stage", choices=["all", "discovery", "condition"], default="all")
    ap.add_argument("--discovery-json", help="required for --stage condition")
    args = ap.parse_args()

    metadata = _load_metadata(args.metadata)

    if args.stage == "condition":
        if not args.discovery_json:
            ap.error("--stage condition needs --discovery-json <path to *_object_discovery.json>")
        result = run_condition_from_artifact(args.discovery_json, metadata, args.image, args.out_dir)

    elif args.stage == "discovery":
        from pipeline.discovery import run_object_discovery
        from pipeline.run_stage3 import _read_image
        from models.vlm_client import VLMClient
        from models.grounding_service import GroundingService

        vlm_client = VLMClient()
        try:
            grounding_service = GroundingService()
        except Exception as e:
            print(f"[CLI] ⚠️ GroundingService disabled: {e}")
            grounding_service = None

        result = run_object_discovery(
            _read_image(args.image), 
            metadata, 
            vlm_client=vlm_client,
            grounding_service=grounding_service,
            image_path=args.image,
            out_dir=args.out_dir
        )
        print(f"[SUCCESS] discovery artifact: {result['artifact_path']}")
        print(f"[INFO] sources={result['sources']} total_objects={result['total_objects']} "
              f"counts={result['object_counts']}")
        return

    else:
        result = run_stage3_pipeline(args.image, metadata, args.out_dir)

    if result.get("status") == "rejected":
        print(f"[REJECTED] capture not processed: reason={result.get('reason')} {result.get('quality_gate')}")
        return
        
    print(f"[SUCCESS] {result['artifacts']}")
    print(f"[INFO] status={result['status']} objects={result['total_objects']} "
          f"extractor={result['extractor_used']}")


if __name__ == "__main__":
    main()