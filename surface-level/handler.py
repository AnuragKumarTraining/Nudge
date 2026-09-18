import json
from process_payload import process_surface_payload

def main():
    # Paths to your local test files
    image_path = "C:\\Users\\anjishnu.kumbhakar\\OneDrive - JK Technosoft Ltd\\Desktop\\nudge\\Nudge\\test_input\\cap_690205.jpg"
    payload_path = "C:\\Users\\anjishnu.kumbhakar\\OneDrive - JK Technosoft Ltd\\Desktop\\nudge\\Nudge\\test_input\\input.json"

    # Read the JSON payload file
    with open(payload_path, "r") as f:
        payload_data = json.load(f)

    print("Running surface extraction pipeline across modules...")
    
    # Trigger Step 1 (which cascades through steps 2, 3, 4, and 5)
    extracted_features = process_surface_payload(payload_data, image_path)

    # Print nicely formatted JSON output
    output = {
        "capture_id": payload_data.get("capture_id", "cap_test"),
        "surface_analysis": extracted_features
    }
    
    print("\n--- EXTRACTION COMPLETE ---")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()