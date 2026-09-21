import os
import urllib.request

def ensure_model_exists(model_path: str, model_url: str) -> None:
    """Verifies existence of face_landmarker task model; downloads if missing."""
    if os.path.isfile(model_path):
        return

    print(f"Task model not detected. Downloading to: {model_path} ...")
    try:
        req = urllib.request.Request(model_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(model_path, "wb") as out_file:
            out_file.write(resp.read())
        print("Model downloaded successfully.")
    except Exception as exc:
        raise SystemExit(f"Failed to fetch model binary: {exc}")