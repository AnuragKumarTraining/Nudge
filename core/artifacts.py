"""Where each stage writes / reads its JSON artifact (one file per capture_id)."""
import json
import os
def artifact_path(out_dir: str, capture_id: str | None, suffix: str) -> str:
    return os.path.join(out_dir, f"{capture_id or 'capture'}_{suffix}.json")
def save_artifact(out_dir: str, capture_id: str | None, suffix: str, data: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = artifact_path(out_dir, capture_id, suffix)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path
def load_artifact(path: str) -> dict:
    with open(path) as f:
        return json.load(f)