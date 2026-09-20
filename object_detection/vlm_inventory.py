import numpy as np

from config.settings import SURFACE_RELEVANT_CLASSES, MATERIAL_HINT_BY_CLASS
from models.vlm_client import VLMClient


def detect_vlm_inventory(
    image: np.ndarray,
    objects: list[dict],
    vlm_client: VLMClient,
) -> list[dict]:

    print("\n" + "=" * 70)
    print("[VLM INVENTORY] Starting VLM scene/object inventory")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Check whether VLM client is configured
    # ---------------------------------------------------------
    if not vlm_client.is_configured:
        print("[VLM INVENTORY] ❌ VLM client is NOT configured")
        print("[VLM INVENTORY] Check NVIDIA_API_KEY / VLM_API_KEY_ENV")
        print("[VLM INVENTORY] Skipping VLM inventory")
        print("=" * 70)
        return []

    print("[VLM INVENTORY] ✅ VLM client is configured")

    # ---------------------------------------------------------
    # 2. Image information
    # ---------------------------------------------------------
    if image is None:
        print("[VLM INVENTORY] ❌ Image is None")
        return []

    h, w = image.shape[:2]

    print(f"[VLM INVENTORY] Image size: {w} x {h}")
    print(f"[VLM INVENTORY] Existing detector objects: {len(objects)}")

    # ---------------------------------------------------------
    # 3. Build known object list
    # ---------------------------------------------------------
    known = ", ".join(
        sorted({
            str(o.get("class", "")).strip()
            for o in objects
            if o.get("class")
        })
    )

    print(f"[VLM INVENTORY] Known classes passed to VLM: {known}")

    # ---------------------------------------------------------
    # 4. Call VLM
    # ---------------------------------------------------------
    print("[VLM INVENTORY] 🚀 Calling VLM get_scene_inventory()...")
    print("[VLM INVENTORY] Waiting for NVIDIA model response...")

    try:
        items = vlm_client.get_scene_inventory(
            image,
            known,
        )
    except Exception as exc:
        print("[VLM INVENTORY] ❌ Exception while calling VLM")
        print(f"[VLM INVENTORY] {type(exc).__name__}: {exc}")
        return []

    # ---------------------------------------------------------
    # 5. Check raw result
    # ---------------------------------------------------------
    if items is None:
        print("[VLM INVENTORY] ❌ VLM returned None")
        print("[VLM INVENTORY] This usually means:")
        print("    - API request failed")
        print("    - model returned empty response")
        print("    - response could not be parsed")
        print("    - VLM client was unable to process the response")
        print("=" * 70)
        return []

    if not items:
        print("[VLM INVENTORY] ⚠️ VLM call completed, but returned 0 items")
        print("[VLM INVENTORY] This means the call may have succeeded,")
        print("[VLM INVENTORY] but the parsed inventory was empty.")
        print("=" * 70)
        return []

    print(f"[VLM INVENTORY] ✅ VLM returned {len(items)} candidate items")

    # ---------------------------------------------------------
    # 6. Convert VLM items into pipeline objects
    # ---------------------------------------------------------
    inventory = []

    for i, item in enumerate(items):

        try:
            x1, y1, x2, y2 = item["bbox_normalized"]
            cls = str(item["class"]).strip().lower()

            print(
                f"[VLM INVENTORY] Item {i + 1}: "
                f"class='{cls}', "
                f"count={item.get('count', 1)}, "
                f"bbox={[x1, y1, x2, y2]}"
            )

            inventory.append({
                "object_id": (
                    f"obj_vlm_{i:03d}_{cls.replace(' ', '_')}"
                ),
                "class": cls,
                "confidence": 0.5,

                "bbox": {
                    "x1": int(x1 * w),
                    "y1": int(y1 * h),
                    "x2": int(x2 * w),
                    "y2": int(y2 * h),
                },

                "bbox_normalized": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "centroid": {
                    "x": int((x1 + x2) * w / 2),
                    "y": int((y1 + y2) * h / 2),
                },

                "reported_count": item.get("count", 1),

                "surface_relevant": (
                    cls in SURFACE_RELEVANT_CLASSES
                ),

                "material_hint": MATERIAL_HINT_BY_CLASS.get(
                    cls,
                    "pending_vlm",
                ),

                "condition": "pending_vlm",
                "relationships": {},
                "source": "vlm_inventory",
            })

        except Exception as exc:
            print(
                f"[VLM INVENTORY] ⚠️ Failed to process item {i}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"[VLM INVENTORY] Item data: {item}")

    # ---------------------------------------------------------
    # 7. Final result
    # ---------------------------------------------------------
    print("-" * 70)
    print(
        f"[VLM INVENTORY] ✅ Final inventory objects: "
        f"{len(inventory)}"
    )

    if inventory:
        print("[VLM INVENTORY] Detected classes:")
        for obj in inventory:
            print(
                f"    - {obj['class']} "
                f"(confidence={obj['confidence']}, "
                f"bbox={obj['bbox']})"
            )
    else:
        print("[VLM INVENTORY] ⚠️ No usable VLM objects survived")

    print("=" * 70)

    return inventory