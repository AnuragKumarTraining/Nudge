'''
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
'''

# services/inventory_service.py
import numpy as np

from config.settings import SURFACE_RELEVANT_CLASSES, MATERIAL_HINT_BY_CLASS
from models.vlm_client import VLMClient
from models.grounding_service import GroundingService
from utils.box_utils import compute_iou, apply_nms, validate_and_normalize_bbox


def detect_vlm_inventory(
    image: np.ndarray,
    objects: list[dict],
    vlm_client: VLMClient,
    grounding_service: GroundingService,
    iou_conflict_threshold: float = 0.40,
    grounding_confidence_threshold: float = 0.30
) -> list[dict]:

    print("\n" + "=" * 70)
    print("[VLM INVENTORY] Starting Decoupled VLM + Grounding Pipeline")
    print("=" * 70)

    # ---------------------------------------------------------
    # CHECK 1: Client Configuration
    # ---------------------------------------------------------
    if not vlm_client.is_configured:
        print("[VLM INVENTORY] ❌ VLM client not configured. Skipping.")
        return []

    if grounding_service is None:
        print("[VLM INVENTORY] ❌ Grounding service not initialized. Skipping.")
        return []

    # ---------------------------------------------------------
    # CHECK 2: Input Image Verification
    # ---------------------------------------------------------
    if image is None or image.size == 0 or len(image.shape) != 3:
        print("[VLM INVENTORY] ❌ Invalid image passed to pipeline.")
        return []

    h, w = image.shape[:2]

    # ---------------------------------------------------------
    # CHECK 3: Known Objects Pre-Filter
    # ---------------------------------------------------------
    known_classes = sorted({
        str(o.get("class", "")).strip().lower()
        for o in objects
        if o.get("class")
    })
    known_summary = ", ".join(known_classes) if known_classes else "None"
    print(f"[VLM INVENTORY] Known YOLO classes ({len(known_classes)}): {known_summary}")

    # Extract normalized bboxes from existing YOLO objects for later IoU checks
    existing_yolo_boxes = []
    for o in objects:
        if "bbox_normalized" in o:
            nb = o["bbox_normalized"]
            box = [nb.get("x1"), nb.get("y1"), nb.get("x2"), nb.get("y2")]
            val_box = validate_and_normalize_bbox(box, w, h)
            if val_box:
                existing_yolo_boxes.append(val_box)

    # ---------------------------------------------------------
    # STEP 1: Semantic VLM Discovery
    # ---------------------------------------------------------
    print("[VLM INVENTORY] 🚀 Step 1: Asking VLM for missing text classes...")
    discovered_items = vlm_client.get_scene_inventory(image, known_summary)

    if not discovered_items:
        print("[VLM INVENTORY] ⚠️ VLM found 0 new classes.")
        return []

    # Extract unique class strings found by VLM
    candidate_classes = list({item["class"] for item in discovered_items})
    print(f"[VLM INVENTORY] ✅ VLM discovered {len(candidate_classes)} candidate classes: {candidate_classes}")

    # ---------------------------------------------------------
    # STEP 2: Zero-Shot Grounding Localization
    # ---------------------------------------------------------
    print("[VLM INVENTORY] 🚀 Step 2: Running Grounding DINO to find coordinates...")
    raw_grounded_detections = grounding_service.detect_objects(
        image_bgr=image,
        candidate_classes=candidate_classes,
        box_threshold=grounding_confidence_threshold
    )

    if not raw_grounded_detections:
        print("[VLM INVENTORY] ⚠️ Grounding model could not locate candidate classes.")
        return []

    print(f"[VLM INVENTORY] Raw grounded detections found: {len(raw_grounded_detections)}")

    # ---------------------------------------------------------
    # CHECK 11: Non-Maximum Suppression across grounded items
    # ---------------------------------------------------------
    nms_detections = apply_nms(raw_grounded_detections, iou_threshold=0.50)

    # ---------------------------------------------------------
    # FILTERING & CONSTRUCTING FINAL PIPELINE OBJECTS
    # ---------------------------------------------------------
    final_inventory = []
    dropped_iou_count = 0

    for i, det in enumerate(nms_detections):
        cls = det["class"]
        box_norm = det["bbox_normalized"]  # Guaranteed [x1, y1, x2, y2] in 0..1
        confidence = det["confidence"]

        # ---------------------------------------------------------
        # CHECK 10: IoU Overlap against existing YOLO objects
        # ---------------------------------------------------------
        is_duplicate = False
        for yolo_box in existing_yolo_boxes:
            if compute_iou(box_norm, yolo_box) >= iou_conflict_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            dropped_iou_count += 1
            print(f"[VLM INVENTORY] ✂️ Dropped '{cls}' - conflicts with existing YOLO object (IoU >= {iou_conflict_threshold})")
            continue

        # ---------------------------------------------------------
        # CHECK 12: Build Final Schema Safe Object
        # ---------------------------------------------------------
        x1, y1, x2, y2 = box_norm

        obj_dict = {
            "object_id": f"obj_vlm_{i:03d}_{cls.replace(' ', '_')}",
            "class": cls,
            "confidence": round(confidence, 3),

            # Absolute integer coordinates
            "bbox": {
                "x1": int(x1 * w),
                "y1": int(y1 * h),
                "x2": int(x2 * w),
                "y2": int(y2 * h),
            },

            # Normalized float coordinates
            "bbox_normalized": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },

            # Centroid calculations
            "centroid": {
                "x": int((x1 + x2) * w / 2),
                "y": int((y1 + y2) * h / 2),
            },

            "reported_count": 1,
            "surface_relevant": cls in SURFACE_RELEVANT_CLASSES,
            "material_hint": MATERIAL_HINT_BY_CLASS.get(cls, "pending_vlm"),
            "condition": "pending_vlm",
            "cleanliness": "pending_vlm",
            "relationships": {},
            "source": "vlm_grounded",
        }

        final_inventory.append(obj_dict)

    print("-" * 70)
    print(f"[VLM INVENTORY] ✅ Final pipeline inventory: {len(final_inventory)} items "
          f"(Dropped IoU duplicates: {dropped_iou_count})")
    print("=" * 70)

    return final_inventory