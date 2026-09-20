from __future__ import annotations
import math
from config.inference_config import MAX_DRIFT_PIXELS
from pipeline_helper.context import PipelineContext

def run(ctx: PipelineContext) -> bool:
    """STEP 4: Delta Calculations (Inventory Missing, Drift, Clutter)."""
    master_objs = ctx.master_data.get("objects", [])
    current_objs = ctx.current_data.get("objects", []).copy()
    matched_master = []
    ctx.drift_alerts = []

    for m_obj in master_objs:
        best_match_idx = None
        min_dist = float("inf")

        for idx, c_obj in enumerate(current_objs):
            if m_obj["label"] == c_obj["label"]:
                dist = math.hypot(
                    m_obj["centroid"][0] - c_obj["centroid"][0],
                    m_obj["centroid"][1] - c_obj["centroid"][1]
                )
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx

        if best_match_idx is not None:
            c_obj = current_objs.pop(best_match_idx)
            matched_master.append(m_obj)
            if min_dist > MAX_DRIFT_PIXELS:
                ctx.drift_alerts.append(f"{m_obj['label'].capitalize()} shifted {int(min_dist)}px")

    ctx.missing_items = [obj["label"] for obj in master_objs if obj not in matched_master]
    ctx.clutter_items = [obj["label"] for obj in current_objs]

    print("\n--- Reset Checklist ---")
    if not ctx.missing_items:
        print("[OK] All required room items present.")
    else:
        for item in set(ctx.missing_items):
            print(f"[MISSING] {ctx.missing_items.count(item)} x {item}")

    if not ctx.drift_alerts:
        print("[OK] Furniture positions verified.")
    else:
        for alert in ctx.drift_alerts:
            print(f"[DRIFT] {alert}")

    if not ctx.clutter_items:
        print("[OK] Room is clean (No clutter detected).")
    else:
        for item in set(ctx.clutter_items):
            print(f"[CLUTTER] Remove {ctx.clutter_items.count(item)} x {item}")

    return True
