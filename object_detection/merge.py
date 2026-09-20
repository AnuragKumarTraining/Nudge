def calculate_iou(box_a, box_b):

    ax1, ay1 = box_a["x1"], box_a["y1"]
    ax2, ay2 = box_a["x2"], box_a["y2"]

    bx1, by1 = box_b["x1"], box_b["y1"]
    bx2, by2 = box_b["x2"], box_b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (
        (inter_x2 - inter_x1) *
        (inter_y2 - inter_y1)
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def merge_detections(
    standard_objects,
    open_vocab_objects,
    iou_threshold=0.6
):

    merged = list(standard_objects)

    for ov_obj in open_vocab_objects:

        duplicate = False

        for existing in merged:

            if ov_obj["class"].lower() != existing["class"].lower():
                continue

            iou = calculate_iou(
                ov_obj["bbox"],
                existing["bbox"]
            )

            if iou >= iou_threshold:

                # Keep the higher-confidence detection
                if (
                    ov_obj["confidence"]
                    > existing["confidence"]
                ):
                    existing.update(ov_obj)

                duplicate = True
                break

        if not duplicate:
            merged.append(ov_obj)

    return merged