# Nudge — Stage 3 Feature Extraction Pipeline

Restructured, fixed, and extended version of the `shash` branch. Verified
end-to-end against the repo's own `images/daily_2.jpg` + `metadata/daily_image.json`
— see "Verified working" below.

## Pipeline (current) - ONE image + its input JSON, nothing else
```
 image ─► quality gate ─► lighting (CLAHE if dim, from ambient_lux)          pipeline/preprocess.py
        │
        ├─ STAGE A  OBJECT DISCOVERY                                          pipeline/discovery.py
        │    YOLOv8 ┐
        │    YOLO-World ├─► 3-way merge + dedup ─► inventory ─► bboxes · polygons · relationships
        │    VLM inventory ┘                     ──► outputs/<capture_id>_object_discovery.json
        │
        └─ STAGE B  CONDITION                                                 pipeline/condition_analysis.py
             condition VLM ─► condition + cleanliness (+ material, issues)
             surface / stain ─► OpenCV candidates ─► VLM verdict (stain / pattern / shadow)
                                                 ──► outputs/<capture_id>_condition_analysis.json
 final JSON ──► outputs/<capture_id>_final.json
```
There is no master/reference image anywhere in this flow. `preprocessing/alignment.py`
is kept only for a possible future "diff against a stored setup" feature and is not imported.
`master_reference_id` and `spatial_alignment` in the input JSON are ignored.
```bash
python main.py                                   # whole pipeline
python main.py --stage discovery                 # Stage A only
python main.py --stage condition --discovery-json outputs/<id>_object_discovery.json   # Stage B only
python tests/test_pipeline_no_master.py          # regression tests (fakes for YOLO + VLM)
```

## What was broken in the original branch (fixed here)

- `main.py` and `requirements.txt` had **unresolved git merge conflicts**
  (`<<<<<<< Updated upstream` markers left in the file — neither would run/install as-is).
- `utils/model.py` was **completely empty**, but `main.py` and `master.py` both
  `import load_yolo` from it. That import would crash immediately.
- `master.py` imported `config.master_config`, which didn't exist anywhere in the repo.
- `utils/features.py`, `utils/brightness.py`, `utils/matcher.py` were empty stubs.
- `quality_gate` boolean silently serialized as the *string* `"True"`/`"False"`
  via numpy bool leaking into `json.dump` — fixed by explicit `bool()` cast
  (this would have broken `if not quality["is_passed"]` checks after any
  JSON round-trip, since non-empty strings are truthy).

## Project structure

```
nudge_pipeline/
├── config/settings.py          # every threshold/class-list lives here, nowhere else
├── core/schemas.py             # documents the JSON contract between stages
├── models/
│   ├── yolo_loader.py          # the load_yolo() that was missing
│   └── vlm_client.py           # material_hint / condition / stain confirm / layout via Qwen-VL (DashScope)
├── preprocessing/
│   ├── quality_gate.py         # blur check
│   ├── lighting.py             # CLAHE normalization
│   └── alignment.py            # ORB homography + gyro pre-rotation
├── object_detection/
│   ├── detector.py             # object identity, count, bbox (+ open-vocab hook)
│   ├── polygon.py              # polygon + orientation per object
│   └── relationships.py        # "belongs_to" / relative-position grouping
├── surface_condition/
│   ├── masking.py              # exposed-surface crop (table minus objects on it)
│   └── stain_detection.py      # stain/mark candidates
├── semantic/
│   └── enrich.py                # VLM enrichment: material_hint, condition, stain confirm, layout
├── pipeline/
│   └── run_stage3.py            # orchestrates all of the above for one capture
├── main.py                      # CLI entry point
└── requirements.txt
```

## How the features map to functions

| Feature you asked for | Function | File |
|---|---|---|
| Object identity, count, bbox | `detect_objects()` | `object_detection/detector.py` |
| Polygon | `extract_polygon()` | `object_detection/polygon.py` |
| Stain / mark candidates | `detect_stain_candidates()` | `surface_condition/stain_detection.py` |
| Material hint | `get_material_hint()` via `enrich_objects_with_vlm()` | `models/vlm_client.py`, `semantic/enrich.py` |
| Object condition | `get_object_condition()` via `enrich_objects_with_vlm()` | `models/vlm_client.py`, `semantic/enrich.py` |
| Layout description | `describe_layout()` via `generate_layout_description()` | `models/vlm_client.py`, `semantic/enrich.py` |

## Running it

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-...   # optional — without it, material_hint/condition/
                                   # layout_description/stain vlm_verdict all degrade
                                   # gracefully instead of crashing (see semantic/enrich.py)
python main.py
```

VLM provider is Qwen-VL, called through Alibaba Cloud DashScope's
OpenAI-compatible endpoint (`models/vlm_client.py`). Default model is
`qwen3-vl-plus` and default region is Singapore/international
(`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`) — override either
with the `QWEN_VL_MODEL` or `DASHSCOPE_BASE_URL` environment variables (see
`config/settings.py`; Beijing and Virginia base URLs are noted there too).

Output goes to `metadata/stage_3.json`.

## Verified working

Ran against the repo's own test assets (`images/daily_2.jpg`, `images/master.jpg`,
`metadata/daily_image.json`, no `DASHSCOPE_API_KEY` set):

- Quality gate passed (`blur_variance: 865.25`)
- Alignment correctly **flagged** the capture: `yaw_delta_deg=146.7` in the sample
  metadata exceeds the configured warning threshold, and ORB homography did in
  fact fail (`only_18_good_matches_need_20`) — the warning system caught a real
  problem case, not a hypothetical one.
- 13 objects detected (chairs, people, potted plants), each with bbox, polygon,
  material_hint, and relationships populated.
- Stain candidates detected on the chair surface, correctly downweighted
  (`raw_confidence: 0.16`) because they were flagged as periodic patterns —
  the false-positive suppression is working as designed.
- With no API key set, `layout_description`, `condition`, and `vlm_verdict`
  all cleanly fell back to their pending/null defaults instead of crashing.

## Step-by-step: adding a new feature to this architecture

Say you want to add a new feature — for example, "flower freshness" (from the
decor-condition feature we discussed).

1. **Decide which category it belongs to** (see the feature/source table):
   is it a *measurement* (→ OpenCV, lives in `preprocessing/` or
   `surface_condition/`), a *judgment* (→ VLM, lives in `semantic/enrich.py`
   + a new method on `VLMClient`), or a *localization* (→ `object_detection/`)?
   Flower freshness is judgment → it belongs in `semantic/`.

2. **Add the method to `VLMClient`** (`models/vlm_client.py`), following the
   exact pattern of `get_object_condition()` — same `_ask()` call, same
   graceful-degradation return value if `self._client is None`.

3. **Add a config entry if it needs one** (`config/settings.py`) — e.g. which
   object classes this check applies to (`vase`, `potted plant`).

4. **Wire it into `semantic/enrich.py`** — add a loop similar to
   `enrich_objects_with_vlm()` that calls your new `VLMClient` method and
   writes the result onto the object dict.

5. **Call it from `pipeline/run_stage3.py`** — one line, alongside the
   existing `enrich_objects_with_vlm()` call.

6. **Add the field to `core/schemas.py`** so the contract stays documented.

7. **Test it in isolation first** — write a small script that loads one
   image, crops one object, and calls just your new function directly,
   before wiring it into the full pipeline. This is exactly how
   `detect_stain_candidates()` and `extract_polygon()` were validated above.

The rule that keeps this extensible: **every new feature is a new function in
exactly one file, called from exactly one place in `run_stage3.py`.** Nothing
should ever require touching more than 2-3 files to add one feature.

## Known gaps / next steps

- **Floors, windows, and other non-COCO classes**: `yolov8n.pt` (COCO) has no
  classes for these — this is exactly the gap you flagged. See
  `detect_open_vocabulary_objects()` in `object_detection/detector.py` for the
  YOLO-World-based extensibility hook; it's written but not wired into the
  default pipeline yet — validate its accuracy on real cafe photos before
  switching it on.
- **Polygon precision**: `extract_polygon()` uses Otsu-threshold contours
  within each bbox, which is an approximation, not a true segmentation mask.
  Upgrading `YOLO_WEIGHTS_PATH` to a `-seg` variant (e.g. `yolov8n-seg.pt`)
  gives real instance masks directly from the model — see the docstring in
  `object_detection/polygon.py`.
- **Master reference storage**: this pipeline processes one capture; it does
  not yet persist a `FeatureBundle` to a database or diff against a stored
  master. That's the `storage/` and `comparison/` layers from the earlier
  architecture discussion — not built here yet.
