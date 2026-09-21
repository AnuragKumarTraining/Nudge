"""
Thin wrapper around the VLM (OpenAI-compatible endpoint - NVIDIA NIM by default, see
config/settings.py). It is used only for JUDGMENT questions that YOLO / OpenCV cannot
answer:
  * get_scene_inventory()              - discovery branch 3: objects the detectors missed
  * get_condition_and_cleanliness()    - per object: material + condition + cleanliness
  * confirm_stain()                    - is this OpenCV candidate a real stain?
  * describe_layout()                  - natural-language description of the arrangement
If no API key is set, or a call fails, every method degrades to a safe default so the
rest of the pipeline still runs end-to-end.
"""
import base64
import os
import json
import re
import cv2
import numpy as np

from openai import OpenAI

from config.settings import (
    VLM_PROVIDER, VLM_API_KEY_ENV, VLM_BASE_URL, VLM_MODEL, VLM_TEMPERATURE,
    VLM_INVENTORY_MAX_TOKENS, VLM_CONDITION_MAX_TOKENS, VLM_DEBUG,
    VLM_EXTRA_BODY, VLM_IMAGE_URL_FORMAT,
)

_MAX_SIDE = 1280      # downscale big photos before upload (cost + latency)
_MIN_SIDE = 96        # upscale tiny crops (stain candidates are often ~20px)

_MATERIALS = {"wood", "fabric", "ceramic", "metal", "glass", "plastic", "stone", "tile",
              "paint", "leather", "paper", "unknown"}
_CONDITIONS = {"good", "worn", "damaged", "broken", "missing_parts", "unclear"}
_CLEANLINESS = {"clean", "slightly_dirty", "dirty", "unclear"}


def _to_unit_bbox(bbox, img_w: int, img_h: int) -> list[float] | None:
    """
    VLMs disagree on box coordinates: 0-1 fractions, 0-1000 grid (Qwen-style), or raw
    pixels. The old code accepted ONLY 0-1 and silently dropped everything else, which
    makes the VLM branch return 0 objects. Normalise all three here.
    """
    try:
        v = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return None
    
    if len(v) != 4:
        return None
        
    m = max(v)
    if m <= 1.0:
        x1, y1, x2, y2 = v
    elif m <= 1000.0:
        x1, y1, x2, y2 = (c / 1000.0 for c in v)
    else:
        x1, y1, x2, y2 = v[0] / img_w, v[1] / img_h, v[2] / img_w, v[3] / img_h
        
    x1, y1, x2, y2 = (min(1.0, max(0.0, c)) for c in (x1, y1, x2, y2))
    
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 1e-4:
        return None
        
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _extract_json(text: str, opener: str):
    """Pull the first JSON list ('[') or object ('{') out of a reply, fences or not."""
    closer = "]" if opener == "[" else "}"
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class VLMClient:
    def __init__(self):
        api_key = os.getenv(VLM_API_KEY_ENV)
        self._client = OpenAI(api_key=api_key, base_url=VLM_BASE_URL) if api_key else None
        
        if VLM_DEBUG:
            print(f"[VLM] provider={VLM_PROVIDER} model={VLM_MODEL} base_url={VLM_BASE_URL} "
                  f"configured={self._client is not None}")

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > _MAX_SIDE:
            scale = _MAX_SIDE / max(h, w)
        elif min(h, w) < _MIN_SIDE:
            scale = _MIN_SIDE / min(h, w)
            
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def _encode_data_uri(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode image crop to JPEG")
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _build_image_content(data_uri: str) -> dict:
        """
        NVIDIA NIM (and the strict OpenAI spec) expect the nested form:
            {"type": "image_url", "image_url": {"url": "..."}}
        Ollama's own docs show a flat string instead:
            {"type": "image_url", "image_url": "..."}
        VLM_IMAGE_URL_FORMAT (set per-provider in config/settings.py) picks
        which one gets sent.
        """
        if VLM_IMAGE_URL_FORMAT == "flat":
            return {"type": "image_url", "image_url": data_uri}
        return {"type": "image_url", "image_url": {"url": data_uri}}

    def _ask(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int = 100,
        json_mode: bool = False,
    ) -> str | None:
        if self._client is None:
            return None
            
        kwargs = dict(
            model=VLM_MODEL,
            max_tokens=max_tokens,
            temperature=VLM_TEMPERATURE,
            messages=[{"role": "user", "content": [
                self._build_image_content(self._encode_data_uri(image)),
                {"type": "text", "text": prompt},
            ]}],
        )

        # Force JSON constraint to prevent rambling / endless reasoning
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        if VLM_EXTRA_BODY:
            kwargs["extra_body"] = VLM_EXTRA_BODY
            
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            print(f"VLM request failed ({VLM_PROVIDER}:{VLM_MODEL} @ {VLM_BASE_URL}): {exc}")
            return None

        # print("coming response: ",response)

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        
        if VLM_DEBUG:
            print(f"[VLM] finish={choice.finish_reason} reply={content[:300]!r}")
            
        if not content:
            print(f"[VLM] empty reply (finish_reason={choice.finish_reason}) - "
                  f"raise max_tokens or check the model's thinking switch")
            return None
            
        return content

    # ---- the four judgment features (same prompts/logic as before the swap) ----

    def get_scene_inventory(self, full_image: np.ndarray, known_summary: str) -> list[dict]:
        """
        CHECK 4 & 5: VLM Semantic Discovery step. Requests text object names only.
        """
        if self._client is None:
            print("[VLM] ❌ Client not configured")
            return []

        prompt = (
            "You are a meticulous visual inspector examining a photo. "
            "Your task is to find every objects, you can check .\n\n"
     
            "LOOK CAREFULLY FOR THESE CATEGORIES, IN THIS ORDER:\n"
            "1. Small items, accessories, and tools (e.g., utensils, toiletries, stationery, kitchenware, loose objects).\n"
            "2. Decor and furnishings (e.g., plants, artwork, mirrors, rugs, curtains, small furniture).\n"
            "3. Room structure and fixed elements (e.g., floor, wall, ceiling, window, door, counters, cabinetry, shelving).\n"
            "4. Equipment, appliances, and fixtures (e.g., lighting, HVAC, electronics, plumbing fixtures, bins, outlets).\n"
            "5. People and personal belongings (only if clearly visible: person, bags, clothing items, phones).\n\n"
            "STRICT RULES - FOLLOW EXACTLY:\n"
            "- Only report objects you clearly see.\n"
            "- Do not repeat objects from the ALREADY DETECTED list.\n"
            "- Use short, singular, lowercase class names (e.g., 'spoon', 'napkin').\n"
            "- DO NOT output coordinates, bounding boxes, or spatial locations.\n"
            "- Do not include reasoning or markdown formatting.\n\n"
            "Reply with JSON ONLY in this format:\n"
            '{"objects": [{"class": "spoon", "count": 2}, {"class": "napkin", "count": 1}]}\n'
            'If no new objects are seen, reply with {"objects": []}.'
        )

        text = self._ask(full_image, prompt, max_tokens=VLM_INVENTORY_MAX_TOKENS, json_mode=True)
        if not text:
            return []

        # CHECK 4: Extract JSON safely
        data = _extract_json(text, "{")
        items = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []

        # CHECK 5: Label normalization
        cleaned_items = []
        for item in items:
            try:
                raw_cls = str(item.get("class", "")).strip().lower()
                # Basic string sanity checks
                raw_cls = re.sub(r'[^a-z0-9_\s-]', '', raw_cls)
                if not raw_cls or len(raw_cls) < 2:
                    continue

                cleaned_items.append({
                    "class": raw_cls,
                    "count": max(1, int(item.get("count", 1)))
                })
            except Exception:
                continue

        return cleaned_items

    def get_condition_and_cleanliness(self, crop: np.ndarray, object_class: str) -> dict:
        """ONE call per object -> material + condition + cleanliness + issues."""
        default = {"material": "unknown", "condition": "unclear", "cleanliness": "unclear",
                   "issues": []}
        if self._client is None:
            return default

        prompt = (
             f'This is a cropped photo of a "{object_class}" in a cafe. Judge ONLY what is '
             "visible in the crop.\n"
             "CRITICAL INSTRUCTION: DO NOT output any thinking, reasoning, or explanations. "
             "Reply with JSON ONLY:\n"
             '{"material": "wood|fabric|ceramic|metal|glass|plastic|stone|tile|paint|leather|paper|unknown", '
             '"condition": "good|worn|damaged|broken|missing_parts|unclear", '
             '"cleanliness": "clean|slightly_dirty|dirty|unclear", '
             '"issues": ["short phrases such as \'chipped rim\', \'coffee stain\'; empty list if none"]}'
        )

        text = self._ask(crop, prompt, max_tokens=VLM_CONDITION_MAX_TOKENS, json_mode=True)
        data = _extract_json(text, "{") if text else None
        
        if not isinstance(data, dict):
            return default
            
        def pick(key: str, allowed: set) -> str:
            val = str(data.get(key, "")).strip().lower().replace(" ", "_")
            return val if val in allowed else default[key]
            
        issues = data.get("issues", [])
        return {
            "material": pick("material", _MATERIALS),
            "condition": pick("condition", _CONDITIONS),
            "cleanliness": pick("cleanliness", _CLEANLINESS),
            "issues": [str(i)[:80] for i in issues][:5] if isinstance(issues, list) else [],
        }

    def confirm_stain(self, crop: np.ndarray) -> str:
        prompt = (
             "This is a small cropped region from a cafe surface (table, chair, floor, "
             "dishware). Is the highlighted spot a real stain/dirt/mark, or just a pattern, "
             "texture, reflection or shadow? Reply with ONLY one word: "
             "stain, pattern, shadow, or unclear."
        )
        # No JSON mode here, as we only expect 1 word back.
        text = self._ask(crop, prompt, max_tokens=VLM_CONDITION_MAX_TOKENS, json_mode=False)
        if not text:
            return "unclear"
        m = re.search(r"\b(stain|pattern|shadow|unclear)\b", text.lower())
        return m.group(1) if m else "unclear"

    def describe_layout(self, full_image: np.ndarray, object_summary: str) -> str:
        prompt = (
            "Here is a photo of a cafe table setup. Objects detected by an object "
            f"detector: {object_summary}. In 1-2 plain sentences, describe the "
            "arrangement and note anything that looks incomplete, misplaced or unusual."
        )
        # Natural language generation (no JSON)
        text = self._ask(full_image, prompt, max_tokens=300, json_mode=False)
        return text if text else "pending_vlm"