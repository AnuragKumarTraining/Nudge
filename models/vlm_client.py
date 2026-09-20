"""
Thin wrapper around Qwen-VL, called through Alibaba Cloud DashScope's
OpenAI-compatible endpoint, for the four judgment-based features nothing in
OpenCV/YOLO can answer: material_hint, object_condition, stain confirmation,
and layout_description.

Requires `pip install openai` (DashScope's compatible mode is accessed
through the standard OpenAI SDK, just pointed at a different base_url —
no dashscope-specific package needed) and a DASHSCOPE_API_KEY environment
variable. If no key is set, every method degrades gracefully (returns None /
a safe default) so the rest of the pipeline still runs end-to-end offline
during development — same behavior as before the provider swap, and nothing
in semantic/enrich.py or pipeline/run_stage3.py had to change to make this
swap, since the public interface below is identical to the previous
Anthropic-backed client.
"""
import base64
import os

import json
import re
import cv2
import numpy as np

# from config.settings import DASHSCOPE_API_KEY_ENV, DASHSCOPE_BASE_URL, QWEN_VL_MODEL
from openai import OpenAI
from config.settings import VLM_API_KEY_ENV, VLM_BASE_URL, VLM_MODEL

class VLMClient:
    def __init__(self):
        api_key = os.getenv(VLM_API_KEY_ENV)

        self._client = None

        if api_key:
            self._client = OpenAI(
                api_key=api_key,
                base_url=VLM_BASE_URL
            )
    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @staticmethod
    def _encode_data_uri(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode image crop to JPEG")
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _ask(self, image: np.ndarray, prompt: str, max_tokens: int = 100) -> str | None:
        if self._client is None:
            return None
        data_uri = self._encode_data_uri(image)
        try:
            response = self._client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    # Keep answers short/JSON-only; thinking tokens would blow
                    # the small max_tokens budget used by material/stain prompts.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        except Exception as exc:
            print(f"VLM request failed ({VLM_MODEL} @ {VLM_BASE_URL}): {exc}")
            return None
        content = response.choices[0].message.content
        return content.strip() if content else None

    # ---- the four judgment features (same prompts/logic as before the swap) ----

    def get_scene_inventory(self, full_image: np.ndarray, known_summary: str) -> list[dict] | None:
        if self._client is None:
            return None                      # no key configured → skip branch

        prompt = (
            "List every distinct object visible in this cafe photo as JSON: "
            '[{"class": "...", "count": N, "bbox_normalized": [x1,y1,x2,y2]}] '
            "using normalized 0-1 coordinates, best-effort. Known detections (skip these): "
            f"{known_summary}. Include only additional objects. Reply with JSON only."
        )
        text = self._ask(full_image, prompt, max_tokens=800)
        if not text:
            return None                      # API/network failure → skip branch

        # ---- robust JSON extraction (models love wrapping JSON in ```json fences) ----
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None                      # reply wasn't a JSON list → skip branch

        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        # ---- validate each item; drop malformed ones, keep the survivors ----
        inventory = []
        for item in items:
            try:
                bbox = [float(v) for v in item["bbox_normalized"]]
                if len(bbox) != 4 or not all(0.0 <= v <= 1.0 for v in bbox):
                    continue
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:   # x2<=x1 or y2<=y1 → garbage
                    continue
                inventory.append({
                    "class": str(item["class"]).strip().lower(),
                    "count": int(item.get("count", 1)),
                    "bbox_normalized": bbox,
                })
            except (KeyError, TypeError, ValueError):
                continue                     # one bad item shouldn't kill the whole list

        return inventory
    
    def get_material_hint(self, crop: np.ndarray, object_class: str) -> str:
        prompt = (
            f"This is a cropped photo of a {object_class} in a cafe. "
            "Reply with ONLY one word for its primary material: "
            "wood, fabric, ceramic, metal, glass, plastic, or unknown."
        )
        text = self._ask(crop, prompt, max_tokens=10)
        return text.lower() if text else "unknown"

    def get_object_condition(self, crop: np.ndarray, object_class: str) -> str:
        prompt = (
            f"This is a cropped photo of a {object_class}. Reply with ONLY one word "
            "describing its condition: clean, dirty, broken, worn, or unclear."
        )
        text = self._ask(crop, prompt, max_tokens=10)
        return text.lower() if text else "unclear"

    def confirm_stain(self, crop: np.ndarray) -> str:
        prompt = (
            "This is a small cropped region from a table or chair surface. "
            "Is it a real stain/mark, or a pattern, texture, or shadow? "
            "Reply with ONLY one word: stain, pattern, shadow, or unclear."
        )
        text = self._ask(crop, prompt, max_tokens=10)
        return text.lower() if text else "unclear"

    def describe_layout(self, full_image: np.ndarray, object_summary: str) -> str:
        prompt = (
            "Here is a photo of a cafe table setup. Objects detected by an object "
            f"detector: {object_summary}. In 1-2 plain sentences, describe the "
            "arrangement and note anything that looks incomplete or unusual."
        )
        text = self._ask(full_image, prompt, max_tokens=150)
        return text if text else "pending_vlm"
