import os
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIG (Matching VLMClient setup)
# ============================================================

IMAGE_PATH = (
    r"C:\Users\anjishnu.kumbhakar"
    r"\OneDrive - JK Technosoft Ltd"
    r"\Desktop\nudge_pipeline"
    r"\images\daily_2.jpg"
)

# Hugging Face Access Token
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# Hugging Face OpenAI-Compatible Base URL
VLM_BASE_URL = "https://router.huggingface.co/v1"
VLM_IMAGE_URL_FORMAT = "nested"  # "nested" for OpenAI/HF standard, "flat" for Ollama

# Model targeted on Hugging Face Hub
MODEL = "Qwen/Qwen3.6-35B-A3B"

TEMPERATURE = 0.1
MAX_TOKENS = 4096  # Raised to 4096 to prevent empty responses due to max token limits

# ============================================================
# HELPER METHODS (Identical to VLMClient)
# ============================================================

def encode_image_file_to_data_uri(file_path: str) -> str:
    """Reads image file and returns base64 data URI."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def build_image_content(data_uri: str) -> dict:
    """Matches VLMClient logic for formatting image URLs in OpenAI payloads."""
    if VLM_IMAGE_URL_FORMAT == "flat":
        return {"type": "image_url", "image_url": data_uri}
    return {"type": "image_url", "image_url": {"url": data_uri}}

# ============================================================
# HEADER & INITIALIZATION
# ============================================================

print("=" * 80)
print("OPENAI SDK -> HUGGING FACE ROUTER VLM TEST")
print("=" * 80)
print("Base URL:", VLM_BASE_URL)
print("Model:", MODEL)
print("Image:", IMAGE_PATH)

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found:\n{IMAGE_PATH}")

# Initialize OpenAI client targeting Hugging Face Endpoint
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=VLM_BASE_URL
)

# ============================================================
# PROMPT
# ============================================================

prompt = """
Analyze this image as a visual object discovery task.
Identify ALL distinct visible objects you can find.

Return ONLY valid JSON.
Format:
{
    "objects": [
        {
            "class": "chair",
            "description": "wooden chair",
            "bbox_normalized": [x1, y1, x2, y2]
        }
    ]
}
x1 = left, y1 = top, x2 = right, y2 = bottom (normalized 0 to 1).
Do not return markdown or explanations. Return JSON only.
"""

# ============================================================
# CALL MODEL VIA OPENAI CLIENT
# ============================================================

data_uri = encode_image_file_to_data_uri(IMAGE_PATH)
image_content_block = build_image_content(data_uri)

print("\nSending request via OpenAI Client...")
print("-" * 80)

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content_block,
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"}
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason

    if finish_reason == "length":
        print("⚠️ Warning: Model response truncated due to max_tokens limit!")

    content = choice.message.content or ""

except Exception as e:
    print("\nAPI CALL FAILED:", type(e).__name__, "-", e)
    raise e

# ============================================================
# JSON PARSING & DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("RAW MODEL CONTENT")
print("=" * 80)
print(content)

try:
    if not content.strip():
        raise ValueError("Model returned an empty text string.")

    result = json.loads(content)
    objects = result.get("objects", [])

    print("\n" + "=" * 80)
    print(f"OBJECTS DETECTED: {len(objects)}")
    print("=" * 80)

    for i, obj in enumerate(objects, 1):
        print(f"{i:03d}. {obj.get('class', 'unknown')} {obj.get('bbox_normalized', [])}")

except Exception as e:
    print("\nParsing Error:", type(e).__name__, "-", e)