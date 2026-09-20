import os
import base64
import json

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD .env
# ============================================================

load_dotenv()
# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = r"C:\Users\anjishnu.kumbhakar\OneDrive - JK Technosoft Ltd\Desktop\nudge_pipeline\images\daily_2.jpg"

API_KEY = os.getenv("NVIDIA_API_KEY")

BASE_URL = "https://integrate.api.nvidia.com/v1"

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


# ============================================================
# CHECK API KEY
# ============================================================

print("=" * 80)
print("NVIDIA NEMOTRON VLM TEST")
print("=" * 80)

print("API key found:", bool(API_KEY))
print("Base URL:", BASE_URL)
print("Model:", MODEL)
print("Image:", IMAGE_PATH)

if not API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set."
    )


# ============================================================
# LOAD IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


# ============================================================
# BASE64 IMAGE
# ============================================================

image_b64 = base64.b64encode(
    image_bytes
).decode("utf-8")

image_data_uri = (
    f"data:image/jpeg;base64,{image_b64}"
)


print("Image loaded successfully.")
print("Image size:", len(image_bytes), "bytes")


# ============================================================
# NVIDIA CLIENT
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# PROMPT
# ============================================================

prompt = """
Analyze this image as a visual object discovery task.

Identify ALL distinct visible objects you can find.

Include:
- furniture
- people
- electronics
- lighting
- plants
- decorations
- tables
- chairs
- floor
- ceiling
- walls
- doors
- windows
- small objects
- cafe/table items
- anything else clearly visible

For every object, provide an approximate bounding box.

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

The bounding box coordinates must be normalized from 0 to 1:

x1 = left
y1 = top
x2 = right
y2 = bottom

Do not return markdown.
Do not return explanations.
Return JSON only.
"""


# ============================================================
# CALL MODEL
# ============================================================

print("\nCalling NVIDIA model...")
print("-" * 80)

try:

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": [

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    },

                    {
                        "type": "text",
                        "text": prompt
                    }

                ]
            }
        ],

        temperature=0.1,

        max_tokens=2000,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

except Exception as e:

    print("\n" + "=" * 80)
    print("NVIDIA API CALL FAILED")
    print("=" * 80)

    print("Error type:")
    print(type(e).__name__)

    print("\nError:")
    print(str(e))

    raise


# ============================================================
# RAW RESPONSE
# ============================================================

print("\n" + "=" * 80)
print("NVIDIA API CALL SUCCESSFUL")
print("=" * 80)

print("Model returned:")
print(response.model)

print("\nFinish reason:")
print(response.choices[0].finish_reason)


# ============================================================
# CONTENT
# ============================================================

content = response.choices[0].message.content

print("\n" + "=" * 80)
print("RAW MODEL CONTENT")
print("=" * 80)

print(content)


# ============================================================
# TRY JSON PARSING
# ============================================================

print("\n" + "=" * 80)
print("JSON PARSING")
print("=" * 80)

try:

    result = json.loads(content)

    print("JSON parsing: SUCCESS")

    objects = result.get("objects", [])

    print("Objects returned:", len(objects))

    for i, obj in enumerate(objects, 1):

        print(
            f"{i:03d}. "
            f"{obj.get('class', 'unknown')} "
            f"{obj.get('bbox_normalized', [])}"
        )

except Exception as e:

    print("JSON parsing: FAILED")
    print(type(e).__name__)
    print(str(e))