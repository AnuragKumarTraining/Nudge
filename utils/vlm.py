from __future__ import annotations

"""Room state visual inspection module using Google GenAI SDK and Hugging Face.

This module provides structured schema validation and comparative multimodal analysis
with automatic fallback failover between cloud providers.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:

    def load_dotenv():
        pass


# Logger setup for failover tracking
logger = logging.getLogger(__name__)


# --- Type Aliases ---

IssueType = Literal[
    "MISSING",
    "DRIFT",
    "CLUTTER",
    "LIGHTING",
    "OTHER",
]

Severity = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


# --- Schema Models ---


class VLMIssue(BaseModel):
    """Represents a single visual discrepancy between baseline and current scene."""

    # Alias choices allow parsing hallucinated LLM keys like 'category', 'item', or 'name'
    type: IssueType = Field(validation_alias=AliasChoices("type", "category"))
    object: str = Field(validation_alias=AliasChoices("object", "item", "name"))
    description: str
    severity: Severity
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score bounded between 0.0 and 1.0 inclusive.",
    )

    @field_validator("type", "severity", mode="before")
    @classmethod
    def uppercase_literals(cls, v: Any) -> Any:
        """Coerce lowercase or mixed-case string literals to uppercase."""
        return v.upper() if isinstance(v, str) else v


class VLMResult(BaseModel):
    """Root response structure returned by the multimodal inspection pipeline."""

    status: Literal["OK", "ISSUES_FOUND"]
    summary: str = Field(
        default="Visual inspection completed. See issues list for detailed discrepancies."
    )
    issues: list[VLMIssue] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def uppercase_status(cls, v: Any) -> Any:
        """Coerce status string to uppercase."""
        return v.upper() if isinstance(v, str) else v


# --- Helper Functions ---


def load_prompt(prompt_path: str | Path) -> str:
    """Reads and decodes system prompt template from disk."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def load_image(
    image: bytes | str | Path,
    fallback_mime: str = "image/jpeg",
) -> tuple[bytes, str]:
    """Loads raw image bytes and infers the corresponding IANA media type."""
    if isinstance(image, bytes):
        if image.startswith(b"\x89PNG\r\n\x1a\n"):
            return image, "image/png"
        if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
            return image, "image/webp"
        return image, fallback_mime

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower())

    if mime_type is None:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            "Supported formats: .jpg, .jpeg, .png, .webp"
        )

    return path.read_bytes(), mime_type


def _clean_json_response(text: str) -> str:
    """Strips Markdown formatting code blocks if returned by the model."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# --- Provider API Implementations ---


def call_gemini(
    instruction: str,
    master_bytes: bytes,
    master_mime: str,
    current_bytes: bytes,
    current_mime: str,
) -> VLMResult:
    """Executes visual inspection using the Google GenAI SDK."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            instruction,
            "MASTER IMAGE (BASELINE):",
            types.Part.from_bytes(data=master_bytes, mime_type=master_mime),
            "CURRENT IMAGE (RUNTIME):",
            types.Part.from_bytes(data=current_bytes, mime_type=current_mime),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VLMResult,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini API returned an empty response.")

    return VLMResult.model_validate_json(response.text)


def call_huggingface(
    instruction: str,
    master_bytes: bytes,
    master_mime: str,
    current_bytes: bytes,
    current_mime: str,
) -> VLMResult:
    """Executes visual inspection using the Hugging Face Inference API."""
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    model = os.getenv("HF_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    client = InferenceClient(token=token)

    b64_master = base64.b64encode(master_bytes).decode("utf-8")
    master_uri = f"data:{master_mime};base64,{b64_master}"

    b64_current = base64.b64encode(current_bytes).decode("utf-8")
    current_uri = f"data:{current_mime};base64,{b64_current}"

    # Strict JSON target template injected directly into prompt
    hf_prompt = f"""{instruction}

IMPORTANT: You must return ONLY valid JSON matching this exact schema.
Do not include markdown tags, conversational text, or alter key names.

{{
  "status": "ISSUES_FOUND",
  "summary": "Summary of overall room inspection findings",
  "issues": [
    {{
      "type": "MISSING",
      "object": "chair",
      "description": "Chair missing from table",
      "severity": "HIGH",
      "confidence": 0.95
    }}
  ]
}}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": hf_prompt},
                {"type": "text", "text": "MASTER IMAGE (BASELINE):"},
                {"type": "image_url", "image_url": {"url": master_uri}},
                {"type": "text", "text": "CURRENT IMAGE (RUNTIME):"},
                {"type": "image_url", "image_url": {"url": current_uri}},
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10000,
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content
    if not raw_text:
        raise RuntimeError("Hugging Face API returned an empty response.")

    json_text = _clean_json_response(raw_text)
    return VLMResult.model_validate_json(json_text)


# --- Core Pipeline Orchestrator ---


def analyze_room_with_vlm(
    processed_image: bytes | str | Path,
    master_image: bytes | str | Path,
    master_json: dict[str, Any] | str,
    master_prompt: str,
    primary_provider: Literal["gemini", "huggingface"] = "gemini",
) -> VLMResult:
    """Executes comparative visual inspection between a baseline and a current capture.

    Leverages primary provider execution with automatic failover fallback.

    Args:
        processed_image: Runtime capture to evaluate (file path or raw bytes).
        master_image: Approved baseline reference image (file path or raw bytes).
        master_json: Target state metadata dictionary or formatted JSON string.
        master_prompt: System-level guidelines defining inspection criteria.
        primary_provider: Provider to try first ('gemini' or 'huggingface').

    Returns:
        A validated `VLMResult` instance.

    Raises:
        RuntimeError: If all configured VLM providers fail execution.
    """
    load_dotenv()

    current_bytes, current_mime = load_image(processed_image)
    master_bytes, master_mime = load_image(master_image)

    if isinstance(master_json, dict):
        master_json_text = json.dumps(master_json, indent=2)
    else:
        master_json_text = str(master_json)

    instruction = f"""
{master_prompt}

### EXPECTED SPECIFICATION (MASTER JSON)
{master_json_text}

### TASK INSTRUCTIONS
Compare the CURRENT IMAGE against the expected baseline state established by the
MASTER IMAGE and the EXPECTED SPECIFICATION JSON. Identify any discrepancies in object
placement, missing items, unauthorized clutter, or lighting deviations.
"""

    providers = {
        "gemini": call_gemini,
        "huggingface": call_huggingface,
    }

    fallback_provider = "huggingface" if primary_provider == "gemini" else "gemini"
    execution_order = [primary_provider, fallback_provider]

    last_error: Exception | None = None
    for provider_name in execution_order:
        try:
            logger.info(f"Attempting VLM analysis using provider: '{provider_name}'")
            return providers[provider_name](
                instruction=instruction,
                master_bytes=master_bytes,
                master_mime=master_mime,
                current_bytes=current_bytes,
                current_mime=current_mime,
            )
        except Exception as e:
            logger.warning(f"VLM provider '{provider_name}' failed: {e}")
            last_error = e

    raise RuntimeError(
        f"All VLM providers failed. Last error: {last_error}"
    ) from last_error