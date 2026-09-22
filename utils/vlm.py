from __future__ import annotations

"""Room state visual inspection module using Google GenAI SDK.

This module provides structured schema validation and comparative multimodal analysis
between an expected baseline state (master image + JSON state) and a processed runtime image.
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass


# --- Type Aliases ---
IssueType = Literal[
    "MISSING",
    "DRIFT",
    "CLUTTER",
    "LIGHTING",
    "OTHER",
]
"""Categorization of visual defects or anomalies detected in the scene."""

Severity = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]
"""Impact level of a detected visual anomaly."""


# --- Schema Models ---

class VLMIssue(BaseModel):
    """Represents a single visual discrepancy between the baseline and current scene.

    Attributes:
        type: The category of defect (e.g., MISSING, DRIFT, CLUTTER).
        object: Name or identifier of the object/region affected.
        description: Detailed human-readable explanation of the discrepancy.
        severity: Assessed impact level of the issue.
        confidence: Normalized model confidence score ranging between 0.0 and 1.0.
    """

    type: IssueType
    object: str
    description: str
    severity: Severity
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score bounded between 0.0 and 1.0 inclusive."
    )


class VLMResult(BaseModel):
    """Root response structure returned by the multimodal inspection pipeline.

    Attributes:
        status: Overall inspection result; 'OK' if aligned, 'ISSUES_FOUND' otherwise.
        summary: High-level overview of scene alignment and primary findings.
        issues: Collection of all granular visual anomalies detected.
    """

    status: Literal["OK", "ISSUES_FOUND"]
    summary: str
    issues: list[VLMIssue] = Field(default_factory=list)


# --- Global Client & Configuration ---
def get_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set GEMINI_API_KEY in your environment or .env file.")
    return genai.Client(api_key=api_key)



# --- Helper Functions ---
def load_prompt(prompt_path: str | Path) -> str:
    """Reads and decodes system prompt template from disk.

    Args:
        prompt_path: File system path to the prompt file (as a string or Path object).

    Returns:
        The content of the prompt file decoded as a UTF-8 string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        OSError: If reading the file fails due to permissions or I/O errors.
    """
    path = Path(prompt_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path.resolve()}")

    return path.read_text(encoding="utf-8")


def load_image(
    image: bytes | str | Path,
    fallback_mime: str = "image/jpeg",
) -> tuple[bytes, str]:
    """Loads raw image bytes and infers the corresponding IANA media type.

    Accepts either raw in-memory bytes or a filesystem path. For path-based inputs,
    supported extensions include .jpg, .jpeg, .png, and .webp.

    Args:
        image: File path to the target image or raw byte payload.
        fallback_mime: Default MIME type applied when raw bytes are passed directly.

    Returns:
        A tuple of `(image_bytes, mime_type)`.

    Raises:
        FileNotFoundError: If the image path does not exist on disk.
        ValueError: If the file extension is not recognized or supported.
    """
    if isinstance(image, bytes):
        # Infer basic magic bytes if possible, otherwise rely on fallback_mime
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


# --- Core Pipeline ---
def analyze_room_with_vlm(
    processed_image: bytes | str | Path,
    master_image: bytes | str | Path,
    master_json: dict[str, Any] | str,
    master_prompt: str,
) -> VLMResult:
    """Executes comparative visual inspection between a baseline and a current room capture.

    Leverages Gemini's structured outputs with enforced Pydantic schemas to identify
    discrepancies, clutter, missing items, or lighting changes relative to baseline
    specifications.

    Args:
        processed_image: Runtime capture to evaluate (file path or raw bytes).
        master_image: Approved baseline reference image (file path or raw bytes).
        master_json: Target state metadata dictionary or formatted JSON string.
        master_prompt: System-level guidelines defining inspection criteria.

    Returns:
        A validated `VLMResult` instance containing status, summary, and detected issues.

    Raises:
        RuntimeError: If the API returns empty contents or safety filters block output.
        pydantic.ValidationError: If the response cannot be parsed into `VLMResult`.
        google.genai.errors.APIError: If the Gemini API request fails.
    """
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
The first image attached is the MASTER IMAGE (Baseline).
The second image attached is the CURRENT IMAGE (To be evaluated).

Compare the CURRENT IMAGE against the expected baseline state established by the
MASTER IMAGE and the EXPECTED SPECIFICATION JSON. Identify any discrepancies in object
placement, missing items, unauthorized clutter, or lighting deviations.

Return only the structured output adhering to the defined schema.
"""

    client = get_client()
    model = get_model()

    response = client.models.generate_content(
        model=model,
        contents=[
            instruction,
            "MASTER IMAGE (BASELINE):",
            types.Part.from_bytes(
                data=master_bytes,
                mime_type=master_mime,
            ),
            "CURRENT IMAGE (RUNTIME):",
            types.Part.from_bytes(
                data=current_bytes,
                mime_type=current_mime,
            ),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VLMResult,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response. Verify safety filter ratings "
            "and prompt token usage."
        )

    return VLMResult.model_validate_json(response.text)