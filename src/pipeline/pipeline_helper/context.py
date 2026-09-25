from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class PipelineContext:
    """Holds runtime state passed between pipeline steps."""
    img_path: str
    filename: str = ""
    room_name: str = ""
    
    # Image matrices
    master_img: np.ndarray | None = None
    raw_current_img: np.ndarray | None = None
    aligned_current_img: np.ndarray | None = None
    
    # Feature & Baseline Metadata
    master_data: dict[str, Any] = field(default_factory=dict)
    current_data: dict[str, Any] = field(default_factory=dict)
    model: Any = None
    
    # Computer Vision Metrics
    ssim_score: float = 0.0
    brightness_diff: float = 0.0
    
    # Reset Checklist Delta
    missing_items: list[str] = field(default_factory=list)
    drift_alerts: list[str] = field(default_factory=list)
    clutter_items: list[str] = field(default_factory=list)
    
    # VLM Rules & Findings
    room_rule: str | None = None
    vlm_result: Any = None
    
    # Control Flow & Logging
    errors: list[str] = field(default_factory=list)
    halt: bool = False
