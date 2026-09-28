from __future__ import annotations
from typing import Callable, Any
from pipeline_helper.context import PipelineContext
from pipeline import (
    step1_alignment,
    step2_feature_extraction,
    step3_ssim_lighting,
    step3b_bulb_detection,  # <-- Added
    step4_delta_checklist,
    step5_vlm_inspection,
)

StepCallable = Callable[[PipelineContext], bool]

class PipelineRunner:
    """Pluggable pipeline execution engine."""
    
    def __init__(self, steps: list[Any] | None = None):
        self.steps: list[StepCallable] = []
        default_steps = steps or [
            step1_alignment,
            step2_feature_extraction,
            step3_ssim_lighting,
            step3b_bulb_detection,  # <-- Registered
            step4_delta_checklist,
            step5_vlm_inspection,
        ]
        for step in default_steps:
            self.register_step(step)

    def register_step(self, step: Any) -> None:
        """Registers a new step plugin function or module with a run() method."""
        if hasattr(step, "run") and callable(step.run):
            self.steps.append(step.run)
        elif callable(step):
            self.steps.append(step)
        else:
            raise ValueError(f"Step {step} must be a callable or a module with a run() function.")

    def _get_step_label(self, step_fn: StepCallable) -> str:
        """Derives a clean, readable name from the step module or function."""
        module_name = getattr(step_fn, "__module__", "")
        
        name_map = {
            "step1_alignment": "Image Alignment & Homography",
            "step2_feature_extraction": "Object & Feature Extraction",
            "step3_ssim_lighting": "Lighting & SSIM Comparison",
            "step3b_bulb_detection": "Bulb & Fixture Verification",
            "step4_delta_checklist": "Delta & Inventory Checklist",
            "step5_vlm_inspection": "Multimodal VLM Visual Inspection",
        }
        
        for key, display_name in name_map.items():
            if key in module_name:
                return display_name
                
        # Fallback to function docstring or formatted function name
        doc = getattr(step_fn, "__doc__", "")
        if doc and ":" in doc:
            return doc.split(":")[1].strip().split(".")[0]
            
        return getattr(step_fn, "__name__", "Processing Step").replace("_", " ").title()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Executes registered pipeline steps sequentially."""
        for step_fn in self.steps:
            step_name = self._get_step_label(step_fn)

            if ctx.halt:
                print(f"\n[PIPELINE ABORTED] Inspection stopped early for '{ctx.filename}'.")
                break

            success = step_fn(ctx)

            if not success or ctx.halt:
                ctx.halt = True
                print("\n" + "=" * 55)
                print(f"[PIPELINE STOPPED] Verification failed at: {step_name}")
                if ctx.errors:
                    print(f"Reason: {ctx.errors[-1]}")
                print("Downstream verification (Bulbs, Delta, VLM) skipped.")
                print("=" * 55 + "\n")
                break

        return ctx