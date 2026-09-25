from __future__ import annotations
from typing import Callable, Any
from src.pipeline.pipeline_helper.context import PipelineContext
from src.pipeline.steps import (
    step5_vlm_inspection,
)
from src.pipeline.steps import step1_alignment, step2_feature_extraction, step3_ssim_lighting, step4_delta_checklist

StepCallable = Callable[[PipelineContext], bool]

class PipelineRunner:
    """Pluggable pipeline execution engine."""
    
    def __init__(self, steps: list[Any] | None = None):
        self.steps: list[StepCallable] = []
        default_steps = steps or [
            step1_alignment,
            step2_feature_extraction,
            step3_ssim_lighting,
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

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Executes registered pipeline steps sequentially."""
        for step_fn in self.steps:
            if ctx.halt:
                print(f"[PIPELINE HALTED] Stopping further steps for '{ctx.filename}'.")
                break
            success = step_fn(ctx)
            if not success and ctx.halt:
                break
        return ctx
