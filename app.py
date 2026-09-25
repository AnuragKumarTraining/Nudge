from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import streamlit as st

from src.utils.model import load_yolo
from src.pipeline.runner import PipelineRunner
from inference import process_single_image
from config.inference_config import BASELINES_DIR, CURRENT_DIR, VALID_EXTENSIONS


APP_DIR = Path(__file__).resolve().parent
BASELINES_PATH = APP_DIR / BASELINES_DIR
CURRENT_PATH = APP_DIR / CURRENT_DIR



st.set_page_config(
    page_title="DV",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️DeltaVision")
st.caption(
    "5-step inspection: alignment → YOLO features → SSIM/lighting → "
    "delta checklist → VLM inspection"
)


def get_available_rooms() -> list[str]:
    """Return rooms for which both baseline JSON and reference image exist."""
    if not BASELINES_PATH.exists():
        return []

    rooms = []
    for json_file in BASELINES_PATH.glob("*_baseline.json"):
        room = json_file.name.removesuffix("_baseline.json")
        ref_file = BASELINES_PATH / f"{room}_ref.jpg"
        if ref_file.exists():
            rooms.append(room)

    return sorted(rooms)


@st.cache_resource
def initialize_engine():
    """Load heavyweight models only once."""
    model = load_yolo()
    runner = PipelineRunner()
    return model, runner


def save_uploaded_image(uploaded_file, room_name: str) -> Path:
    """
    Save Streamlit's uploaded bytes to a real file path under CURRENT_DIR.

    This is important because the existing pipeline is path-based:
        cv2.imread(path)
        YOLO(path)
        process_single_image(path)
    """
    room_dir = CURRENT_PATH / room_name
    room_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in VALID_EXTENSIONS:
        suffix = ".jpg"

    # Avoid overwriting an existing image with the same name.
    safe_name = Path(uploaded_file.name).name
    if Path(safe_name).suffix.lower() not in VALID_EXTENSIONS:
        safe_name = Path(safe_name).stem + suffix

    destination = room_dir / safe_name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def prepare_path_image(source_path: str, room_name: str) -> Path:
    """
    Copy a user-provided local file path into the pipeline's expected
    current_images/<room>/ directory.

    The original file is never modified.
    """
    source = Path(source_path).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Image file does not exist: {source}")

    if not source.is_file():
        raise ValueError(f"Path is not a file: {source}")

    if source.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{source.suffix}'. "
            f"Use: {', '.join(VALID_EXTENSIONS)}"
        )

    room_dir = CURRENT_PATH / room_name
    room_dir.mkdir(parents=True, exist_ok=True)

    destination = room_dir / source.name

    # If it is already in the expected location, use it directly.
    if source == destination.resolve():
        return destination

    shutil.copy2(source, destination)
    return destination


def render_results(ctx):
    """Render the PipelineContext as a Streamlit dashboard."""

    st.success(
        f"Pipeline completed successfully for **{ctx.room_name.upper()}** "
        f"({ctx.filename})"
    )

    st.subheader("1. Spatial Alignment")

    col1, col2 = st.columns(2)

    with col1:
        if ctx.master_img is not None:
            master_rgb = cv2.cvtColor(ctx.master_img, cv2.COLOR_BGR2RGB)
            st.image(
                master_rgb,
                caption="Master Reference",
                use_container_width=True,
            )

    with col2:
        if ctx.aligned_current_img is not None:
            aligned_rgb = cv2.cvtColor(
                ctx.aligned_current_img,
                cv2.COLOR_BGR2RGB,
            )
            st.image(
                aligned_rgb,
                caption="Aligned Current Image",
                use_container_width=True,
            )
    st.subheader("2. Structural & Lighting Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "SSIM Score",
        f"{ctx.ssim_score:.3f}",
        help="1.0 means identical structural appearance.",
    )

    m2.metric(
        "Lighting Delta",
        f"{ctx.brightness_diff:+.2f}",
        help="Current brightness minus baseline brightness.",
    )

    m3.metric(
        "Objects Detected",
        len(ctx.current_data.get("objects", [])),
    )

    m4.metric(
        "Pipeline Errors",
        len(ctx.errors),
    )

    st.subheader("3. Reset Checklist Delta")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Missing Items**")
        if not ctx.missing_items:
            st.success("All required items present.")
        else:
            for item in sorted(set(ctx.missing_items)):
                count = ctx.missing_items.count(item)
                st.error(f"{count} × {item.capitalize()}")

    with c2:
        st.markdown("**Furniture Drift**")
        if not ctx.drift_alerts:
            st.success("Positions verified.")
        else:
            for alert in ctx.drift_alerts:
                st.warning(alert)

    with c3:
        st.markdown("**Clutter Detected**")
        if not ctx.clutter_items:
            st.success("Room is clean.")
        else:
            for item in sorted(set(ctx.clutter_items)):
                count = ctx.clutter_items.count(item)
                st.warning(f"Remove {count} × {item.capitalize()}")

    st.subheader("4. VLM Visual Inspection")

    if ctx.vlm_result is None:
        st.warning("VLM inspection was skipped or returned no result.")
        return

    if ctx.vlm_result.status == "OK":
        st.success("VLM status: OK")
    else:
        st.warning(f"VLM status: {ctx.vlm_result.status}")

    st.write(f"**Summary:** {ctx.vlm_result.summary}")

    if not ctx.vlm_result.issues:
        st.success("No visual discrepancies detected by VLM.")
    else:
        st.markdown("**Detected Discrepancies:**")

        for issue in ctx.vlm_result.issues:
            severity = issue.severity.upper()

            if severity == "HIGH":
                st.error(
                    f"[{issue.type}] **{issue.object}** — "
                    f"{issue.description} "
                    f"(Confidence: {issue.confidence:.2f})"
                )
            elif severity == "MEDIUM":
                st.warning(
                    f"[{issue.type}] **{issue.object}** — "
                    f"{issue.description} "
                    f"(Confidence: {issue.confidence:.2f})"
                )
            else:
                st.info(
                    f"[{issue.type}] **{issue.object}** — "
                    f"{issue.description} "
                    f"(Confidence: {issue.confidence:.2f})"
                )

    with st.expander("Pipeline Details"):
        st.write("**Runtime image path:**", ctx.img_path)
        st.write("**Room:**", ctx.room_name)
        st.write("**Filename:**", ctx.filename)

        st.markdown("**Baseline objects**")
        st.json(ctx.master_data.get("objects", []))

        st.markdown("**Current objects**")
        st.json(ctx.current_data.get("objects", []))


st.sidebar.header("Inspection Settings")

rooms = get_available_rooms()

if not rooms:
    st.error(
        f"No valid baselines found in `{BASELINES_PATH}`. "
        "Each room needs `<room>_baseline.json` and `<room>_ref.jpg`."
    )
    st.stop()

room_name = st.sidebar.selectbox(
    "Select Room",
    rooms,
    help="The selected room determines which baseline JSON and reference image are used.",
)

input_mode = st.sidebar.radio(
    "Image Source",
    ["Upload Image", "Local File Path"],
)

uploaded_file = None
file_path = ""

if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Select current room image",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    file_path = st.sidebar.text_input(
        "Current image file path",
        placeholder=r"C:\path\to\room.jpg",
        help=(
            "Path must be accessible by the machine running Streamlit. "
            "The file will be copied into current_images/<room>/ before processing."
        ),
    )

run_pipeline = st.sidebar.button(
    "🚀 Run Inspection Pipeline",
    type="primary",
    use_container_width=True,
)

if not run_pipeline:
    st.info(
        "Select the room, provide the current image, and click "
        "**Run Inspection Pipeline**."
    )
    st.stop()

if input_mode == "Upload Image" and uploaded_file is None:
    st.warning("Please upload an image first.")
    st.stop()

if input_mode == "Local File Path" and not file_path.strip():
    st.warning("Please enter the current image file path.")
    st.stop()

# The pipeline's room resolver expects:
# current_images/<room_name>/<image>
try:
    if input_mode == "Upload Image":
        runtime_image = save_uploaded_image(uploaded_file, room_name)
    else:
        runtime_image = prepare_path_image(file_path.strip(), room_name)

except Exception as exc:
    st.error(f"Could not prepare image: {exc}")
    st.stop()

#st.info(f"Runtime image path: `{runtime_image}`")

# Load YOLO + runner only when actually needed.
try:
    with st.spinner("Loading YOLO model and pipeline..."):
        model, runner = initialize_engine()
except Exception as exc:
    st.error(f"Could not initialize the inspection engine: {exc}")
    st.stop()

# Run the existing path-based pipeline.
try:
    with st.spinner(
        "Running pipeline: Alignment → YOLO → SSIM → Delta → VLM..."
    ):
        ctx = process_single_image(
            img_path=str(runtime_image),
            model=model,
            runner=runner,
        )

    if ctx is None:
        st.error(
            "Pipeline could not initialize. Check that the selected room "
            "has the required baseline JSON and reference image."
        )
    elif ctx.halt:
        st.error("Pipeline halted because of a critical error.")
        for error in ctx.errors:
            st.warning(error)
    else:
        render_results(ctx)

except Exception as exc:
    st.exception(exc)
