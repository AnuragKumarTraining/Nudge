from __future__ import annotations
import os
import sys
from config.inference_config import CURRENT_DIR

def select_image_dialog() -> str | None:
    """Opens a native GUI file selection pop-up window."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_dir = os.path.abspath(CURRENT_DIR) if os.path.exists(CURRENT_DIR) else os.getcwd()
        file_path = filedialog.askopenfilename(
            title="Select Room Image for Visual Inspection",
            initialdir=init_dir,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[WARN] Could not open GUI file selector: {e}")
        return None
