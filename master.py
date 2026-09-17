"""Master reference setup and baseline generation script."""

import os


def generate_baselines(master_dir: str, baselines_dir: str):
    """Generate baseline JSONs and reference data from master images."""
    print(f"Generating baselines from '{master_dir}' into '{baselines_dir}'...")


if __name__ == "__main__":
    generate_baselines("master_images", "baselines")
