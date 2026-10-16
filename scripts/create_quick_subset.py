#!/usr/bin/env python3
"""
Create a quick subset of 25 diverse images for fast labeling and testing
"""

import os
import shutil
import random
from pathlib import Path

def create_subset():
    """Select 25 diverse images for quick labeling"""

    print("=" * 70)
    print("CREATING QUICK SUBSET DATASET")
    print("=" * 70)
    print()

    # Paths
    all_images_dir = Path("../data/to_label")
    subset_images_dir = Path("../data/quick_subset/images")
    subset_labels_dir = Path("../data/quick_subset/labels")

    # Create directories
    subset_images_dir.mkdir(parents=True, exist_ok=True)
    subset_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    all_images = sorted(all_images_dir.glob("*.png"))
    total_images = len(all_images)

    print(f"Total available images: {total_images}")
    print()

    # Strategy: Take images at regular intervals to ensure diversity
    # This captures different scenes/angles/lighting conditions

    subset_size = 25
    step = total_images // subset_size

    selected_images = []
    for i in range(subset_size):
        idx = i * step
        if idx < total_images:
            selected_images.append(all_images[idx])

    print(f"Selected {len(selected_images)} diverse images (every {step}th image)")
    print()

    # Copy selected images to subset directory
    print("Copying images to subset directory...")
    for i, img_path in enumerate(selected_images, 1):
        dest = subset_images_dir / img_path.name
        shutil.copy2(img_path, dest)
        print(f"  [{i}/{len(selected_images)}] {img_path.name}")

    print()
    print("=" * 70)
    print("SUBSET CREATED!")
    print("=" * 70)
    print(f"Location: {subset_images_dir.absolute()}")
    print(f"Images: {len(selected_images)}")
    print()
    print("NEXT STEP:")
    print("  Label these 25 images using the web-based labeler!")
    print("  Estimated time: 10-15 minutes")
    print("=" * 70)

    return subset_images_dir, len(selected_images)

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    create_subset()
