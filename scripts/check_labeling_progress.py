#!/usr/bin/env python3
"""
Check labeling progress for fire extinguisher dataset
"""

import os
from pathlib import Path

def check_progress():
    """Check how many images have been labeled"""

    images_dir = Path("../data/to_label")
    labels_dir = Path("../data/yolo_labels")

    # Get all images
    images = list(images_dir.glob("*.png"))
    total_images = len(images)

    # Get all labels
    labels = list(labels_dir.glob("*.txt"))
    total_labels = len(labels)

    # Calculate progress
    progress_pct = (total_labels / total_images * 100) if total_images > 0 else 0
    remaining = total_images - total_labels

    # Display results
    print("=" * 60)
    print("FIRE EXTINGUISHER LABELING PROGRESS")
    print("=" * 60)
    print(f"Total images:     {total_images}")
    print(f"Labeled:          {total_labels}")
    print(f"Remaining:        {remaining}")
    print(f"Progress:         {progress_pct:.1f}%")
    print("=" * 60)

    if total_labels == 0:
        print("\n⚠️  No labels found yet!")
        print("Run: ./scripts/launch_labelimg.sh to start labeling")
    elif total_labels == total_images:
        print("\n✅ ALL IMAGES LABELED! Ready for training!")
    else:
        print(f"\n📊 Keep going! {remaining} images left")
        print(f"   Average time: ~30 seconds per image")
        print(f"   Estimated time remaining: ~{remaining * 0.5:.0f} minutes")

    print()

    # Show sample labeled files
    if total_labels > 0:
        print("Recently labeled:")
        for label in sorted(labels)[-5:]:
            print(f"  ✓ {label.name}")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    check_progress()
