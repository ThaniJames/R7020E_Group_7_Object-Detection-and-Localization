#!/usr/bin/env python3
"""
Auto-label fire extinguisher images using pre-trained YOLOv11
Then user can review/correct in LabelImg (much faster than manual labeling!)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Install ultralytics if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics (YOLOv11)...")
    os.system("pip install ultralytics -q")
    from ultralytics import YOLO

def auto_label_images():
    """Auto-label images using pre-trained YOLOv11"""

    print("=" * 70)
    print("AUTO-LABELING FIRE EXTINGUISHER IMAGES")
    print("=" * 70)
    print()

    # Paths
    images_dir = Path("../data/to_label")
    labels_dir = Path("../data/yolo_labels")
    labels_dir.mkdir(exist_ok=True)

    # Get all images
    image_files = sorted(images_dir.glob("*.png"))
    total_images = len(image_files)

    print(f"Found {total_images} images to label")
    print()

    # Load pre-trained YOLOv11 model (trained on COCO dataset)
    print("Loading YOLOv11n model (pre-trained on COCO)...")
    model = YOLO('yolo11n.pt')  # Nano model - fast and lightweight
    print("✓ Model loaded")
    print()

    # COCO classes that might be fire extinguishers
    # Fire extinguishers often detected as: bottle, vase, cup, etc.
    target_classes = {
        39: 'bottle',      # Most common
        75: 'vase',        # Also common
        41: 'cup',         # Sometimes
        46: 'wine glass',  # Occasionally
    }

    print("Will detect objects that could be fire extinguishers:")
    for class_id, class_name in target_classes.items():
        print(f"  - {class_name} (class {class_id})")
    print()

    # Process each image
    labeled_count = 0
    skipped_count = 0

    print(f"Processing {total_images} images...")
    print("-" * 70)

    for i, image_path in enumerate(image_files, 1):
        # Progress
        print(f"[{i}/{total_images}] {image_path.name}...", end=" ")

        # Check if already labeled
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            print("(already labeled - skipping)")
            skipped_count += 1
            continue

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print("❌ Failed to read")
            continue

        img_height, img_width = img.shape[:2]

        # Run detection
        results = model(img, verbose=False)

        # Extract detections for target classes
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only keep target classes with high confidence
                if class_id in target_classes and confidence > 0.3:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to YOLO format (center_x, center_y, width, height) normalized
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Class 0 for fire_extinguisher (our single class)
                    detections.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        # Save labels
        if detections:
            with open(label_path, 'w') as f:
                f.write('\n'.join(detections))
            print(f"✓ {len(detections)} object(s) detected")
            labeled_count += 1
        else:
            # Create empty label file (no detections)
            label_path.touch()
            print("⚠️  No objects detected (created empty label)")
            labeled_count += 1

    # Summary
    print("-" * 70)
    print()
    print("=" * 70)
    print("AUTO-LABELING COMPLETE!")
    print("=" * 70)
    print(f"Total images:           {total_images}")
    print(f"Auto-labeled:           {labeled_count}")
    print(f"Skipped (existing):     {skipped_count}")
    print()
    print("NEXT STEP:")
    print("  Open LabelImg to REVIEW and CORRECT the labels:")
    print("  ./scripts/launch_labelimg.sh")
    print()
    print("  Common corrections needed:")
    print("  - Add missed fire extinguishers (press W)")
    print("  - Delete false detections (select box, press Del)")
    print("  - Adjust box sizes to fit tightly")
    print()
    print("  Estimated review time: ~15-20 minutes (much faster!)")
    print("=" * 70)

if __name__ == "__main__":
    # Change to script directory
    os.chdir(Path(__file__).parent)
    auto_label_images()
