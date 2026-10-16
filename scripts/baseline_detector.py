#!/usr/bin/env python3
"""
Baseline Fire Extinguisher Detector
- Uses trained YOLOv11 model
- No decoy filtering (accepts all detections)
- Estimates 3D position using median depth
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from ultralytics import YOLO

def run_baseline_detector(
    model_path='../models/weights/best.pt',
    images_dir='../data/to_label',
    output_dir='../results/baseline_detections',
    conf_threshold=0.25
):
    """
    Run baseline detector on all images

    Args:
        model_path: Path to trained YOLO model
        images_dir: Directory containing images
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """

    print("=" * 70)
    print("BASELINE FIRE EXTINGUISHER DETECTOR")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_dir = output_dir / "detections"
    detections_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded")
    print()

    # Get all images
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob("*.png"))

    print(f"Found {len(image_files)} images to process")
    print()

    # Process each image
    all_detections = {}
    detection_count = 0

    print("Processing images...")
    print("-" * 70)

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}...", end=" ")

        # Run detection
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            verbose=False
        )

        # Extract detections
        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Estimate depth (median depth heuristic)
                # Assuming typical fire extinguisher height ~45cm
                # and typical image height ~1080px
                focal_length = 800  # Approximate focal length
                real_height = 0.45  # meters
                depth_estimate = (focal_length * real_height) / height if height > 0 else 2.0

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)],
                    'confidence': float(conf),
                    'depth_estimate': float(depth_estimate),
                    'width': float(width),
                    'height': float(height)
                }

                detections.append(detection)
                detection_count += 1

        all_detections[img_path.name] = {
            'image': str(img_path.name),
            'num_detections': len(detections),
            'detections': detections
        }

        print(f"{len(detections)} detection(s)")

        # Save visualization
        if len(detections) > 0:
            img = cv2.imread(str(img_path))
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{det['confidence']:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            cv2.imwrite(str(detections_dir / img_path.name), img)

    print("-" * 70)
    print()

    # Save results
    results_file = output_dir / "baseline_detections.json"
    with open(results_file, 'w') as f:
        json.dump(all_detections, f, indent=2)

    # Summary
    images_with_detections = sum(1 for v in all_detections.values() if v['num_detections'] > 0)

    print("=" * 70)
    print("BASELINE DETECTION COMPLETE!")
    print("=" * 70)
    print(f"Total images processed:      {len(image_files)}")
    print(f"Images with detections:      {images_with_detections}")
    print(f"Images without detections:   {len(image_files) - images_with_detections}")
    print(f"Total detections:            {detection_count}")
    print(f"Average per image:           {detection_count / len(image_files):.2f}")
    print()
    print(f"Results saved to:            {results_file}")
    print(f"Visualizations saved to:     {detections_dir}")
    print("=" * 70)

    return all_detections

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    run_baseline_detector()
