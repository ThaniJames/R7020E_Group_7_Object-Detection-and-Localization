#!/usr/bin/env python3
"""
Evaluation Script for Fire Extinguisher Detector
Calculates TPR (True Positive Rate) and FPR (False Positive Rate)

TPR = TP / (TP + FN)  - Percentage of real fire extinguishers detected
FPR = FP / (FP + TN)  - Percentage of decoys incorrectly detected as real

Target: TPR ≥ 0.75, FPR ≤ 0.15
"""

import os
import sys
from pathlib import Path
import json

def load_ground_truth(labels_file='../data/annotations/real_vs_decoy_labels.txt'):
    """
    Load ground truth labels for real vs decoy fire extinguishers

    Returns:
        dict: {image_name: 'REAL' or 'DECOY'}
    """

    ground_truth = {}
    labels_file = Path(labels_file)

    if not labels_file.exists():
        print(f"⚠️  Ground truth file not found: {labels_file}")
        print("Please label images as REAL or DECOY in the file")
        return ground_truth

    with open(labels_file, 'r') as f:
        lines = f.readlines()

    current_type = None
    for line in lines:
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Check section headers
        if 'REAL FIRE EXTINGUISHERS' in line:
            current_type = 'REAL'
            continue
        elif 'DECOY FIRE EXTINGUISHERS' in line:
            current_type = 'DECOY'
            continue

        # Parse filename
        if current_type and line.endswith('.png'):
            ground_truth[line] = current_type

    return ground_truth

def evaluate_detections(
    detections_file='../results/baseline_detections/baseline_detections.json',
    labels_file='../data/annotations/real_vs_decoy_labels.txt',
    output_file='../results/evaluation_metrics.json'
):
    """
    Evaluate detector performance and calculate TPR/FPR

    Args:
        detections_file: JSON file with baseline detections
        labels_file: Ground truth labels file
        output_file: Output file for evaluation metrics
    """

    print("=" * 70)
    print("DETECTOR EVALUATION - TPR & FPR CALCULATION")
    print("=" * 70)
    print()

    # Load detections
    print(f"Loading detections: {detections_file}")
    with open(detections_file, 'r') as f:
        detections = json.load(f)
    print(f"✓ Loaded detections for {len(detections)} images")
    print()

    # Load ground truth
    print(f"Loading ground truth: {labels_file}")
    ground_truth = load_ground_truth(labels_file)
    print(f"✓ Loaded ground truth for {len(ground_truth)} images")
    print()

    if len(ground_truth) == 0:
        print("❌ ERROR: No ground truth labels found!")
        print("Please fill in the real_vs_decoy_labels.txt file")
        return None

    # Calculate metrics
    print("Calculating metrics...")
    print()

    TP = 0  # True Positives: Real fire extinguisher detected as real
    FN = 0  # False Negatives: Real fire extinguisher not detected
    FP = 0  # False Positives: Decoy detected as real
    TN = 0  # True Negatives: Decoy not detected (correctly rejected)

    real_images = []
    decoy_images = []

    for img_name, img_type in ground_truth.items():
        # Get detection for this image
        detection_info = detections.get(img_name, {'num_detections': 0})
        has_detection = detection_info['num_detections'] > 0

        if img_type == 'REAL':
            real_images.append(img_name)
            if has_detection:
                TP += 1  # Correctly detected real fire extinguisher
            else:
                FN += 1  # Missed real fire extinguisher

        elif img_type == 'DECOY':
            decoy_images.append(img_name)
            if has_detection:
                FP += 1  # Incorrectly detected decoy as real
            else:
                TN += 1  # Correctly rejected decoy

    # Calculate rates
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity / Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    # Results
    results = {
        'confusion_matrix': {
            'TP': TP,  # True Positives
            'FN': FN,  # False Negatives
            'FP': FP,  # False Positives
            'TN': TN   # True Negatives
        },
        'metrics': {
            'TPR': round(TPR, 4),  # True Positive Rate (Recall)
            'FPR': round(FPR, 4),  # False Positive Rate
            'Precision': round(precision, 4),
            'Accuracy': round(accuracy, 4)
        },
        'dataset_info': {
            'total_images': len(ground_truth),
            'real_images': len(real_images),
            'decoy_images': len(decoy_images)
        },
        'targets': {
            'TPR': 0.75,
            'FPR': 0.15
        },
        'meets_requirements': {
            'TPR': TPR >= 0.75,
            'FPR': FPR <= 0.15,
            'overall': TPR >= 0.75 and FPR <= 0.15
        }
    }

    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print()
    print("Dataset:")
    print(f"  Total images:       {len(ground_truth)}")
    print(f"  Real images:        {len(real_images)}")
    print(f"  Decoy images:       {len(decoy_images)}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives (TP):   {TP:3d}  (Real detected as real)")
    print(f"  False Negatives (FN):  {FN:3d}  (Real missed)")
    print(f"  False Positives (FP):  {FP:3d}  (Decoy detected as real)")
    print(f"  True Negatives (TN):   {TN:3d}  (Decoy correctly rejected)")
    print()
    print("Performance Metrics:")
    print(f"  TPR (True Positive Rate):  {TPR:.4f}  (Target: ≥ 0.75) {'✅' if TPR >= 0.75 else '❌'}")
    print(f"  FPR (False Positive Rate): {FPR:.4f}  (Target: ≤ 0.15) {'✅' if FPR <= 0.15 else '❌'}")
    print(f"  Precision:                 {precision:.4f}")
    print(f"  Accuracy:                  {accuracy:.4f}")
    print()

    if results['meets_requirements']['overall']:
        print("✅ PASSED: System meets project requirements!")
    else:
        print("❌ FAILED: System does not meet project requirements")
        if not results['meets_requirements']['TPR']:
            print(f"   - TPR too low: {TPR:.4f} < 0.75")
        if not results['meets_requirements']['FPR']:
            print(f"   - FPR too high: {FPR:.4f} > 0.15")

    print()
    print("=" * 70)

    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    # Check if ground truth exists
    labels_file = Path('../data/annotations/real_vs_decoy_labels.txt')

    if not labels_file.exists() or labels_file.stat().st_size < 100:
        print("⚠️  Ground truth labels not found or incomplete!")
        print()
        print("Before running evaluation, you need to:")
        print("1. Review each image in data/to_label/")
        print("2. Label each as REAL or DECOY in:")
        print(f"   {labels_file}")
        print()
        print("Format:")
        print("  # REAL FIRE EXTINGUISHERS")
        print("  camera_color_image_1234.png")
        print("  camera_color_image_5678.png")
        print()
        print("  # DECOY FIRE EXTINGUISHERS")
        print("  camera_color_image_9012.png")
    else:
        evaluate_detections()
