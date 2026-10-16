#!/usr/bin/env python3
"""
Evaluation Script for Fire Extinguisher Detector
Calculates TPR (True Positive Rate) and FPR (False Positive Rate)
"""

import os
import sys
from pathlib import Path
import json

def load_ground_truth(labels_file='../data/annotations/real_vs_decoy_labels.txt'):
    """Load ground truth labels"""

    ground_truth = {}
    labels_file = Path(labels_file)

    if not labels_file.exists():
        print(f"⚠️  Ground truth file not found: {labels_file}")
        return ground_truth

    with open(labels_file, 'r') as f:
        lines = f.readlines()

    current_type = None
    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for section headers (in comments)
        if line.startswith('#'):
            if 'REAL' in line.upper() and 'FIRE' in line.upper():
                current_type = 'REAL'
            elif 'DECOY' in line.upper():
                current_type = 'DECOY'
            continue

        # Parse filename
        if current_type and '.png' in line.lower():
            ground_truth[line] = current_type

    return ground_truth

def evaluate_detections():
    """Evaluate detector performance"""

    print("=" * 70)
    print("DETECTOR EVALUATION - TPR & FPR CALCULATION")
    print("=" * 70)
    print()

    # Load detections
    detections_file = '../results/baseline_detections/baseline_detections.json'
    print(f"Loading detections: {detections_file}")
    with open(detections_file, 'r') as f:
        detections = json.load(f)
    print(f"✓ Loaded detections for {len(detections)} images")
    print()

    # Load ground truth
    labels_file = '../data/annotations/real_vs_decoy_labels.txt'
    print(f"Loading ground truth: {labels_file}")
    ground_truth = load_ground_truth(labels_file)
    print(f"✓ Loaded ground truth for {len(ground_truth)} images")
    print()

    if len(ground_truth) == 0:
        print("❌ ERROR: No ground truth labels found!")
        return None

    # Calculate metrics
    print("Calculating metrics...")
    print()

    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives
    TN = 0  # True Negatives

    for img_name, img_type in ground_truth.items():
        detection_info = detections.get(img_name, {'num_detections': 0})
        has_detection = detection_info['num_detections'] > 0

        if img_type == 'REAL':
            if has_detection:
                TP += 1
            else:
                FN += 1
        elif img_type == 'DECOY':
            if has_detection:
                FP += 1
            else:
                TN += 1

    # Calculate rates
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    # Print results
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
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

    if TPR >= 0.75 and FPR <= 0.15:
        print("✅ PASSED: System meets project requirements!")
    else:
        print("❌ System does not fully meet requirements")
        if TPR < 0.75:
            print(f"   - TPR: {TPR:.4f} < 0.75 (need to detect more real fire extinguishers)")
        if FPR > 0.15:
            print(f"   - FPR: {FPR:.4f} > 0.15 (too many decoys detected as real)")

    print()
    print("=" * 70)

    # Save results
    results = {
        'confusion_matrix': {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN},
        'metrics': {
            'TPR': round(TPR, 4),
            'FPR': round(FPR, 4),
            'Precision': round(precision, 4),
            'Accuracy': round(accuracy, 4)
        },
        'meets_requirements': TPR >= 0.75 and FPR <= 0.15
    }

    output_file = Path('../results/evaluation_metrics.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    evaluate_detections()
