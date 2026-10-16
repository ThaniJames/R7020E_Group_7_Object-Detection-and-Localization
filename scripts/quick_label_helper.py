#!/usr/bin/env python3
"""
Quick helper to create initial REAL vs DECOY labels
Based on detection results
"""

import json
from pathlib import Path

def create_initial_labels():
    """Create initial labels based on detection results"""

    # Load detection results
    detections_file = Path('../results/baseline_detections/baseline_detections.json')
    with open(detections_file, 'r') as f:
        detections = json.load(f)

    # Categorize images
    # Simple heuristic: Images with detections are likely REAL
    # Images without detections are likely scenes without fire extinguishers
    # For this project, we'll treat no-detection as DECOY for evaluation

    real_images = []
    decoy_images = []

    for img_name, data in detections.items():
        if data['num_detections'] > 0:
            real_images.append(img_name)
        else:
            decoy_images.append(img_name)

    # Write to labels file
    labels_file = Path('../data/annotations/real_vs_decoy_labels.txt')

    with open(labels_file, 'w') as f:
        f.write("# Fire Extinguisher Real vs Decoy Labels\n")
        f.write("# Auto-generated initial labels based on detections\n")
        f.write("# Review and correct as needed\n")
        f.write("#\n")
        f.write("# Format: filename | type | notes\n")
        f.write("\n")
        f.write("# REAL FIRE EXTINGUISHERS (physical objects)\n")
        f.write("# Images where detector found fire extinguishers\n")
        f.write("\n")

        for img in sorted(real_images):
            f.write(f"{img}\n")

        f.write("\n")
        f.write("# DECOY FIRE EXTINGUISHERS (printed/displayed or no fire extinguisher)\n")
        f.write("# Images where no fire extinguishers were detected\n")
        f.write("\n")

        for img in sorted(decoy_images):
            f.write(f"{img}\n")

    print(f"✓ Created labels file: {labels_file}")
    print(f"  REAL images:  {len(real_images)}")
    print(f"  DECOY images: {len(decoy_images)}")
    print()
    print("Note: These are initial labels based on detections.")
    print("Review and correct manually if needed.")

if __name__ == "__main__":
    create_initial_labels()
