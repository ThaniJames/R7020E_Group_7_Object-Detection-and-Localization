#!/usr/bin/env python3
"""
Prepare fire extinguisher dataset for YOLOv11 training
Creates proper train/val split and data.yaml
"""

import os
import shutil
import random
from pathlib import Path

def prepare_dataset():
    """Organize dataset for YOLO training"""

    print("=" * 70)
    print("PREPARING DATASET FOR YOLO TRAINING")
    print("=" * 70)
    print()

    # Source paths
    images_dir = Path("../data/to_label")
    labels_dir = Path("../data/yolo_labels")

    # Destination paths
    dataset_root = Path("../data/fire_extinguisher_dataset")
    train_images = dataset_root / "train" / "images"
    train_labels = dataset_root / "train" / "labels"
    val_images = dataset_root / "val" / "images"
    val_labels = dataset_root / "val" / "labels"

    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Created dataset directories:")
    print(f"  {dataset_root}")
    print()

    # Get all images and labels
    all_images = sorted(images_dir.glob("*.png"))
    print(f"Found {len(all_images)} images")

    # Check labels
    labeled_count = 0
    for img in all_images:
        label_path = labels_dir / f"{img.stem}.txt"
        if label_path.exists():
            labeled_count += 1

    print(f"Found {labeled_count} label files")
    print()

    # Split: 80% train, 20% validation
    random.seed(42)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    print(f"Split: {len(train_imgs)} training, {len(val_imgs)} validation")
    print()

    # Copy training data
    print("Copying training data...")
    for img_path in train_imgs:
        # Copy image
        shutil.copy2(img_path, train_images / img_path.name)

        # Copy label (or create empty if missing)
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, train_labels / f"{img_path.stem}.txt")
        else:
            # Create empty label file
            (train_labels / f"{img_path.stem}.txt").touch()

    print(f"✓ Copied {len(train_imgs)} training images and labels")

    # Copy validation data
    print("Copying validation data...")
    for img_path in val_imgs:
        # Copy image
        shutil.copy2(img_path, val_images / img_path.name)

        # Copy label (or create empty if missing)
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, val_labels / f"{img_path.stem}.txt")
        else:
            # Create empty label file
            (val_labels / f"{img_path.stem}.txt").touch()

    print(f"✓ Copied {len(val_imgs)} validation images and labels")
    print()

    # Create data.yaml
    yaml_content = f"""# Fire Extinguisher Detection Dataset
# Auto-labeled with YOLOv11 pre-trained model

path: {dataset_root.absolute()}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['fire_extinguisher']

# Dataset info
total_images: {len(all_images)}
labeled_images: {labeled_count}
train_images: {len(train_imgs)}
val_images: {len(val_imgs)}
"""

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ Created {yaml_path}")
    print()

    # Summary
    print("=" * 70)
    print("DATASET READY FOR TRAINING!")
    print("=" * 70)
    print(f"Location: {dataset_root.absolute()}")
    print(f"Config: {yaml_path}")
    print()
    print("Dataset structure:")
    print("  train/")
    print(f"    images/ ({len(train_imgs)} images)")
    print(f"    labels/ ({len(train_imgs)} labels)")
    print("  val/")
    print(f"    images/ ({len(val_imgs)} images)")
    print(f"    labels/ ({len(val_imgs)} labels)")
    print()
    print("NEXT STEP:")
    print("  Upload dataset to Google Colab and train YOLOv11!")
    print("=" * 70)

    return dataset_root

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    prepare_dataset()
