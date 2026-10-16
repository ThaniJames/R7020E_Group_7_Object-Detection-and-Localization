"""
Download Fire Extinguisher Dataset from Roboflow via API
Bypasses the "Generate Version" paywall by downloading raw annotations directly
"""

import os
import json
import requests
from pathlib import Path
import shutil

# Configuration from .env
API_KEY = "ukXcCk2tVYVS5jYLHEqf"
WORKSPACE = "kaycee-hn2a6"
PROJECT = "fireextinguisher-z5atr-6ktlv"

# Output directories
OUTPUT_DIR = Path("../data/roboflow_export")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

def setup_directories():
    """Create output directory structure"""
    print("Setting up directories...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    LABELS_DIR.mkdir(exist_ok=True)
    print(f"✓ Created: {OUTPUT_DIR}")

def get_project_images():
    """Get list of all images in the project via API"""
    print("\nFetching project images from Roboflow...")

    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/images"
    params = {"api_key": API_KEY}

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    data = response.json()
    print(f"✓ Found {len(data.get('images', []))} images")
    return data.get('images', [])

def download_image_with_annotation(image_info):
    """Download single image and its annotation"""
    image_id = image_info.get('id')
    image_name = image_info.get('name')

    if not image_id or not image_name:
        print(f"⚠️  Skipping invalid image: {image_info}")
        return False

    print(f"Processing: {image_name}")

    # Download image
    try:
        image_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/images/{image_id}"
        params = {"api_key": API_KEY}

        img_response = requests.get(image_url, params=params)

        if img_response.status_code == 200:
            # Save image
            img_path = IMAGES_DIR / image_name
            with open(img_path, 'wb') as f:
                f.write(img_response.content)

            # Get annotation data from response
            annotation_data = img_response.json()

            # Convert to YOLO format and save
            save_yolo_annotation(image_name, annotation_data)

            return True
        else:
            print(f"  ❌ Failed to download: {img_response.status_code}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def save_yolo_annotation(image_name, annotation_data):
    """Convert Roboflow annotation to YOLO format and save"""
    # YOLO format: class_id center_x center_y width height (normalized 0-1)

    annotations = annotation_data.get('annotations', [])
    image_width = annotation_data.get('width', 1)
    image_height = annotation_data.get('height', 1)

    if not annotations:
        print(f"  ⚠️  No annotations for {image_name}")
        return

    # Create .txt file for labels
    label_name = Path(image_name).stem + '.txt'
    label_path = LABELS_DIR / label_name

    yolo_lines = []

    for ann in annotations:
        # Roboflow format: {x, y, width, height} in pixels
        x = ann.get('x', 0)
        y = ann.get('y', 0)
        w = ann.get('width', 0)
        h = ann.get('height', 0)
        class_name = ann.get('class', 'fire_extinguisher')

        # Convert to YOLO format (normalized)
        center_x = (x + w/2) / image_width
        center_y = (y + h/2) / image_height
        norm_width = w / image_width
        norm_height = h / image_height

        # Class ID (0 for fire_extinguisher - single class)
        class_id = 0

        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
        yolo_lines.append(yolo_line)

    # Save to file
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"  ✓ Saved {len(yolo_lines)} annotations")

def create_dataset_yaml():
    """Create dataset.yaml for YOLOv11 training"""
    print("\nCreating dataset.yaml...")

    yaml_content = f"""# Fire Extinguisher Detection Dataset
# Downloaded from Roboflow via API

path: {OUTPUT_DIR.absolute()}
train: images
val: images
test: images

# Classes
nc: 1  # number of classes
names: ['fire_extinguisher']
"""

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✓ Created: {yaml_path}")

def main():
    """Main execution"""
    print("=" * 60)
    print("Roboflow Dataset Downloader (API - No Credits Required)")
    print("=" * 60)

    # Step 1: Setup
    setup_directories()

    # Step 2: Get image list
    images = get_project_images()

    if not images:
        print("\n❌ Failed to fetch images. Check your API key and project name.")
        return

    # Step 3: Download each image with annotations
    print(f"\nDownloading {len(images)} images with annotations...")
    success_count = 0

    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", end=" ")
        if download_image_with_annotation(img):
            success_count += 1

    # Step 4: Create dataset.yaml
    create_dataset_yaml()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"✓ Images downloaded: {success_count}/{len(images)}")
    print(f"✓ Dataset location: {OUTPUT_DIR.absolute()}")
    print(f"✓ Images: {IMAGES_DIR.absolute()}")
    print(f"✓ Labels: {LABELS_DIR.absolute()}")
    print(f"✓ Config: {OUTPUT_DIR / 'data.yaml'}")
    print("\nReady for YOLOv11 training in Google Colab!")
    print("=" * 60)

if __name__ == "__main__":
    main()
