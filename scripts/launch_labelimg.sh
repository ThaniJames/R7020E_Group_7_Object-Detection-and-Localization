#!/bin/bash
# Launch LabelImg for Fire Extinguisher Labeling

echo "=========================================="
echo "LabelImg - Fire Extinguisher Labeling"
echo "=========================================="
echo ""
echo "Opening LabelImg..."
echo ""
echo "INSTRUCTIONS:"
echo "1. Click 'Open Dir' → Select: Project_work/data/to_label"
echo "2. Click 'Change Save Dir' → Select: Project_work/data/yolo_labels"
echo "3. Make sure 'YOLO' format is selected (not PascalVOC)"
echo "4. Press 'W' to draw bounding box"
echo "5. Select 'fire_extinguisher' class"
echo "6. Press 'D' for next image"
echo "7. Repeat for all 127 images"
echo ""
echo "Progress will be saved automatically!"
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

# Launch labelImg with correct arguments
# Usage: labelImg [IMAGE_DIR] [PRE-DEFINED CLASS FILE]
labelImg data/to_label data/classes.txt
