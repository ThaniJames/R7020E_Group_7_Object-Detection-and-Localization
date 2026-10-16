# LabelImg - Fire Extinguisher Labeling Guide

## Quick Start

### Launch LabelImg:
```bash
cd /Users/kelechiezekwe/Workspace/Computer_vision_lab/Project_work
./scripts/launch_labelimg.sh
```

### Check Progress:
```bash
python scripts/check_labeling_progress.py
```

---

## LabelImg Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **W** | Create bounding box |
| **D** | Next image |
| **A** | Previous image |
| **Del** | Delete selected box |
| **Ctrl+S** | Save |
| **Space** | Flag current image as verified |

---

## Step-by-Step Instructions

### 1. **Initial Setup (First Time Only):**
   - LabelImg will open automatically
   - Click **"Open Dir"** → Navigate to: `Project_work/data/to_label`
   - Click **"Change Save Dir"** → Navigate to: `Project_work/data/yolo_labels`
   - Verify format is set to **"YOLO"** (top menu bar)

### 2. **Label Each Image:**
   1. Press **W** to start drawing a box
   2. Click and drag around the fire extinguisher
   3. Release mouse
   4. Select class: **"fire_extinguisher"**
   5. Press **D** to go to next image
   6. Repeat!

### 3. **Tips for Good Labels:**
   - Draw box tightly around fire extinguisher
   - Include the entire object (top to bottom)
   - Don't include too much background
   - If multiple fire extinguishers → draw multiple boxes
   - If fire extinguisher is partially visible → still label it
   - If it's a DECOY (poster/printed image) → still label it (we'll filter later)

### 4. **Track Real vs Decoy:**
   As you label, note which images are REAL vs DECOY in:
   `data/annotations/real_vs_decoy_labels.txt`

---

## Progress Tracking

- **Total images:** 127
- **Estimated time:** 60-90 minutes total
- **Average:** 30-45 seconds per image

**You'll get faster as you go!**

---

## What Happens Next?

After labeling all 127 images:

1. ✅ Check progress: `python scripts/check_labeling_progress.py`
2. ✅ Upload dataset to Colab
3. ✅ Train YOLOv11 model
4. ✅ Evaluate performance
5. ✅ Submit project!

---

## Troubleshooting

### LabelImg won't open?
```bash
pip install labelImg
```

### Can't find images?
Make sure you're in `Project_work` directory:
```bash
cd /Users/kelechiezekwe/Workspace/Computer_vision_lab/Project_work
```

### Wrong format (showing PascalVOC)?
In LabelImg menu bar: **View → YOLO**

### Lost your work?
All labels auto-save to `data/yolo_labels/*.txt`

---

## Ready to Start!

Run this command now:
```bash
./scripts/launch_labelimg.sh
```

**Happy labeling! 🏷️**
