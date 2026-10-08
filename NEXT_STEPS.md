# Next Steps - Project 1 Implementation

**Current Status:** ✅ Project structure set up  
**Hardware:** <8GB RAM (all heavy work → Colab)  
**Model:** YOLOv11  
**Deadline:** October 30, 2026

---

## Immediate Actions (This Week)

### 1. Access Roboflow Dataset ⚠️ **DO THIS FIRST**

**Your Roboflow Project:**
- https://app.roboflow.com/kaycee-hn2a6/fireextinguisher-z5atr-6ktlv/train

**Steps:**
1. Log in to Roboflow
2. Go to your project workspace
3. Navigate to the dataset version
4. Click "Generate" → Select "YOLOv8" format (works for YOLOv11)
5. Copy the API key and download code snippet
6. **Paste API key into Colab notebook** (Section 3, line: `API_KEY = "..."`)

**What you'll get:**
- Training images with fire extinguisher bounding boxes
- Validation split
- `data.yaml` configuration file

---

### 2. Open Colab Notebook

**File:** `notebooks/train_yolov11.ipynb`

**How to open:**
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Select `train_yolov11.ipynb` from your computer
4. **CRITICAL:** Runtime → Change runtime type → **T4 GPU**

**Or:** Upload to GitHub first, then open from GitHub in Colab

---

### 3. Label Evaluation Dataset (127 images)

**Location:** `raw/eval/camera_color_image_raw/`

**What to label:**
- Real fire extinguishers (class: "real")
- Printed/displayed decoys (class: "decoy")
- Bounding boxes around each

**Labeling Options:**

**Option A: Roboflow (Recommended)**
1. Create new project "Fire_Extinguisher_Eval"
2. Upload 127 images from `raw/eval/camera_color_image_raw/`
3. Label each image (real vs decoy + bounding box)
4. Export annotations in YOLO format

**Option B: LabelImg (Local - but slow with low RAM)**
```bash
# NOT recommended for <8GB RAM
pip install labelImg
labelImg
```

**Option C: CVAT (Online)**
- https://cvat.ai/
- Free online labeling tool

---

### 4. Start Training (In Colab)

**After getting Roboflow API key:**

1. Open `train_yolov11.ipynb` in Colab
2. Enable GPU (T4)
3. Run cells 1-8 sequentially
4. Training will take ~3-4 hours for 100 epochs
5. Download `best.pt` weights when done

**Monitor training:**
- Watch loss curves decrease
- Check mAP50 metric (should reach >0.7)
- Early stopping will trigger if no improvement

---

## This Week's Checklist

- [ ] Access Roboflow training dataset
- [ ] Get Roboflow API key
- [ ] Upload Colab notebook to Google Colab
- [ ] Enable T4 GPU runtime
- [ ] Start first training run (YOLOv11n - 100 epochs)
- [ ] Begin labeling eval dataset (at least 50/127 images)
- [ ] Download trained weights (`best.pt`)

---

## Next Week's Tasks

- [ ] Complete eval dataset labeling (all 127 images)
- [ ] Create baseline detector script
- [ ] Implement evaluation metrics (TPR, FPR)
- [ ] Run experiments and collect results
- [ ] Create visualizations
- [ ] Start presentation slides

---

## Files You'll Create

### From Colab:
1. `best.pt` → `models/weights/best.pt`
2. `training_summary.json` → `results/training_summary.json`
3. `results.png` → `results/figures/training_curves.png`

### Locally (low RAM - simple scripts only):
1. `scripts/baseline/baseline_detector.py`
2. `scripts/evaluation/evaluate.py`
3. `scripts/visualization/visualize_results.py`
4. Presentation slides (PowerPoint/Google Slides)

---

## Troubleshooting

**Problem:** Colab GPU disconnects during training  
**Solution:** 
- Use Google Drive mount to save checkpoints
- Training resumes from last checkpoint
- Consider shorter training runs (50 epochs)

**Problem:** Roboflow API key not working  
**Solution:**
- Regenerate key in Roboflow settings
- Check workspace and project names are correct

**Problem:** Out of memory in Colab  
**Solution:**
- Reduce batch size: `BATCH_SIZE = 8` instead of 16
- Use YOLOv11n (nano) instead of YOLOv11s (small)

**Problem:** Can't label 127 images quickly  
**Solution:**
- Label in batches (30 images per session)
- Ask teammates to help split labeling work
- Use Roboflow Smart Polygon tool for faster boxing

---

## Questions to Clarify with Teammates

1. **Integration:** How will you pass bounding boxes to their decoy filter?
2. **Data format:** What format do they need for 3D localization?
3. **Evaluation:** Who runs the final integrated pipeline?

---

## Resources

- **YOLOv11 Docs:** https://docs.ultralytics.com/models/yolo11/
- **Roboflow Guide:** https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/
- **Colab GPU Guide:** https://research.google.com/colaboratory/faq.html#gpu-availability
- **CVAT Labeling:** https://cvat.ai/

---

**Author:** Kelechi Ezekwe (@kelechiezekwe)  
**Updated:** October 7, 2026
