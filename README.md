# Project 1: Fire Extinguisher Detection and Localization

**Team:** 3 members  
**Deadline:** October 30, 2026, 11:59pm  
**Your Role:** Object detection (bounding boxes), YOLO training, baseline, evaluation

---

## **Project Overview**

Build a system that:
1. Detects fire extinguishers using **YOLOv11**
2. Distinguishes real extinguishers from printed decoys (teammates' part)
3. Estimates 3D position of real objects (teammates' part)

**Your Responsibilities:**
- Training data preparation & augmentation
- YOLOv11 model training
- Baseline detector (no decoy filter)
- Evaluation script (TPR, FPR metrics)

---

## **Performance Targets**

| Metric | Target | Definition |
|--------|--------|------------|
| **Real-object TPR** | ≥ 0.75 | True Positive Rate for real fire extinguishers |
| **Decoy FPR** | ≤ 0.15 | False Positive Rate for decoys |

*Detection is correct when bbox center is inside reference bbox*

---

## **Datasets**

### 1. Training Data (Roboflow)
- **Source:** https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr
- **Forked:** https://app.roboflow.com/kaycee-hn2a6/fireextinguisher-z5atr-6ktlv/train
- **Use:** Train YOLOv11 model
- **Status:** Labeled with bounding boxes

### 2. Evaluation Data (Custom)
- **Location:** `raw/eval/camera_color_image_raw/`
- **Images:** 127 RGB images
- **Content:** Real fire extinguishers + printed decoys
- **Labels:** Need to create/verify annotations
- **Depth data:** `raw/eval/camera_depth_image_raw/` (for teammates)

---

## **Workflow**

### **Phase 1: Data Preparation** (Roboflow + Colab)
1. Access Roboflow dataset
2. Apply augmentation (flip, rotate, brightness, etc.)
3. Export in YOLOv11 format
4. Upload to Google Drive

### **Phase 2: Model Training** (Google Colab - FREE GPU)
⚠️ **Local RAM <8GB → ALL training must be in Colab**

1. Open `notebooks/train_yolov11.ipynb` in Colab
2. Mount Google Drive
3. Load augmented dataset
4. Train YOLOv11
5. Save best weights to `models/weights/best.pt`

### **Phase 3: Baseline Implementation** (Colab)
- Detector with **NO decoy filter**
- Accept all detections as real
- Use median depth for 3D position

### **Phase 4: Evaluation** (Colab - light batching for low RAM)
1. Run trained model on eval dataset
2. Compare baseline vs final system
3. Compute TPR, FPR, raw counts (TP, FP, FN)
4. Generate visualizations

### **Phase 5: Report** (Local)
- Create presentation slides (8-12 slides)
- Write demonstration script
- Package submission ZIP

---

## **Project Structure**

```
Project_work/
├── data/
│   ├── train/              # Roboflow training data (exported)
│   ├── val/                # Validation split
│   ├── test/               # Evaluation dataset (127 images)
│   └── annotations/        # Label files
├── notebooks/
│   └── train_yolov11.ipynb # Colab training notebook
├── scripts/
│   ├── baseline/
│   │   └── baseline_detector.py
│   ├── evaluation/
│   │   ├── evaluate.py     # TPR, FPR computation
│   │   └── metrics.py
│   └── visualization/
│       └── visualize_results.py
├── models/
│   ├── weights/
│   │   └── best.pt         # Trained YOLOv11 weights
│   └── exports/
├── results/
│   ├── baseline/
│   │   └── baseline_results.json
│   ├── final/
│   │   └── final_results.json
│   └── figures/            # Visual evidence
├── raw/                    # Original evaluation dataset
│   └── eval/
│       ├── camera_color_image_raw/    # 127 RGB images
│       └── camera_depth_image_raw/    # Depth data
├── README.md
└── requirements.txt
```

---

## **Hardware Constraints**

**Local Machine:** <8GB RAM, macOS  
**Solution:** Use Google Colab for ALL heavy computation

| Task | Where to Run |
|------|--------------|
| Training YOLOv11 | Google Colab (T4 GPU) |
| Inference/Evaluation | Google Colab (batch processing) |
| Data augmentation | Roboflow or Colab |
| Code editing | Local |
| Report writing | Local |
| Results visualization | Local (from Colab outputs) |

---

## **Key Files**

- `notebooks/train_yolov11.ipynb` - Main training notebook (run in Colab)
- `scripts/baseline/baseline_detector.py` - Baseline with no filter
- `scripts/evaluation/evaluate.py` - TPR/FPR computation
- `requirements.txt` - Minimal local dependencies

---

## **Next Steps**

1. ✅ Project structure created
2. ⬜ Access Roboflow dataset, apply augmentation
3. ⬜ Create Colab training notebook
4. ⬜ Label evaluation dataset (real/decoy annotations)
5. ⬜ Train YOLOv11 on Roboflow data
6. ⬜ Implement baseline detector
7. ⬜ Create evaluation script
8. ⬜ Run experiments, collect results
9. ⬜ Generate presentation slides
10. ⬜ Package submission

---

## **Resources**

- **YOLOv11 Docs:** https://docs.ultralytics.com/models/yolo11/
- **Roboflow:** https://roboflow.com/
- **Google Colab:** https://colab.research.google.com/
- **Project Requirements:** `Projects.pdf`

---

## **Team Contributions**

| Member | Responsibilities |
|--------|------------------|
| **Kelechi (You)** | Object detection, YOLO training, baseline, evaluation script |
| **Teammate 2** | Real vs decoy filtering (cues: flatness, curvature, depth, IR texture) |
| **Teammate 3** | 3D localization (depth-to-3D, tracking, coordinates, accuracy check) |

**Note:** You must understand the complete system for oral examination.

---

**GitHub:** kelechiezekwe  
**Due:** October 30, 11:59pm
