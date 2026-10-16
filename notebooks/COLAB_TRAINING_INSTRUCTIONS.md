# YOLOv11 Training in Google Colab - Instructions

## **Step-by-Step Guide**

### **1. Upload Dataset to Colab**

Open Google Colab: https://colab.research.google.com/

Run these cells:

```python
# Cell 1: Check GPU
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

```python
# Cell 2: Install Ultralytics
!pip install -q ultralytics

from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import glob
```

```python
# Cell 3: Upload dataset ZIP
from google.colab import files

print("Upload fire_extinguisher_dataset.zip from your computer...")
uploaded = files.upload()

# Extract dataset
!unzip -q fire_extinguisher_dataset.zip
!ls -lh
```

```python
# Cell 4: Verify dataset structure
DATA_PATH = '/content/data/fire_extinguisher_dataset'

print("Dataset structure:")
!ls -lh {DATA_PATH}
!echo "\nTrain images:"
!ls {DATA_PATH}/train/images | wc -l
!echo "Train labels:"
!ls {DATA_PATH}/train/labels | wc -l
!echo "\nVal images:"
!ls {DATA_PATH}/val/images | wc -l
!echo "Val labels:"
!ls {DATA_PATH}/val/labels | wc -l
```

```python
# Cell 5: Read data.yaml
with open(f"{DATA_PATH}/data.yaml", 'r') as f:
    data_config = yaml.safe_load(f)

print("Dataset Configuration:")
print(yaml.dump(data_config, default_flow_style=False))
```

---

### **2. Train YOLOv11**

```python
# Cell 6: Initialize model
MODEL_SIZE = 'n'  # nano - fast training
model = YOLO(f'yolo11{MODEL_SIZE}.pt')

print(f"Model loaded: YOLOv11{MODEL_SIZE}")
```

```python
# Cell 7: Configure training
training_config = {
    'data': f"{DATA_PATH}/data.yaml",
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'patience': 15,
    'project': 'fire_extinguisher_detection',
    'name': 'yolo11n_v1',
    'device': 0,
    'augment': True,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 15.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
}

print("Training configuration:")
for key, value in training_config.items():
    print(f"  {key}: {value}")
```

```python
# Cell 8: START TRAINING (this will take ~2-3 hours)
print("Starting training...")
print("Estimated time: ~3 hours on T4 GPU")

results = model.train(**training_config)

print("\n✓ Training complete!")
```

---

### **3. Evaluate Results**

```python
# Cell 9: Validation metrics
metrics = model.val()

print("\n=== Validation Metrics ===")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

```python
# Cell 10: View training curves
from IPython.display import Image, display

results_dir = "fire_extinguisher_detection/yolo11n_v1"

print("Training curves:")
display(Image(filename=f"{results_dir}/results.png"))
```

```python
# Cell 11: Test predictions
best_model = YOLO(f"{results_dir}/weights/best.pt")

# Test on sample validation images
test_images = glob.glob(f"{DATA_PATH}/val/images/*")[:5]

for img_path in test_images:
    results = best_model.predict(img_path, conf=0.25, save=True)
    print(f"✓ Predicted: {os.path.basename(img_path)}")
```

---

### **4. Download Trained Weights**

```python
# Cell 12: Download best weights
from google.colab import files

print("Downloading best.pt...")
files.download(f"{results_dir}/weights/best.pt")

print("Downloading results.png...")
files.download(f"{results_dir}/results.png")

print("\n✓ Downloads complete!")
print("Save best.pt to: Project_work/models/weights/")
```

---

## **Expected Results**

After training, you should see:
- **mAP50:** ~0.60-0.80 (depends on label quality)
- **Precision:** ~0.70-0.85
- **Recall:** ~0.65-0.80

**Note:** Since we used auto-labeled data (16 real detections + 111 empty labels), accuracy may be lower than ideal. If results are poor, we can manually label more images and retrain.

---

## **Files to Download**

After training completes, download these files:

1. `best.pt` → Save to `models/weights/best.pt`
2. `results.png` → Save to `results/training_curves.png`
3. `confusion_matrix.png` → Save to `results/confusion_matrix.png`

---

## **Next Steps After Training**

1. Test model on evaluation images
2. Calculate TPR and FPR metrics
3. Create baseline detector script
4. Prepare final presentation

---

**Estimated Total Time:**
- Upload dataset: 2-3 minutes
- Training: 2-3 hours
- Evaluation: 5 minutes
- Download results: 2 minutes

**Total: ~3 hours**
