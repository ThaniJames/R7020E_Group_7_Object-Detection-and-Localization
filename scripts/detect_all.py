"""Detections on ALL colour images.

Output: outputs/detections_{split}.json, one entry per box, fields as in src/contracts.py Detection:
    {"stamp": int (number in the colour image filename), "box": [x1, y1, x2, y2] pixels, "conf": float}
All boxes with conf >= 0.1 are kept, real and fake.
"""
import sys, json
sys.path.insert(0, "src")
from pathlib import Path
import cv2
from ultralytics import YOLO
from rgbd import DATA_ROOT, _stamp

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "models/best.pt"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
model = YOLO(WEIGHTS)

for split in ["test", "eval"]:
    images = sorted((DATA_ROOT / split / "camera_color_image_raw").glob("*.png"), key=_stamp)
    results = []
    for path in images:
        det = model(cv2.imread(str(path)), conf=0.1, verbose=False)[0]
        for box, conf in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist()):
            results.append({"stamp": _stamp(path), "box": [round(b, 1) for b in box], "conf": round(conf, 3)})
    json.dump(results, open(OUT / f"detections_{split}.json", "w"), indent=1)
    print(f"{split}: {len(images)} images, {len(results)} boxes -> {OUT / f'detections_{split}.json'}")
