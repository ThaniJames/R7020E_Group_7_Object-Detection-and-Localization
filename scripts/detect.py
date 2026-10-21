import sys, json
sys.path.insert(0, "src")
from ultralytics import YOLO
from rgbd import pair_frames, load_intrinsics, object_centre

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else "models/best.pt"
model = YOLO(WEIGHTS)
K = load_intrinsics(SPLIT)

results = []
for p in pair_frames(SPLIT):
    det = model(p.color(), conf=0.1, verbose=False)[0]          # low conf: catch more, filter later
    depth = p.depth_m()
    for box, conf in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist()):
        centre = object_centre(depth, box, K) 
        results.append({
            "stamp": p.stamp_ns,
            "box": [round(b, 1) for b in box],
            "conf": round(conf, 3),
            "xyz": None if centre is None else [round(float(c), 3) for c in centre],
        })
    print(p.color_path.name[-23:], len(det.boxes), "boxes")

json.dump(results, open(f"outputs/detections_xyz_{SPLIT}.json", "w"), indent=1)
print(len(results), "detections saved")
