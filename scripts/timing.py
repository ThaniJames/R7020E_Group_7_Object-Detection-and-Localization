"""Processing time per frame for each stage of the final pipeline (for the speed trade-off)."""
import sys, time
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import json
import numpy as np
from ultralytics import YOLO
from rgbd import pair_frames, load_intrinsics, object_centre, DATA_ROOT
from extract_depth_cues import extract_cues, read_focal_lengths

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else "models/best.pt"

model = YOLO(WEIGHTS)
K = load_intrinsics(SPLIT)
pairs = pair_frames(SPLIT)
FX, FY = read_focal_lengths(sorted((DATA_ROOT / SPLIT / "camera_depth_camera_info").glob("*.txt"))[0])
FILTER = json.load(open("outputs/models/real_decoy_filter.json"))


def real_score(depth, box):
    c = extract_cues(depth, box, FX, FY)
    h, r = c["projected_box_height_m"], c["relative_ring_minus_core"]
    return float(h is not None and r is not None and h >= FILTER["height_threshold_m"] and r >= FILTER["relative_depth_threshold"])


model(pairs[0].color(), verbose=False) 

t = {"detect": [], "filter": [], "3d": []}
for p in pairs:
    img, depth = p.color(), p.depth_m()
    t0 = time.perf_counter()
    boxes = model(img, conf=0.1, verbose=False)[0].boxes.xyxy.tolist()
    t1 = time.perf_counter()
    scores = [real_score(depth, b) for b in boxes]
    t2 = time.perf_counter()
    [object_centre(depth, b, K) for b, s in zip(boxes, scores) if s >= 0.5]
    t3 = time.perf_counter()
    t["detect"].append(t1 - t0); t["filter"].append(t2 - t1); t["3d"].append(t3 - t2)

total = sum(np.mean(v) for v in t.values())
for k, v in t.items():
    print(f"{k:7} {1000 * np.mean(v):6.1f} ms/frame")
print(f"total   {1000 * total:6.1f} ms/frame = {1 / total:.0f} FPS ({len(pairs)} frames, {WEIGHTS})")
