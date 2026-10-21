import sys, json
sys.path.insert(0, "src")
import numpy as np
import open3d as o3d
from rgbd import pair_frames, load_intrinsics, depth_to_points

SPLIT = "test"
K = load_intrinsics(SPLIT)
dets = json.load(open(f"outputs/tracks_{SPLIT}.json"))
pairs = {p.stamp_ns: p for p in pair_frames(SPLIT)}

o3d.utility.random.seed(0)
for stamp in sorted({d["stamp"] for d in dets}):
    depth = pairs[stamp].depth_m()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_to_points(depth, K)))
    (a, b, c, d0), _ = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    for d in dets:
        if d["stamp"] == stamp:
            x, y, z = d["xyz"]
            d["height"] = (a * x + b * y + c * z + d0) * np.sign(d0)

tracks = {}
for d in dets:
    tracks.setdefault(d["track"], []).append(d)
    
#TODO: Update after Task 2
for tid, ds in sorted(tracks.items()):
    if len(ds) < 3:
        continue
    h = np.array([d["height"] for d in ds])
    print(f"track {tid:2d}: {len(ds):2d} frames | height above floor {h.mean():.2f} ±{h.std():.2f} m")

json.dump(dets, open(f"outputs/heights_{SPLIT}.json", "w"), indent=1)
