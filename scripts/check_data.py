#Sanity checks: frame pairing, back-projection vs provided point clouds, colour/depth alignment."""
import sys
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rgbd import DATA_ROOT, _stamp, depth_to_points, load_intrinsics, pair_frames

OUT = Path(__file__).resolve().parents[1] / "outputs" / "check_data"
OUT.mkdir(parents=True, exist_ok=True)


# Depth image whose timestamp is closest to stamp.
def nearest_depth_frame(split: str, stamp: int):
    depths = sorted((DATA_ROOT / split / "camera_depth_image_raw").glob("*.png"), key=_stamp)
    return min(depths, key=lambda p: abs(_stamp(p) - stamp))

#  Back-project the depth image closest in time to a provided cloud; report nearest-neighbour distances.
def compare_with_pcd(split: str, pcd_dir: str, K):
    pcd_path = sorted((DATA_ROOT / split / pcd_dir).glob("*.pcd"), key=_stamp)[len(list((DATA_ROOT / split / pcd_dir).glob("*.pcd"))) // 2]
    ref = np.asarray(o3d.io.read_point_cloud(str(pcd_path)).points)
    depth_path = nearest_depth_frame(split, _stamp(pcd_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    ours = depth_to_points(depth, K)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref)))
    sample = ours[np.random.default_rng(0).choice(len(ours), 2000, replace=False)]
    dists = np.array([np.sqrt(tree.search_knn_vector_3d(p, 1)[2][0]) for p in sample])
    gap = abs(_stamp(pcd_path) - _stamp(depth_path)) / 1e6
    print(f"  {pcd_dir}: ref {len(ref)} pts vs ours {len(ours)} pts, time gap {gap:.0f} ms, "
          f"NN dist median {np.median(dists)*1000:.1f} mm, p90 {np.percentile(dists, 90)*1000:.1f} mm")

# Save the colour image next to a colour+depth blend, to check alignment by eye.
def overlay(pair, name):
    color = pair.color()
    depth = pair.depth_m()
    vis = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 8), alpha=255 / 8), cv2.COLORMAP_TURBO)
    vis[depth == 0] = 0
    edges = cv2.Canny(cv2.convertScaleAbs(depth, alpha=255 / 8), 30, 90)
    blend = cv2.addWeighted(color, 0.6, vis, 0.4, 0)
    blend[edges > 0] = (255, 255, 255)
    cv2.imwrite(str(OUT / name), np.hstack([color, blend]))


if __name__ == "__main__":
    for split in ["test", "eval"]:
        K = load_intrinsics(split, "color")
        Kd = load_intrinsics(split, "depth")
        pairs = pair_frames(split)
        n_color = len(list((DATA_ROOT / split / "camera_color_image_raw").glob("*.png")))
        print(f"[{split}] colour frames {n_color}, paired (<=50 ms) {len(pairs)}, "
              f"median gap {np.median([p.gap_ms for p in pairs]):.1f} ms")
        print(f"  colour K == depth K: {np.allclose(K.K, Kd.K)}; colour D = {np.round(K.D, 3)}")
        compare_with_pcd(split, "depth_registered_colored_pointclouds", K)
        compare_with_pcd(split, "camera_depth_points", K)
        for i in (0, len(pairs) // 2):
            overlay(pairs[i], f"{split}_overlay_{i:03d}.png")
    print(f"overlays written to {OUT}")
