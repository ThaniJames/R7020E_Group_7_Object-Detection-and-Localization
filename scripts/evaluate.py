"""Baseline vs final system, per annotated object (the course metric).

Input:  labels/gt_{split}.json, outputs/scored_detections_xyz_{split}.json (Task 2 output: detections + xyz + real_score)
Output: outputs/results_{split}.json   (table + one record per annotated object, used for the visual evidence)

A detection belongs to an annotated object when the centre of its box lies inside the reference box.
    TPR (real)  = real objects with an accepted detection / all real objects
    FPR (decoy) = decoy objects with an accepted detection / all decoy objects

Baseline: every detection with conf >= CONF is accepted as real, 3D = median depth of the whole box.
Final:    conf >= CONF and real_score >= ACCEPT, 3D = histogram front surface + radius (object_centre).

3D check (no 3D ground truth exists): the real extinguisher stands still on the floor, so its height above the
RANSAC floor plane must stay the same while the camera moves. Lower spread = more stable 3D position.
"""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np
import open3d as o3d
from rgbd import pair_frames, load_intrinsics, depth_to_points, box_position_median

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
CONF = 0.10             # detector confidence threshold, same for both systems
ACCEPT = 0.5            # real_score threshold (Task 2 filter output: 1 = real, 0 = decoy)


def inside(box, ref):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ref[0] <= cx <= ref[2] and ref[1] <= cy <= ref[3]


def floor_plane(depth, K):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_to_points(depth, K)))
    (a, b, c, d), _ = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    return np.array([a, b, c, d]) * np.sign(d)          # sign: camera side of the floor is positive


def run(gt, dets, accept):
    """Per annotated object: the best accepted detection whose centre is inside it (or None)."""
    records = []
    for g in gt:
        hits = [d for d in dets if d["stamp"] == g["stamp"] and accept(d) and inside(d["box"], g["box"])]
        best = max(hits, key=lambda d: d["conf"]) if hits else None
        records.append({**g, "det": best})
    real = [r for r in records if r["cls"] == "real"]
    decoy = [r for r in records if r["cls"] == "decoy"]
    tp = sum(r["det"] is not None for r in real)
    fp = sum(r["det"] is not None for r in decoy)
    # accepted detections that belong to no annotated object at all (background)
    other = sum(accept(d) and not any(g["stamp"] == d["stamp"] and inside(d["box"], g["box"]) for g in gt) for d in dets)
    table = {"real": len(real), "tp": tp, "fn": len(real) - tp, "tpr": round(tp / len(real), 3),
             "decoy": len(decoy), "fp": fp, "tn": len(decoy) - fp, "fpr": round(fp / len(decoy), 3),
             "accepted_background": other}
    return table, records


if __name__ == "__main__":
    K = load_intrinsics(SPLIT)
    gt = json.load(open(f"labels/gt_{SPLIT}.json"))
    dets = json.load(open(f"outputs/scored_detections_xyz_{SPLIT}.json"))
    pairs = {p.stamp_ns: p for p in pair_frames(SPLIT)}

    systems = {
        "baseline": lambda d: d["conf"] >= CONF,
        "final": lambda d: d["conf"] >= CONF and d["real_score"] >= ACCEPT,
    }
    out = {"split": SPLIT, "conf": CONF, "accept": ACCEPT}
    o3d.utility.random.seed(0)
    planes = {}
    for name, accept in systems.items():
        table, records = run(gt, dets, accept)
        heights = []
        for r in records:
            if r["cls"] != "real" or r["det"] is None:
                continue
            depth = pairs[r["stamp"]].depth_m()
            if name == "baseline":
                xyz = box_position_median(depth, r["det"]["box"], K)
                xyz = None if xyz is None else [round(float(v), 3) for v in xyz]
            else:
                xyz = r["det"]["xyz"]
            r["xyz"] = xyz
            if xyz is None:
                continue
            if r["stamp"] not in planes:
                planes[r["stamp"]] = floor_plane(depth, K)
            heights.append(float(planes[r["stamp"]][:3] @ xyz + planes[r["stamp"]][3]))
        h = np.array(heights)
        table["height_mean"] = round(float(h.mean()), 3) if len(h) else None
        table["height_std"] = round(float(h.std()), 3) if len(h) else None
        out[name] = {"table": table, "records": records}

    json.dump(out, open(f"outputs/results_{SPLIT}.json", "w"), indent=1)
    print(f"{SPLIT}: detector conf >= {CONF}, real_score >= {ACCEPT}")
    print(f"{'system':9} {'real':>4} {'TP':>3} {'FN':>3} {'TPR':>5} | {'decoy':>5} {'FP':>3} {'FPR':>5} | bg | height above floor")
    for name in systems:
        t = out[name]["table"]
        print(f"{name:9} {t['real']:4d} {t['tp']:3d} {t['fn']:3d} {t['tpr']:5.2f} | {t['decoy']:5d} {t['fp']:3d} {t['fpr']:5.2f} | "
              f"{t['accepted_background']:2d} | {t['height_mean']:.2f} ±{t['height_std']:.2f} m")
