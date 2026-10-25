"""Extension result: final system vs final + tracking, same metric and labels as evaluate.py.

Input:  labels/gt_{split}.json, outputs/scores_tracked_{split}.json
Output: outputs/results_tracked_{split}.json
"""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np
import open3d as o3d
from rgbd import pair_frames, load_intrinsics
from evaluate import run, floor_plane, CONF, ACCEPT

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"

if __name__ == "__main__":
    K = load_intrinsics(SPLIT)
    gt = json.load(open(f"labels/gt_{SPLIT}.json"))
    dets = json.load(open(f"outputs/scores_tracked_{SPLIT}.json"))
    pairs = {p.stamp_ns: p for p in pair_frames(SPLIT)}
    o3d.utility.random.seed(0)
    systems = {
        "final": (lambda d: d["conf"] >= CONF and d["real_score"] >= ACCEPT, "xyz"),
        "final+tracking": (lambda d: d["conf"] >= CONF and d["accepted"], "xyz_final"),
    }
    out, planes = {"split": SPLIT}, {}
    for name, (accept, key) in systems.items():
        table, records = run(gt, dets, accept)
        h = []
        for r in records:
            if r["cls"] == "real" and r["det"] and r["det"][key]:
                if r["stamp"] not in planes:
                    planes[r["stamp"]] = floor_plane(pairs[r["stamp"]].depth_m(), K)
                p = planes[r["stamp"]]
                h.append(float(p[:3] @ r["det"][key] + p[3]))
        table["height_mean"], table["height_std"] = round(float(np.mean(h)), 3), round(float(np.std(h)), 3)
        table["rescued_real"] = sum(1 for r in records if r["cls"] == "real" and r["det"] and r["det"].get("rescued"))
        out[name] = {"table": table, "records": records}
    json.dump(out, open(f"outputs/results_tracked_{SPLIT}.json", "w"), indent=1)
    print(f"{SPLIT}: conf >= {CONF}")
    print(f"{'system':15} {'real':>4} {'TP':>3} {'FN':>3} {'TPR':>5} | {'decoy':>5} {'FP':>3} {'FPR':>5} | bg | rescued | height above floor")
    for name in systems:
        t = out[name]["table"]
        print(f"{name:15} {t['real']:4d} {t['tp']:3d} {t['fn']:3d} {t['tpr']:5.2f} | {t['decoy']:5d} {t['fp']:3d} {t['fpr']:5.2f} | "
              f"{t['accepted_background']:2d} | {t['rescued_real']:7d} | {t['height_mean']:.2f} ±{t['height_std']:.2f} m")
