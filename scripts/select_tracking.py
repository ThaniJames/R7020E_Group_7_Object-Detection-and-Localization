"""Choose the tracking settings on the eval recording only (development). Rule (written before running it): see README, section 5.

Input:  outputs/scored_detections_xyz_eval.json (chosen model + Task 2 filter), labels/gt_eval.json
Output: outputs/tracking_selection.json
"""
import sys, json, copy
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from rgbd import pair_frames
from evaluate import run, CONF, ACCEPT
from track_vote import track

SPLIT = "eval"
gt = json.load(open(f"labels/gt_{SPLIT}.json"))
base = json.load(open(f"outputs/scored_detections_xyz_{SPLIT}.json"))
stamps = [p.stamp_ns for p in pair_frames(SPLIT)]
filt, _ = run(gt, base, lambda d: d["conf"] >= CONF and d["real_score"] >= ACCEPT)

rows = []
for max_dist in (80, 120, 160):
    for max_gap in (1, 2, 3):
        dets = track(copy.deepcopy(base), stamps, max_dist, max_gap)
        t, _ = run(gt, dets, lambda d: d["conf"] >= CONF and d["accepted"])
        rows.append({"max_dist": max_dist, "max_gap": max_gap, "tp": t["tp"], "fp": t["fp"],
                     "background": t["accepted_background"],
                     "rescued_real": t["tp"] - filt["tp"], "extra_decoys": t["fp"] - filt["fp"],
                     "extra_background": t["accepted_background"] - filt["accepted_background"]})
        print(rows[-1])

# rule: 0 extra decoys; most rescued real; fewest extra background; smallest settings
ok = [r for r in rows if r["extra_decoys"] == 0]
best = min(ok, key=lambda r: (-r["rescued_real"], r["extra_background"], r["max_dist"], r["max_gap"]))
json.dump({"filter_only": filt, "grid": rows, "choice": best}, open("outputs/tracking_selection.json", "w"), indent=1)
print("chosen:", best)
