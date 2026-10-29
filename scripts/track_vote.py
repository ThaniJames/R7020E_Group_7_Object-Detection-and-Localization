"""Extension: temporal tracking that keeps a confirmed real object "real" through short filter failures.
Designed on the eval recording (development): the filter misses there were 1-2 frame gaps inside runs of
accepted frames. MAX_DIST and MAX_GAP are chosen on eval only"""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np
from rgbd import pair_frames
from evaluate import ACCEPT

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
CONF = 0.10
MAX_DIST = 80 # px, chosen on eval by scripts/select_tracking.py
MAX_GAP = 2 # frames, chosen on eval by scripts/select_tracking.py
MAX_DZ = 0.5 # m, same as track.py
MIN_HITS = 3 # accepted frames before a track counts as a confirmed real object
SIZE_RATIO = 0.5 # rescued box height >= half of the track's box height


def centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def height(b):
    return b[3] - b[1]


def track(dets, stamps, max_dist=MAX_DIST, max_gap=MAX_GAP):
    """Adds accepted / rescued / xyz_final to every detection in dets (in place)."""
    tracks = []
    for d in dets:
        d["accepted"], d["rescued"], d["xyz_final"] = False, False, None

    for f, stamp in enumerate(stamps):
        frame = [d for d in dets if d["stamp"] == stamp and d["conf"] >= CONF]
        alive = [t for t in tracks if f - t["last"] <= max_gap]
        used = set()
        # 1. boxes the depth filter accepts: accept, and update or start a track
        for d in sorted([d for d in frame if d["real_score"] >= ACCEPT], key=lambda d: -d["conf"]):
            d["accepted"], d["xyz_final"] = True, d["xyz"]
            near = [t for t in alive if id(t) not in used and np.linalg.norm(centre(d["box"]) - centre(t["box"])) <= max_dist]
            if near:
                t = min(near, key=lambda t: np.linalg.norm(centre(d["box"]) - centre(t["box"])))
                t.update(box=d["box"], xyz=d["xyz"], hits=t["hits"] + 1, last=f, rescues=0)
            else:
                t = {"box": d["box"], "xyz": d["xyz"], "hits": 1, "last": f, "rescues": 0}
                tracks.append(t)
            used.add(id(t))
        # 2. confirmed tracks with no accepted box this frame: rescue the best matching rejected box
        for t in alive:
            if id(t) in used or t["hits"] < MIN_HITS or t["rescues"] >= max_gap:
                continue
            cands = [d for d in frame if not d["accepted"]
                     and np.linalg.norm(centre(d["box"]) - centre(t["box"])) <= max_dist
                     and height(d["box"]) >= SIZE_RATIO * height(t["box"])]
            if not cands:
                continue
            d = min(cands, key=lambda d: np.linalg.norm(centre(d["box"]) - centre(t["box"])))
            same_depth = d["xyz"] is not None and t["xyz"] is not None and abs(d["xyz"][2] - t["xyz"][2]) < MAX_DZ
            d["accepted"], d["rescued"] = True, True
            d["xyz_final"] = d["xyz"] if same_depth else t["xyz"]
            t.update(box=d["box"], last=f, rescues=t["rescues"] + 1)
            used.add(id(t))
    return dets


if __name__ == "__main__":
    dets = json.load(open(f"outputs/scored_detections_xyz_{SPLIT}.json"))
    stamps = [p.stamp_ns for p in pair_frames(SPLIT)]
    track(dets, stamps)
    json.dump(dets, open(f"outputs/scores_tracked_{SPLIT}.json", "w"), indent=1)
    n = sum(d["rescued"] for d in dets)
    print(f"{SPLIT}: {sum(d['accepted'] for d in dets)} boxes accepted, {n} of them rescued by tracking "
          f"-> outputs/scores_tracked_{SPLIT}.json")
