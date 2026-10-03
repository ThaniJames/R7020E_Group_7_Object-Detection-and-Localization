import json
import numpy as np

SPLIT = "test"
MAX_DIST = 120
MAX_GAP = 2

dets = json.load(open(f"outputs/detections_{SPLIT}.json"))
stamps = sorted({d["stamp"] for d in dets})

def centre(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

tracks = []
for frame_no, stamp in enumerate(stamps):
    used = set()
    for d in sorted([d for d in dets if d["stamp"] == stamp], key=lambda d: -d["conf"]):
        c = centre(d["box"])
        best, best_dist = None, MAX_DIST
        for t in tracks:
            if t["id"] in used or frame_no - t["last_frame"] > MAX_GAP:
                continue
            if abs(d["xyz"][2] - t["last_z"]) > 0.5:
                continue
            dist = np.linalg.norm(c - t["last_centre"])
            if dist < best_dist:
                best, best_dist = t, dist
        if best is None:
            best = {"id": len(tracks), "dets": []}
            tracks.append(best)
        best["last_centre"], best["last_frame"] = c, frame_no
        best["last_z"] = d["xyz"][2]
        best["dets"].append(d)
        used.add(best["id"])
        d["track"] = best["id"]

for t in tracks:
    if len(t["dets"]) < 3:
        continue
    z = [d["xyz"][2] for d in t["dets"]]
    conf = np.mean([d["conf"] for d in t["dets"]])
    print(f"track {t['id']:2d}: {len(t['dets']):2d} frames, mean conf {conf:.2f}, "
          f"distance {min(z):.2f}-{max(z):.2f} m")

json.dump(dets, open(f"outputs/tracks_{SPLIT}.json", "w"), indent=1)
