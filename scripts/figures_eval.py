"""Figures for the evaluation slides (final evaluation = test recording by default)"""
import sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
res = json.load(open(f"outputs/results_{SPLIT}.json"))
b, f = res["baseline"]["table"], res["final"]["table"]
tr = json.load(open(f"outputs/results_tracked_{SPLIT}.json"))["final+tracking"]["table"]
RED, BLUE, LIGHT, GREY = "#c0392b", "#1f6fb4", "#8fb8de", "#888888"

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(9, 5.6))
for a, key, title, target, good in [(ax[0], "tpr", "Real-object TPR  (target ≥ 0.75 - - -)", 0.75, "≥"),
                                    (ax[1], "fpr", "Decoy FPR  (target ≤ 0.15 - - -)", 0.15, "≤")]:
    vals = [b[key], f[key], tr[key]]
    bars = a.bar(["baseline", "final", "+ tracking\n(extension)"], vals, color=[RED, BLUE, LIGHT], width=0.6,
                 hatch=["", "", "//"], edgecolor=["none", "none", BLUE])
    a.axhline(target, ls="--", color=GREY, lw=1.5)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=15, weight="bold")
    a.set_ylim(0, 1.12)
    a.set_title(title)
fig.tight_layout()
fig.savefig(OUT / "fig_results.png", dpi=170)
print("saved", OUT / "fig_results.png", b, f)

# visual evidence
sys.path.insert(0, "src")
import cv2
from rgbd import pair_frames

pairs = {p.stamp_ns: p for p in pair_frames(SPLIT)}
dets = json.load(open(f"outputs/scored_detections_xyz_{SPLIT}.json"))
final, base = res["final"]["records"], res["baseline"]["records"]
CONF, ACCEPT = res["conf"], res["accept"]


def label(img, text, x, y, colour):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    y = max(y, h + 8)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 8, y + 4), colour, -1)
    cv2.putText(img, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def zoom(img, boxes, aspect=0.85, margin=70):
    """Crop around the drawn boxes (2x coords) so the labels stay readable on a slide."""
    H, W = img.shape[:2]
    x1 = max(min(b[0] for b in boxes) - margin, 0); x2 = min(max(b[2] for b in boxes) + margin, W)
    y1 = max(min(b[1] for b in boxes) - margin - 40, 0); y2 = min(max(b[3] for b in boxes) + margin, H)
    w, h = x2 - x1, y2 - y1
    if w / h < aspect:                                  # too tall: widen
        extra = int(h * aspect) - w; x1 = max(x1 - extra // 2, 0); x2 = min(x1 + int(h * aspect), W)
    else:                                               # too wide: heighten
        extra = int(w / aspect) - h; y1 = max(y1 - extra // 2, 0); y2 = min(y1 + int(w / aspect), H)
    return img[y1:y2, x1:x2]


def tile(stamp, name, missed_box=None, accepted_only=False):
    img = cv2.resize(pairs[stamp].color(), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    drawn = []
    acc_d = [d for d in dets if d["stamp"] == stamp and d["conf"] >= CONF and d["real_score"] >= ACCEPT]
    acc = [d["box"] for d in acc_d]
    for d in sorted(dets, key=lambda d: d["real_score"]):
        if d["stamp"] != stamp or d["conf"] < CONF:
            continue
        c = ((d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2)
        if d["real_score"] < ACCEPT and any(a[0] <= c[0] <= a[2] and a[1] <= c[1] <= a[3] for a in acc):
            continue                    # extra box on the accepted real object (e.g. body only): not shown
        if d["real_score"] >= ACCEPT and any(a["conf"] > d["conf"] and a["box"][0] <= c[0] <= a["box"][2]
                                             and a["box"][1] <= c[1] <= a["box"][3] for a in acc_d):
            continue                    # second accepted box on the same object: only the most confident is shown
        x1, y1, x2, y2 = (int(2 * v) for v in d["box"])
        if missed_box is None and (not accepted_only or d["real_score"] >= ACCEPT):
            drawn.append((x1, y1, x2, y2))
        if d["real_score"] >= ACCEPT:
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 160, 40), 4)
            label(img, "REAL ({:.2f}, {:.2f}, {:.2f}) m".format(*d["xyz"]), x1, y1 - 4, (60, 160, 40))
        elif missed_box is None:
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 200), 3)
            label(img, "DECOY", x1, y2 + 30, (40, 40, 200))   # below the box, never over a REAL label
    if missed_box is not None:
        x1, y1, x2, y2 = (int(2 * v) for v in missed_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 4)
        label(img, "REAL, rejected", x1, y1 - 4, (0, 140, 255))
        drawn.append((x1, y1, x2, y2))
    cv2.imwrite(str(OUT / name), zoom(img, drawn))


# real accepted: the largest accepted real box with a confident detection, fully inside the image
acc = [r for r in final if r["cls"] == "real" and r["det"] and r["det"]["conf"] > 0.7
       and r["det"]["box"][0] > 5 and r["det"]["box"][2] < 635 and r["det"]["box"][3] < 395]
best = max(acc, key=lambda r: r["det"]["box"][3] - r["det"]["box"][1])
tile(best["stamp"], "ev_real_accepted.png", accepted_only=True)
# decoys rejected: the frame where the filter removed the most decoys the baseline accepted
count = {}
for rb, rf in zip(base, final):
    if rf["cls"] == "decoy" and rb["det"] and not rf["det"]:
        count[rf["stamp"]] = count.get(rf["stamp"], 0) + 1
rej = max(count, key=lambda s: (count[s], -sum(r["stamp"] == s and r["cls"] == "real" for r in final)))
tile(rej, "ev_decoys_rejected.png")
# real missed by the filter (first one in time)
miss = sorted([r for r in final if r["cls"] == "real" and not r["det"]], key=lambda r: r["stamp"])
tile(miss[0]["stamp"], "ev_real_missed.png", missed_box=miss[0]["box"])

# extension: the same missed frame, kept as real by tracking
tracked = json.load(open(f"outputs/scores_tracked_{SPLIT}.json"))
img = cv2.resize(pairs[miss[0]["stamp"]].color(), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
drawn = []
for d in tracked:
    if d["stamp"] == miss[0]["stamp"] and d.get("rescued"):
        x1, y1, x2, y2 = (int(2 * v) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 4)
        label(img, "filter: rejected", x1, y1 - 76, (40, 40, 200))
        label(img, "tracking: kept as REAL", x1, y1 - 40, (0, 140, 255))
        label(img, "({:.2f}, {:.2f}, {:.2f}) m".format(*d["xyz_final"]), x1, y1 - 4, (0, 140, 255))
        drawn.append((x1, y1 - 76, x1 + 330, y2))
cv2.imwrite(str(OUT / "ev_real_rescued.png"), zoom(img, drawn, margin=50))
print("evidence tiles: rejected decoys in chosen frame =", count[rej])
