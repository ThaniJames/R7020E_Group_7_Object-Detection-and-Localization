"""Figures for the 3D localisation, from the final evaluation results (test recording).

    fig_3d_method.png   colour crop, depth crop, depth histogram with peak window, front, centre, median
    fig_coords.png      camera coordinate system on a frame + the estimated XYZ
    fig_height.png      distance and height above floor per frame, ours vs baseline
    fig_failure.png     a real extinguisher missed by the team filter: colour/depth time gap while the camera turns
    xyz_table.json      selected detections: XYZ ours vs baseline + height above floor
"""

import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rgbd import pair_frames, load_intrinsics, depth_to_points, object_depths, DATA_ROOT
from extract_depth_cues import extract_cues, read_focal_lengths

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
RADIUS = 0.067
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#1f6fb4", "#e8871e", "#2e9e4f", "#c0392b", "#888888"

K = load_intrinsics(SPLIT)
pairs = {p.stamp_ns: p for p in pair_frames(SPLIT)}
res = json.load(open(f"outputs/results_{SPLIT}.json"))
final, base = res["final"]["records"], res["baseline"]["records"]
t0 = min(pairs)

o3d.utility.random.seed(0)
planes = {}


def floor_plane(stamp):
    if stamp not in planes:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_to_points(pairs[stamp].depth_m(), K)))
        (a, b, c, d), _ = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        planes[stamp] = np.array([a, b, c, d]) * np.sign(d)
    return planes[stamp]


def height(stamp, xyz):
    p = floor_plane(stamp)
    return float(p[:3] @ np.array(xyz) + p[3])


def box_depths(depth, box):
    x1, y1, x2, y2 = (int(round(b)) for b in box)
    patch = depth[max(y1, 0):y2, max(x1, 0):x2]
    return patch[patch > 0]


def histogram_panel(ax, depth, box, title):
    z = box_depths(depth, box)
    counts, edges = np.histogram(z, bins=100)
    peak = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
    obj = object_depths(depth, box)
    front = np.percentile(obj, 5)
    ax.hist(z, bins=100, color="#9db7d5")
    ax.axvspan(peak - 0.05, peak + 0.15, color=GREEN, alpha=0.15, label="kept: object pixels")
    ax.axvline(front, color=BLUE, lw=2, label=f"front (5th pct) {front:.2f} m")
    ax.axvline(front + RADIUS, color=GREEN, lw=2, ls="--", label=f"centre = front + r {front + RADIUS:.2f} m")
    ax.axvline(np.median(z), color=RED, lw=2, label=f"baseline median {np.median(z):.2f} m")
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("pixels in box")
    ax.set_title(title)
    ax.legend(fontsize=9, frameon=False)


def crops(stamp, box, pad=25):
    p = pairs[stamp]
    img, depth = p.color(), p.depth_m()
    x1, y1, x2, y2 = (int(round(b)) for b in box)
    X1, Y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    X2, Y2 = min(x2 + pad, img.shape[1]), min(y2 + pad, img.shape[0])
    c = img[Y1:Y2, X1:X2].copy()
    cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (255, 255, 255), 2)
    d = depth[Y1:Y2, X1:X2]
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB), d, depth


# frames
reals = [(f, b) for f, b in zip(final, base) if f["cls"] == "real" and f["det"] and f.get("xyz") and b.get("xyz")]
# method example: the clearest case of background in the box (largest baseline - ours depth gap)
f_ex, b_ex = max(reals, key=lambda fb: (fb[1]["xyz"][2] - fb[0]["xyz"][2]) * (fb[0]["det"]["conf"] > 0.5))
box_ex = f_ex["det"]["box"]

# method figure
c, d, depth = crops(f_ex["stamp"], box_ex)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 2.4]})
ax[0].imshow(c); ax[0].set_title("1. YOLO box (colour)"); ax[0].axis("off")
dm = np.ma.masked_equal(d, 0)
im = ax[1].imshow(dm, cmap="turbo", vmin=float(np.percentile(d[d > 0], 2)), vmax=float(np.percentile(d[d > 0], 98)))
ax[1].set_title("2. depth in the box"); ax[1].axis("off")
fig.colorbar(im, ax=ax[1], fraction=0.046, label="m")
histogram_panel(ax[2], depth, box_ex, "3. tallest bar = object; wall/floor thrown away")
fig.tight_layout(); fig.savefig(OUT / "fig_3d_method.png", dpi=170); plt.close(fig)

# coordinate system
img = cv2.cvtColor(pairs[f_ex["stamp"]].color(), cv2.COLOR_BGR2RGB)
x1, y1, x2, y2 = box_ex
u, v = (x1 + x2) / 2, (y1 + y2) / 2
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.imshow(img); ax.axis("off")
ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=2))
ax.annotate("", xy=(K.cx + 110, K.cy), xytext=(K.cx, K.cy), arrowprops=dict(color=RED, width=3, headwidth=10))
ax.annotate("", xy=(K.cx, K.cy + 110), xytext=(K.cx, K.cy), arrowprops=dict(color=GREEN, width=3, headwidth=10))
ax.plot(K.cx, K.cy, "o", ms=13, mfc="none", mec=BLUE, mew=3); ax.plot(K.cx, K.cy, ".", ms=8, color=BLUE)
ax.text(K.cx + 115, K.cy + 5, "X right", color="white", fontsize=12, weight="bold", bbox=dict(fc=RED, ec="none", pad=2))
ax.text(K.cx + 5, K.cy + 128, "Y down", color="white", fontsize=12, weight="bold", bbox=dict(fc=GREEN, ec="none", pad=2))
ax.text(K.cx - 170, K.cy + 45, "Z forward\n(into image)", color="white", fontsize=11, weight="bold", bbox=dict(fc=BLUE, ec="none", pad=2))
ax.plot(u, v, "+", ms=16, mew=3, color="yellow")
X, Y, Z = f_ex["xyz"]
ax.text(x1 - 8, y1 - 8, f"X = {X:+.2f} m\nY = {Y:+.2f} m\nZ = {Z:.2f} m", color="black", fontsize=12, ha="right", va="bottom",
        bbox=dict(fc="white", ec="black", alpha=0.9))
ax.set_title("Colour camera optical frame, origin at the camera, metres")
fig.tight_layout(); fig.savefig(OUT / "fig_coords.png", dpi=170); plt.close(fig)

# stability over the recording
rows = []
for f, b in reals:
    rows.append({"t": (f["stamp"] - t0) / 1e9, "stamp": f["stamp"], "z_ours": f["xyz"][2], "z_base": b["xyz"][2],
                 "h_ours": height(f["stamp"], f["xyz"]), "h_base": height(b["stamp"], b["xyz"]),
                 "xyz_ours": f["xyz"], "xyz_base": b["xyz"], "conf": f["det"]["conf"]})
t = np.array([r["t"] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
# baseline: every real object it localised, same set as evaluate.py
base_all = [r for r in base if r["cls"] == "real" and r.get("xyz")]
tb = np.array([(r["stamp"] - t0) / 1e9 for r in base_all])
ax[0].plot(tb, [r["xyz"][2] for r in base_all], "o-", ms=3, color=RED, label="baseline (median of box)")
ax[0].plot(t, [r["z_ours"] for r in rows], "o-", ms=3, color=BLUE, label="ours (object centre)")
ax[0].set_xlabel(f"time in {SPLIT} recording (s)"); ax[0].set_ylabel("Z distance (m)")
ax[0].set_title("Distance to the real extinguisher"); ax[0].legend(frameon=False)
hb, ho = np.array([height(r["stamp"], r["xyz"]) for r in base_all]), np.array([r["h_ours"] for r in rows])
ax[1].plot(tb, hb, "o-", ms=3, color=RED, label=f"baseline {hb.mean():.2f} ± {hb.std():.2f} m")
ax[1].plot(t, ho, "o-", ms=3, color=BLUE, label=f"ours {ho.mean():.2f} ± {ho.std():.2f} m")
ax[1].set_xlabel(f"time in {SPLIT} recording (s)"); ax[1].set_ylabel("height above floor (m)")
ax[1].set_title("Height above floor (object does not move, so it should be flat)")
ax[1].set_ylim(0, max(0.9, hb.max() + 0.05)); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_height.png", dpi=170); plt.close(fig)

# table of selected detections
idx = np.linspace(0, len(rows) - 1, 5).round().astype(int)
table = [{"t_s": round(rows[i]["t"], 1), "conf": rows[i]["conf"],
          "ours_xyz": rows[i]["xyz_ours"], "ours_height": round(rows[i]["h_ours"], 2),
          "base_xyz": rows[i]["xyz_base"], "base_height": round(rows[i]["h_base"], 2)} for i in idx]
json.dump({"n_frames_ours": len(rows), "n_frames_base": len(base_all), "ours_height_mean": round(float(ho.mean()), 3), "ours_height_std": round(float(ho.std()), 3),
           "base_height_mean": round(float(hb.mean()), 3), "base_height_std": round(float(hb.std()), 3),
           "z_range_ours": [round(float(min(r["z_ours"] for r in rows)), 2), round(float(max(r["z_ours"] for r in rows)), 2)],
           "selected": table}, open(OUT / "xyz_table.json", "w"), indent=1)

# failure case (team filter): depth frame older than the colour frame while the camera turns,
#      so in the depth image the object has moved out of the box and the box sees floor
dets = json.load(open(f"outputs/scored_detections_xyz_{SPLIT}.json"))
FX, FY = read_focal_lengths(sorted((DATA_ROOT / SPLIT / "camera_depth_camera_info").glob("*.txt"))[0])
missed = [f for f in final if f["cls"] == "real" and not f["det"]]
cands = []
for m in missed:
    for d in dets:
        if d["stamp"] == m["stamp"] and m["box"][0] <= (d["box"][0] + d["box"][2]) / 2 <= m["box"][2]:
            contrast = extract_cues(pairs[d["stamp"]].depth_m(), d["box"], FX, FY)["relative_ring_minus_core"]
            if contrast is not None:
                cands.append((m, d, contrast))
if cands:
    m, dd, contrast = min(cands, key=lambda c: c[2])
    p = pairs[dd["stamp"]]
    img, depth = cv2.cvtColor(p.color(), cv2.COLOR_BGR2RGB), p.depth_m()
    H, W = depth.shape
    x1, y1, x2, y2 = (int(round(b)) for b in dd["box"])
    w, h = x2 - x1, y2 - y1
    core = depth[y1 + h // 4:y2 - h // 4, x1 + w // 4:x2 - w // 4]
    near = np.percentile(core[core > 0], 25)
    R = (max(x1 - w // 2, 0), max(y1 - h // 2, 0), min(x2 + w // 2, W), min(y2 + h // 2, H))
    ring = depth[R[1]:R[3], R[0]:R[2]].copy()
    ring[y1 - R[1]:y2 - R[1], x1 - R[0]:x2 - R[0]] = 0
    ring = ring[ring > 0]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for a, im, title in [(ax[0], img, "missed real (camera turning)"), (ax[1], None, f"depth is {p.gap_ms:.0f} ms older: object moved right")]:
        if im is None:
            a.imshow(np.ma.masked_equal(depth, 0), cmap="turbo", vmin=0, vmax=6)
        else:
            a.imshow(im)
        a.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, ec=ORANGE, lw=2.5))
        a.add_patch(plt.Rectangle((R[0], R[1]), R[2] - R[0], R[3] - R[1], fill=False, ec="white", lw=1.5, ls="--"))
        a.set_xlim(max(R[0] - 40, 0), min(R[2] + 40, W)); a.set_ylim(min(R[3] + 20, H), max(R[1] - 20, 0))
        a.set_title(title); a.axis("off")
    ax[2].hist(ring, bins=80, color="#9db7d5", label="ring around the box (mostly floor)")
    ax[2].axvline(near, color=BLUE, lw=2, label=f"inside the box {near:.2f} m (floor, not object)")
    ax[2].axvline(np.median(ring), color=RED, lw=2, label=f"ring median {np.median(ring):.2f} m")
    ax[2].set_xlabel("depth (m)"); ax[2].set_ylabel("pixels")
    ax[2].set_title(f"no free space found: contrast {contrast:.2f} < 0.19 → rejected")
    ax[2].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_failure.png", dpi=170); plt.close(fig)

print("saved to", OUT, "| example frame t =", round((f_ex["stamp"] - t0) / 1e9, 1), "s | failure figure:", bool(cands))
print(json.dumps(json.load(open(OUT / "xyz_table.json")), indent=1)[:1500])
