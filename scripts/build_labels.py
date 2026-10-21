"""Build reference labels (ground truth) from the reviewed box proposals."""
import json
from pathlib import Path

LABELS = Path("labels")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def centre_inside(a, b):
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def merge_same(boxes):
    """Merge boxes that describe the same object."""
    merged = []
    for b in boxes:
        for i, m in enumerate(merged):
            if iou(b, m) > 0.3 or centre_inside(b, m) or centre_inside(m, b):
                merged[i] = union(m, b)
                break
        else:
            merged.append(list(b))
    return merged


def build(split):
    proposals = json.load(open(LABELS / f"proposals_{split}.json"))
    decisions = json.load(open(LABELS / "review_decisions.json"))["decisions"][split]
    cls_of = {i: c for c, ids in decisions.items() for i in ids}
    manual_file = LABELS / "manual_boxes.json"
    manual = json.load(open(manual_file)).get(split, []) if manual_file.exists() else []

    frames = {}
    for i, p in enumerate(proposals):
        c = cls_of.get(i)
        if c is None:
            continue                                         # junk
        c = "real" if c == "real_part" else c
        frames.setdefault(p["stamp"], {"real": [], "decoy": []})[c].append(p["box"])
    for m in manual:
        frames.setdefault(m["stamp"], {"real": [], "decoy": []})[m["cls"]].append(m["box"])

    gt = []
    for stamp, objs in sorted(frames.items()):
        if objs["real"]:
            box = objs["real"][0]
            for b in objs["real"][1:]:
                box = union(box, b)
            gt.append({"stamp": stamp, "cls": "real", "box": [round(v, 1) for v in box]})
        for box in merge_same(objs["decoy"]):
            gt.append({"stamp": stamp, "cls": "decoy", "box": [round(v, 1) for v in box]})
    json.dump(gt, open(LABELS / f"gt_{split}.json", "w"), indent=1)
    return gt


if __name__ == "__main__":
    for split in ["test", "eval"]:
        gt = build(split)
        n_real = sum(g["cls"] == "real" for g in gt)
        n_decoy = sum(g["cls"] == "decoy" for g in gt)
        print(f"{split}: {n_real} real, {n_decoy} decoy objects")
