import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "Project_1" / "raw" / "eval"
DETECTIONS_FILE = ROOT / "outputs" / "detections_xyz_eval.json"
LABELS_FILE = ROOT / "labels" / "gt_eval.json"
OUTPUT_FILE = ROOT / "outputs" / "features" / "eval_candidate_depth_cues.csv"

DEPTH_SCALE_M = 0.001
MAX_GAP_MS = 50.0

METADATA_COLUMNS = [
    "candidate_id", "stamp", "confidence",
    "box_x1", "box_y1", "box_x2", "box_y2",
    "depth_gap_ms", "reference_class",
]
FEATURE_COLUMNS = [
    "box_valid_fraction", "core_valid_fraction", "core_valid_pixels",
    "core_near_depth_m", "core_iqr_m", "ring_valid_pixels",
    "ring_median_depth_m", "ring_minus_core_m",
    "relative_ring_minus_core",
    "projected_box_width_m", "projected_box_height_m",
]
COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def timestamp(path):
    return int(path.stem.rsplit("_", 1)[1])


def nearest_depth(depth_files, stamp):
    depth_file = min(depth_files, key=lambda path: abs(timestamp(path) - stamp))
    gap_ms = abs(timestamp(depth_file) - stamp) / 1_000_000
    if gap_ms > MAX_GAP_MS:
        raise ValueError(
            f"Frame {stamp}: nearest depth is {gap_ms:.1f} ms away "
            f"(limit {MAX_GAP_MS:.1f} ms)."
        )
    return depth_file, gap_ms


def read_focal_lengths(info_file):
    text = info_file.read_text(encoding="utf-8")
    match = re.search(r"^K:\s*\[([^\]]+)\]", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"No camera matrix K in {info_file}")

    values = [float(value.strip()) for value in match.group(1).split(",")]
    if len(values) != 9 or values[0] <= 0 or values[4] <= 0:
        raise ValueError(f"Invalid camera matrix in {info_file}")

    return values[0], values[4]


def extract_cues(depth_m, box, fx, fy):
    """Measure geometry using only a detector box and its depth frame"""
    cues = {name: None for name in FEATURE_COLUMNS}
    cues["core_valid_pixels"] = 0
    cues["ring_valid_pixels"] = 0

    image_height, image_width = depth_m.shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(image_width - 1, int(np.floor(x1))))
    y1 = max(0, min(image_height - 1, int(np.floor(y1))))
    x2 = max(0, min(image_width, int(np.ceil(x2))))
    y2 = max(0, min(image_height, int(np.ceil(y2))))

    if x2 <= x1 or y2 <= y1:
        return cues

    box_width = x2 - x1
    box_height = y2 - y1
    box_depth = depth_m[y1:y2, x1:x2]
    cues["box_valid_fraction"] = float(np.mean(box_depth > 0))

    inset_x = box_width // 4
    inset_y = box_height // 4
    core = depth_m[
        y1 + inset_y:y2 - inset_y,
        x1 + inset_x:x2 - inset_x,
    ]
    core_values = core[core > 0]
    cues["core_valid_pixels"] = int(core_values.size)
    cues["core_valid_fraction"] = float(np.mean(core > 0))

    if core_values.size == 0:
        return cues

    near, far = np.percentile(core_values, [25, 75])
    near = float(near)
    cues["core_near_depth_m"] = near
    cues["core_iqr_m"] = float(far - near)

    cues["projected_box_width_m"] = float(box_width * near / fx)
    cues["projected_box_height_m"] = float(box_height * near / fy)

    pad_x = max(1, box_width // 2)
    pad_y = max(1, box_height // 2)
    outer_x1 = max(0, x1 - pad_x)
    outer_y1 = max(0, y1 - pad_y)
    outer_x2 = min(image_width, x2 + pad_x)
    outer_y2 = min(image_height, y2 + pad_y)

    ring = depth_m[outer_y1:outer_y2, outer_x1:outer_x2].copy()
    ring[
        y1 - outer_y1:y2 - outer_y1,
        x1 - outer_x1:x2 - outer_x1,
    ] = 0
    ring_values = ring[ring > 0]
    cues["ring_valid_pixels"] = int(ring_values.size)

    if ring_values.size:
        ring_median = float(np.median(ring_values))
        difference = ring_median - near
        cues["ring_median_depth_m"] = ring_median
        cues["ring_minus_core_m"] = difference
        cues["relative_ring_minus_core"] = difference / near

    return cues


def reference_class(detection, references):
    """Attach a development label; never use it to calculate cues"""
    x1, y1, x2, y2 = detection["box"]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    matches = [
        ref for ref in references
        if ref["box"][0] <= center_x <= ref["box"][2]
        and ref["box"][1] <= center_y <= ref["box"][3]
    ]
    if len(matches) == 1:
        return matches[0]["cls"]
    return "ambiguous" if matches else "unmatched"


def make_candidate_rows(detections, depth_files, fx, fy,
                        references_by_stamp=None):
    rows = []
    current_stamp = None
    depth_m = None
    gap_ms = None

    for candidate_id, detection in enumerate(detections):
        stamp = int(detection["stamp"])
        if stamp != current_stamp:
            depth_file, gap_ms = nearest_depth(depth_files, stamp)
            with Image.open(depth_file) as image:
                depth_raw = np.asarray(image).copy()
            if (depth_raw.ndim != 2
                    or not np.issubdtype(depth_raw.dtype, np.integer)):
                raise ValueError(f"Unexpected depth format in {depth_file}")
            depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_M
            current_stamp = stamp

        box = detection["box"]
        row = {
            "candidate_id": candidate_id,
            "stamp": stamp,
            "confidence": detection["conf"],
            "box_x1": box[0], "box_y1": box[1],
            "box_x2": box[2], "box_y2": box[3],
            "depth_gap_ms": gap_ms,
            "reference_class": (
                reference_class(detection, references_by_stamp[stamp])
                if references_by_stamp is not None else ""
            ),
        }
        row.update(extract_cues(depth_m, box, fx, fy))
        rows.append(row)

    return rows


def main():
    detections = json.loads(DETECTIONS_FILE.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    references_by_stamp = defaultdict(list)
    for label in labels:
        references_by_stamp[int(label["stamp"])].append(label)

    depth_files = sorted(
        (EVAL_DIR / "camera_depth_image_raw").glob("*.png"),
        key=timestamp,
    )
    info_files = sorted(
        (EVAL_DIR / "camera_depth_camera_info").glob("*.txt")
    )
    if not depth_files or not info_files:
        raise FileNotFoundError("Depth images or depth camera info missing")

    fx, fy = read_focal_lengths(info_files[0])
    sources = [
        Path(__file__), DETECTIONS_FILE, LABELS_FILE,
        info_files[0], *depth_files,
    ]

    rows = None
    if OUTPUT_FILE.is_file():
        newest_source = max(path.stat().st_mtime for path in sources)
        if OUTPUT_FILE.stat().st_mtime >= newest_source:
            with OUTPUT_FILE.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                cached = list(reader)
                if (
                    reader.fieldnames == COLUMNS
                    and len(cached) == len(detections)
                ):
                    rows = cached

    if rows is None:
        rows = make_candidate_rows(
            detections, depth_files, fx, fy, references_by_stamp
        )

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Development cues: {len(rows)} candidates -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

