"""Extract test cues using the same geometry as development"""

import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_depth_cues import (
    COLUMNS, make_candidate_rows, read_focal_lengths, timestamp,
)


ROOT = SCRIPT_DIR.parent
TEST_DIR = ROOT / "data" / "Project_1" / "raw" / "test"
DETECTIONS_FILE = ROOT / "outputs" / "detections_xyz_test.json"
OUTPUT_FILE = ROOT / "outputs" / "features" / "test_candidate_depth_cues.csv"


def valid_cached_table(detections):
    if not OUTPUT_FILE.is_file():
        return False
    with OUTPUT_FILE.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if reader.fieldnames != COLUMNS or len(rows) != len(detections):
            return False
    for i, (row, detection) in enumerate(zip(rows, detections)):
        if (int(row["candidate_id"]) != i
                or int(row["stamp"]) != int(detection["stamp"])
                or row["reference_class"] != ""):
            return False
        for j, key in enumerate(("x1", "y1", "x2", "y2")):
            if abs(float(row[f"box_{key}"]) - float(detection["box"][j])) > 1e-5:
                return False
    return True


def main():
    if not DETECTIONS_FILE.is_file():
        raise FileNotFoundError(DETECTIONS_FILE)
    detections = json.loads(DETECTIONS_FILE.read_text(encoding="utf-8"))
    depth_files = sorted(
        (TEST_DIR / "camera_depth_image_raw").glob("*.png"),
        key=timestamp,
    )
    info_files = sorted(
        (TEST_DIR / "camera_depth_camera_info").glob("*.txt")
    )
    if not depth_files or not info_files:
        raise FileNotFoundError("Test depth images or camera info missing")
    fx, fy = read_focal_lengths(info_files[0])

    sources = [
        Path(__file__), SCRIPT_DIR / "extract_depth_cues.py",
        DETECTIONS_FILE, info_files[0], *depth_files,
    ]
    latest_source = max(path.stat().st_mtime for path in sources)
    if (OUTPUT_FILE.is_file()
            and OUTPUT_FILE.stat().st_mtime >= latest_source
            and valid_cached_table(detections)):
        print(f"Test cues: {len(detections)} candidates (cached) -> {OUTPUT_FILE}")
        return

    rows = make_candidate_rows(detections, depth_files, fx, fy)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    missing = sum(
        row["projected_box_height_m"] is None
        or row["relative_ring_minus_core"] is None
        for row in rows
    )
    print(f"Test cues: {len(rows)} candidates, {missing} missing cues -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
