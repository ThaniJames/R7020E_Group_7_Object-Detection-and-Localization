"""Check development detections, labels, and depth-image timing"""

import json
from pathlib import Path

from extract_depth_cues import nearest_depth, timestamp


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "Project_1" / "raw" / "eval"
DETECTIONS = ROOT / "outputs" / "detections_xyz_eval.json"
LABELS = ROOT / "labels" / "gt_eval.json"


def load_list(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def main():
    detections = load_list(DETECTIONS)
    labels = load_list(LABELS)
    depth_files = list((EVAL_DIR / "camera_depth_image_raw").glob("*.png"))
    colour_files = list((EVAL_DIR / "camera_color_image_raw").glob("*.png"))
    if not depth_files or not colour_files:
        raise FileNotFoundError("Development colour or depth images missing.")

    stamps = {int(item["stamp"]) for item in detections}
    colour_stamps = {timestamp(path) for path in colour_files}
    missing = stamps - colour_stamps
    if missing:
        raise ValueError(f"{len(missing)} detector frames lack a colour image")

    gaps = [nearest_depth(depth_files, stamp)[1] for stamp in stamps]
    labeled_stamps = {int(item["stamp"]) for item in labels}
    print(
        f"Development inputs: {len(detections)} candidates, "
        f"{len(labels)} references, {len(stamps)} detector frames; "
        f"max depth gap {max(gaps, default=0):.1f} ms"
    )
    print(f"Labeled frames without candidates: {len(labeled_stamps - stamps)}")


if __name__ == "__main__":
    main()
