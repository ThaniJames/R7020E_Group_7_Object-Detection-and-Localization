"""Apply the saved real/decoy rule to detector candidates"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "outputs" / "models" / "real_decoy_filter.json"


def apply_to_split(split="eval"):
    if split not in {"eval", "test"}:
        raise ValueError("split must be 'eval' or 'test'")

    detections_file = ROOT / "outputs" / f"detections_xyz_{split}.json"
    features_file = ROOT / "outputs" / "features" / f"{split}_candidate_depth_cues.csv"
    scored_file = ROOT / "outputs" / f"scored_detections_xyz_{split}.json"
    accepted_file = ROOT / "outputs" / f"detections_real_xyz_{split}.json"

    for path in (MODEL_FILE, detections_file, features_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    height_cut = float(model["height_threshold_m"])
    contrast_cut = float(model["relative_depth_threshold"])
    if not math.isfinite(height_cut) or not math.isfinite(contrast_cut):
        raise ValueError("Saved filter thresholds must be finite.")

    if split == "eval":
        for path in (detections_file, features_file):
            expected = model["source_sha256"][path.name]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"{path.name} changed since calibration. "
                    "Regenerate the cues and refit before applying."
                )

    detections = json.loads(detections_file.read_text(encoding="utf-8"))
    with features_file.open(newline="", encoding="utf-8") as file:
        features = list(csv.DictReader(file))
    if len(features) != len(detections):
        raise ValueError("Feature and detection row counts differ.")

    scored = []
    accepted = []
    invalid_features = 0
    for candidate_id, (detection, feature) in enumerate(zip(detections, features)):
        if int(feature["candidate_id"]) != candidate_id:
            raise ValueError(f"Candidate order differs at row {candidate_id}.")
        if int(feature["stamp"]) != int(detection["stamp"]):
            raise ValueError(f"Timestamp differs at row {candidate_id}.")
        for position, key in enumerate(("x1", "y1", "x2", "y2")):
            if abs(float(feature[f"box_{key}"]) - float(detection["box"][position])) > 1e-5:
                raise ValueError(f"Bounding box differs at row {candidate_id}.")

        raw_height = feature["projected_box_height_m"]
        raw_contrast = feature["relative_ring_minus_core"]
        height = float(raw_height) if raw_height else float("nan")
        contrast = float(raw_contrast) if raw_contrast else float("nan")
        valid = math.isfinite(height) and math.isfinite(contrast)
        invalid_features += not valid
        is_real = valid and height >= height_cut and contrast >= contrast_cut

        result = {**detection, "candidate_id": candidate_id,
                  "real_score": float(is_real)}
        scored.append(result)
        if is_real:
            accepted.append(result)

    scored_file.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    accepted_file.write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    print(
        f"{split}: {len(accepted)}/{len(scored)} accepted, "
        f"{invalid_features} invalid cue pairs -> {accepted_file}"
    )
    return scored, accepted


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: apply_real_decoy_filter.py [eval|test]")
    apply_to_split(sys.argv[1] if len(sys.argv) == 2 else "eval")
