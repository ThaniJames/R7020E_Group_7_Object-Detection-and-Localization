"""Compare the frozen real/decoy filter with accept-all on test objects"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import PIL

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_filter import associate, evaluate


ROOT = SCRIPT_DIR.parent
DETECTIONS = ROOT / "outputs" / "detections_xyz_test.json"
SCORED = ROOT / "outputs" / "scored_detections_xyz_test.json"
ACCEPTED = ROOT / "outputs" / "detections_real_xyz_test.json"
FEATURES = ROOT / "outputs" / "features" / "test_candidate_depth_cues.csv"
LABELS = ROOT / "labels" / "gt_test.json"
MODEL = ROOT / "outputs" / "models" / "real_decoy_filter.json"
REPORT = ROOT / "outputs" / "filter_validation" / "test_filter_results.json"


def read_json_list(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def print_counts(name, counts):
    print(
        f"{name}: real {counts['real_accepted']}/{counts['real_total']} "
        f"(TPR {counts['tpr']:.3f}); decoys "
        f"{counts['decoys_accepted']}/{counts['decoy_total']} "
        f"(FPR {counts['fpr']:.3f})"
    )


def main():
    sources = (DETECTIONS, SCORED, ACCEPTED, FEATURES, LABELS, MODEL)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    detections = read_json_list(DETECTIONS)
    scored = read_json_list(SCORED)
    accepted = read_json_list(ACCEPTED)
    labels = read_json_list(LABELS)
    model = json.loads(MODEL.read_text(encoding="utf-8"))

    if len(detections) != len(scored):
        raise ValueError("Scored and original candidate counts differ.")
    selected = []
    for i, (original, result) in enumerate(zip(detections, scored)):
        if result.get("candidate_id") != i:
            raise ValueError(f"Candidate ID differs at row {i}")
        if any(result.get(key) != original.get(key)
               for key in ("stamp", "box", "conf", "xyz")):
            raise ValueError(f"Original detection changed at row {i}")
        if result.get("real_score") not in (0.0, 1.0):
            raise ValueError(f"Invalid binary real_score at row {i}")
        if result["real_score"] == 1.0:
            selected.append(i)
    if accepted != [scored[i] for i in selected]:
        raise ValueError("Accepted JSON is not the scored JSON's accepted subset")

    associations, ambiguous = associate(detections, labels)
    baseline = evaluate(labels, associations, range(len(detections)))
    final = evaluate(labels, associations, selected)
    selected_refs = {associations[i] for i in selected if i in associations}
    missed_real = [
        {"reference_id": i, "stamp": int(label["stamp"])}
        for i, label in enumerate(labels)
        if label["cls"] == "real" and i not in selected_refs
    ]
    accepted_decoys = [
        {"reference_id": i, "stamp": int(label["stamp"])}
        for i, label in enumerate(labels)
        if label["cls"] == "decoy" and i in selected_refs
    ]
    unmatched_selected = [i for i in selected if i not in associations]

    report = {
        "split": "test",
        "metric_definition": (
            "Associate when predicted box centre lies inside reference box; "
            "resolve overlapping references by IoU; count each reference "
            "at most once. TPR = accepted real / all real; "
            "decoy FPR = accepted decoys / all decoys."
        ),
        "recording_frames_with_references": len({x["stamp"] for x in labels}),
        "detector_candidates": len(detections),
        "accepted_candidates": len(selected),
        "unmatched_accepted_candidate_ids": unmatched_selected,
        "candidates_in_overlapping_reference_boxes": ambiguous,
        "baseline_accept_all": baseline,
        "frozen_filter": final,
        "missed_real_references": missed_real,
        "accepted_decoy_references": accepted_decoys,
        "height_threshold_m": model["height_threshold_m"],
        "relative_depth_threshold": model["relative_depth_threshold"],
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_counts("Baseline", baseline)
    print_counts("Filter  ", final)
    print(f"Unmatched accepted boxes: {len(unmatched_selected)} -> {REPORT}")


if __name__ == "__main__":
    main()
