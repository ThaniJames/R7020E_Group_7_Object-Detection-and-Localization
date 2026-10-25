import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DETECTIONS_FILE = ROOT / "outputs" / "detections_xyz_eval.json"
LABELS_FILE = ROOT / "labels" / "gt_eval.json"
FEATURES_FILE = ROOT / "outputs" / "features" / "eval_candidate_depth_cues.csv"
FOLDS_FILE = ROOT / "outputs" / "manifests" / "eval_temporal_folds.csv"

FOLD_COLUMNS = [
    "stamp", "fold", "real_references",
    "decoy_references", "detector_candidates",
]


def intersection_over_union(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def associate(detections, labels):
    """Map each detector candidate to at most one reference object"""
    references_by_stamp = defaultdict(list)
    for reference_id, label in enumerate(labels):
        references_by_stamp[int(label["stamp"])].append(reference_id)

    association = {}
    ambiguous = 0

    for candidate_id, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        hits = []
        for reference_id in references_by_stamp[int(detection["stamp"])]:
            ref_box = labels[reference_id]["box"]
            if (
                ref_box[0] <= center_x <= ref_box[2]
                and ref_box[1] <= center_y <= ref_box[3]
            ):
                hits.append(reference_id)

        if len(hits) > 1:
            ambiguous += 1

        if hits:
            association[candidate_id] = max(
                hits,
                key=lambda ref_id: (
                    intersection_over_union(
                        detection["box"], labels[ref_id]["box"]
                    ),
                    -ref_id,
                ),
            )

    return association, ambiguous


def evaluate(labels, association, accepted_candidates, reference_ids=None):
    """Count each reference object once, even with duplicate detections"""
    if reference_ids is None:
        reference_ids = set(range(len(labels)))
    else:
        reference_ids = set(reference_ids)

    accepted_references = {
        association[candidate_id]
        for candidate_id in accepted_candidates
        if candidate_id in association
        and association[candidate_id] in reference_ids
    }

    real_ids = {
        ref_id for ref_id in reference_ids
        if labels[ref_id]["cls"] == "real"
    }
    decoy_ids = {
        ref_id for ref_id in reference_ids
        if labels[ref_id]["cls"] == "decoy"
    }

    real_accepted = len(real_ids & accepted_references)
    decoys_accepted = len(decoy_ids & accepted_references)

    return {
        "real_accepted": real_accepted,
        "real_total": len(real_ids),
        "real_missed": len(real_ids) - real_accepted,
        "decoys_accepted": decoys_accepted,
        "decoy_total": len(decoy_ids),
        "tpr": real_accepted / len(real_ids) if real_ids else 0.0,
        "fpr": decoys_accepted / len(decoy_ids) if decoy_ids else 0.0,
    }


def check_feature_table(detections):
    with FEATURES_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != len(detections):
        raise ValueError("Feature rows do not match detector candidates.")

    box_columns = ("box_x1", "box_y1", "box_x2", "box_y2")
    for candidate_id, (row, detection) in enumerate(zip(rows, detections)):
        if (
            int(row["candidate_id"]) != candidate_id
            or int(row["stamp"]) != int(detection["stamp"])
            or any(
                abs(float(row[column]) - float(detection["box"][position]))
                > 0.00001
                for position, column in enumerate(box_columns)
            )
        ):
            raise ValueError(
                f"Feature table differs from detections at candidate "
                f"{candidate_id}. Re-run extract_depth_cues.py."
            )


def main():
    detections = json.loads(DETECTIONS_FILE.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    check_feature_table(detections)

    association, ambiguous = associate(detections, labels)
    unmatched = len(detections) - len(association)

    accept_all = range(len(detections))
    baseline = evaluate(labels, association, accept_all)
    print(
        f"Development baseline: real {baseline['real_accepted']}/"
        f"{baseline['real_total']}, decoys {baseline['decoys_accepted']}/"
        f"{baseline['decoy_total']}; {unmatched} unmatched candidates, "
        f"{ambiguous} overlapping references"
    )

    stamps = sorted({int(label["stamp"]) for label in labels})
    label_counts = defaultdict(Counter)
    candidate_counts = Counter(int(det["stamp"]) for det in detections)

    for label in labels:
        label_counts[int(label["stamp"])][label["cls"]] += 1

    manifest = []
    for position, stamp in enumerate(stamps):
        fold = 1 + (3 * position) // len(stamps)
        manifest.append({
            "stamp": stamp,
            "fold": fold,
            "real_references": label_counts[stamp]["real"],
            "decoy_references": label_counts[stamp]["decoy"],
            "detector_candidates": candidate_counts[stamp],
        })

    FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    expected = [
        {key: str(value) for key, value in row.items()}
        for row in manifest
    ]
    cached = None
    if FOLDS_FILE.is_file():
        with FOLDS_FILE.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames == FOLD_COLUMNS:
                cached = list(reader)

    if cached != expected:
        with FOLDS_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FOLD_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest)
    print(f"Temporal folds: {len(stamps)} frames -> {FOLDS_FILE}")


if __name__ == "__main__":
    main()
