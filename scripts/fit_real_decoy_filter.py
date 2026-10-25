"""Select depth/size thresholds on development time blocks only"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_filter import associate, evaluate


ROOT = Path(__file__).resolve().parent.parent
DETECTIONS_FILE = ROOT / "outputs" / "detections_xyz_eval.json"
LABELS_FILE = ROOT / "labels" / "gt_eval.json"
FEATURES_FILE = ROOT / "outputs" / "features" / "eval_candidate_depth_cues.csv"
FOLDS_FILE = ROOT / "outputs" / "manifests" / "eval_temporal_folds.csv"
MODEL_FILE = ROOT / "outputs" / "models" / "real_decoy_filter.json"
OOF_FILE = ROOT / "outputs" / "filter_validation" / "eval_oof_decisions.csv"
REAL_TPR_TARGET = 0.75


def read_number(row, column):
    value = row[column]
    return float(value) if value else float("nan")


def threshold_grid(values):
    """Put possible cutoffs halfway between observed development values"""
    unique = np.unique(values[np.isfinite(values)])
    if not unique.size:
        raise ValueError("No finite values to calibrate the filter")
    below_minimum = np.nextafter(unique[0], -np.inf)
    midpoints = unique[:-1] + (unique[1:] - unique[:-1]) / 2
    return np.concatenate(([below_minimum], midpoints))


def fit_thresholds(height, contrast, labels, association, train_refs):
    real_refs = np.array(sorted(
        i for i in train_refs if labels[i]["cls"] == "real"
    ))
    decoy_refs = np.array(sorted(
        i for i in train_refs if labels[i]["cls"] == "decoy"
    ))
    if real_refs.size == 0 or decoy_refs.size == 0:
        raise ValueError("Training fold requires both reference classes")

    candidates = np.array([
        i for i, reference_id in association.items()
        if reference_id in train_refs
        and np.isfinite(height[i])
        and np.isfinite(contrast[i])
    ], dtype=int)
    if not candidates.size:
        raise ValueError("No labeled candidates with both cues")

    candidate_refs = np.array(
        [association[i] for i in candidates], dtype=int
    )
    height_grid = threshold_grid(height[candidates])
    contrast_grid = threshold_grid(contrast[candidates])
    minimum_tp = math.ceil(REAL_TPR_TARGET * real_refs.size)

    best_key = None
    best_thresholds = None

    for height_cut in height_grid:
        selected = height[candidates] >= height_cut

        # Each reference passes if any of its candidates passes
        max_contrast = np.full(len(labels), -np.inf)
        np.maximum.at(
            max_contrast,
            candidate_refs[selected],
            contrast[candidates[selected]],
        )

        real_scores = np.sort(max_contrast[real_refs])
        decoy_scores = np.sort(max_contrast[decoy_refs])
        real_scores = real_scores[np.isfinite(real_scores)]
        decoy_scores = decoy_scores[np.isfinite(decoy_scores)]

        tp = real_scores.size - np.searchsorted(
            real_scores, contrast_grid
        )
        fp = decoy_scores.size - np.searchsorted(
            decoy_scores, contrast_grid
        )

        for index in np.flatnonzero(tp >= minimum_tp):
            # First minimize accepted decoys then retain more real objects
            key = (int(fp[index]), -int(tp[index]))
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = (
                    float(height_cut),
                    float(contrast_grid[index]),
                )

    if best_thresholds is None:
        raise ValueError(
            "No rule reaches the real-object TPR target "
            "on these training frames"
        )
    return best_thresholds


def predict(height, contrast, height_cut, contrast_cut):
    return (
        np.isfinite(height)
        & np.isfinite(contrast)
        & (height >= height_cut)
        & (contrast >= contrast_cut)
    )


def print_counts(title, counts):
    print(
        f"{title}: real "
        f"{counts['real_accepted']}/{counts['real_total']} "
        f"(TPR {counts['tpr']:.3f}), decoys "
        f"{counts['decoys_accepted']}/{counts['decoy_total']} "
        f"(FPR {counts['fpr']:.3f})"
    )


def report_results(report):
    print_counts("Development OOF", report["out_of_fold"])
    print(
        f"Frozen thresholds: height >= {report['height_threshold_m']:.4f} m, "
        f"relative depth >= {report['relative_depth_threshold']:.4f} "
        f"-> {MODEL_FILE}"
    )


def main():
    files = [
        DETECTIONS_FILE,
        LABELS_FILE,
        FEATURES_FILE,
        FOLDS_FILE,
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    detections = json.loads(
        DETECTIONS_FILE.read_text(encoding="utf-8")
    )
    labels = json.loads(
        LABELS_FILE.read_text(encoding="utf-8")
    )
    with FEATURES_FILE.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != len(detections):
        raise ValueError(
            "Feature table has a different number of candidates"
        )

    for index, (row, detection) in enumerate(zip(rows, detections)):
        if (
            int(row["candidate_id"]) != index
            or int(row["stamp"]) != int(detection["stamp"])
        ):
            raise ValueError(
                "Feature table is out of sync with detections"
            )
        if any(
            abs(
                float(row[f"box_{key}"])
                - float(detection["box"][position])
            ) > 1e-5
            for position, key in enumerate(("x1", "y1", "x2", "y2"))
        ):
            raise ValueError(
                "Feature table boxes differ from detections"
            )

    height = np.array([
        read_number(row, "projected_box_height_m")
        for row in rows
    ])
    contrast = np.array([
        read_number(row, "relative_ring_minus_core")
        for row in rows
    ])
    association, _ = associate(detections, labels)

    with FOLDS_FILE.open(newline="", encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))

    label_stamps = {int(label["stamp"]) for label in labels}
    if len(manifest) != len(label_stamps):
        raise ValueError(
            "Time-block manifest does not match reference frames"
        )

    fold_by_stamp = {
        int(row["stamp"]): int(row["fold"])
        for row in manifest
    }
    if set(fold_by_stamp) != label_stamps:
        raise ValueError(
            "Time-block manifest has missing or duplicate frames"
        )
    if set(fold_by_stamp.values()) != {1, 2, 3}:
        raise ValueError("Expected three consecutive time blocks")
    if any(
        int(detection["stamp"]) not in fold_by_stamp
        for detection in detections
    ):
        raise ValueError(
            "Detector frame is missing from time-block manifest"
        )

    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    dependencies = files + [
        Path(__file__),
        Path(__file__).with_name("evaluate_filter.py"),
    ]
    latest_input = max(
        path.stat().st_mtime for path in dependencies
    )

    if MODEL_FILE.is_file() and OOF_FILE.is_file():
        output_time = min(
            MODEL_FILE.stat().st_mtime,
            OOF_FILE.stat().st_mtime,
        )
        if output_time >= latest_input:
            cached = json.loads(
                MODEL_FILE.read_text(encoding="utf-8")
            )
            with OOF_FILE.open(
                newline="", encoding="utf-8"
            ) as file:
                row_count = sum(
                    1 for _ in csv.DictReader(file)
                )
            if (
                row_count == len(detections)
                and cached.get("source_sha256") == source_hashes
            ):
                print(
                    "Checkpoint found: using calibrated "
                    "development filter"
                )
                report_results(cached)
                return

    all_refs = set(range(len(labels)))
    oof = np.zeros(len(detections), dtype=bool)
    folds = []

    for fold_id in (1, 2, 3):
        held_refs = {
            i for i, label in enumerate(labels)
            if fold_by_stamp[int(label["stamp"])] == fold_id
        }
        train_refs = all_refs - held_refs

        height_cut, contrast_cut = fit_thresholds(
            height,
            contrast,
            labels,
            association,
            train_refs,
        )
        decisions = predict(
            height,
            contrast,
            height_cut,
            contrast_cut,
        )
        in_held_frames = np.array([
            fold_by_stamp[int(detection["stamp"])] == fold_id
            for detection in detections
        ])
        oof[in_held_frames] = decisions[in_held_frames]

        folds.append({
            "fold": fold_id,
            "height_threshold_m": height_cut,
            "relative_depth_threshold": contrast_cut,
            "baseline": evaluate(
                labels,
                association,
                range(len(detections)),
                held_refs,
            ),
            "held_out": evaluate(
                labels,
                association,
                np.flatnonzero(decisions),
                held_refs,
            ),
        })

    final_height, final_contrast = fit_thresholds(
        height,
        contrast,
        labels,
        association,
        all_refs,
    )
    report = {
        "method": (
            "projected_box_height AND relative_ring_depth"
        ),
        "decision_rule": (
            "Accept if both features are finite and "
            "each is at least its fitted threshold"
        ),
        "selection": (
            "Training frames only: minimize accepted decoy "
            "objects subject to real TPR >= 0.75; then "
            "maximize accepted real objects. Remaining ties "
            "choose the lowest height then contrast threshold"
        ),
        "height_threshold_m": final_height,
        "relative_depth_threshold": final_contrast,
        "folds": folds,
        "out_of_fold": evaluate(
            labels,
            association,
            np.flatnonzero(oof),
        ),
        "accept_all_baseline": evaluate(
            labels,
            association,
            range(len(detections)),
        ),
        "fit_on_all_development": evaluate(
            labels,
            association,
            np.flatnonzero(
                predict(
                    height,
                    contrast,
                    final_height,
                    final_contrast,
                )
            ),
        ),
        "candidate_count": len(detections),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "source_sha256": source_hashes,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    OOF_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with OOF_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "candidate_id",
            "stamp",
            "fold",
            "accepted_out_of_fold",
        ])
        for i, detection in enumerate(detections):
            stamp = int(detection["stamp"])
            writer.writerow([
                i,
                stamp,
                fold_by_stamp[stamp],
                int(oof[i]),
            ])

    report_results(report)


if __name__ == "__main__":
    main()

