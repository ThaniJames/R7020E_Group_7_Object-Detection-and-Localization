# Project 1: Fire Extinguisher Detection, Decoy Rejection and 3D Localization

## 1. Project details

R7020E, Group 7. The system finds fire extinguishers in RGB-D frames from an Orbbec stereo depth camera.
It rejects printed decoys and gives the 3D position of the real extinguisher in metres.

- **Course task:** detect fire extinguishers, reject printed or displayed decoys, and estimate the 3D position of the
  real one, from recorded RGB-D data (colour + depth).
- **Data:** two recordings from the course, `eval` and `test`, each with colour, depth and IR images and camera info,
  plus the Roboflow *FireExtinguisher* v2 photos for training the detector.
- **Targets:**

| Metric | Target |
|---|---|
| Real-object TPR (real extinguishers detected and accepted as real / all real) | ≥ 0.75 |
| Decoy FPR (decoys accepted as real / all decoys) | ≤ 0.15 |

A detection belongs to an annotated object when the centre of its box lies inside the reference box.

---

## 2. Team and contributions

| Member | Main responsibility |
|---|---|
| Kelechi Athanesious Ezekwe | Task 1: training data, YOLO training, detector |
| Shravani Kuchi | Task 2: real vs decoy filter (depth cues, threshold fitting, validation) |
| Thanigaiarasu Shanmugam | Task 3: 3D localization, frame pairing, tracking extension; evaluation pipeline and final test protocol |

---

## 3. Architecture

```
colour + depth frame (paired by time, ≤ 50 ms)
        │
        ▼
 Task 1  DETECT   YOLO11n → boxes (conf ≥ 0.10, real and printed)
        │
        ▼
 Task 2  FILTER   depth cues: box height in metres + free space around the box → real / decoy
        │
        ▼
 Task 3  LOCATE   object's own depth → cylinder centre → X, Y, Z (m), camera frame
        │         (extension: tracking keeps a confirmed real object through 1–2 rejected frames)
        ▼
 output: real extinguisher + (X, Y, Z)
```

- **Hand-offs:** Task 1 gives boxes with timestamps and confidence; Task 3 pairs each frame with its depth frame and
  adds X, Y, Z (`outputs/detections_xyz_{split}.json`); Task 2 adds `real_score` (1 = real, 0 = decoy)
  (`outputs/scored_detections_xyz_{split}.json`); evaluation and tracking read that file.
- **Diagram:** `doc/system_architecture.png` (source: `doc/system_architecture.puml`).

---

## 4. Method and settings

### Task 1: Detection
- YOLO11n (Ultralytics), pretrained on COCO and fine-tuned on Roboflow *FireExtinguisher* v2 (CC BY 4.0,
  https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr). Validation mAP50 ≈ 0.93.
- The confidence threshold is low on purpose (0.10): a missed real extinguisher is lost for good, while extra boxes
  are removed by the filter.

### Task 2: Real vs decoy (`scripts/real_vs_decoy_main.py` and the scripts it calls)
- **Height:** box height in metres = box height (px) × object depth / fy. Object depth = 25th percentile of the
  inner half of the box.
- **Depth contrast:** (median depth of a ring around the box − object depth) / object depth.
  A print sits on a panel (contrast ≈ 0); the real extinguisher stands in free space.
- **Rule:** real if height ≥ **0.5554 m** AND contrast ≥ **0.1922**. Thresholds are fitted on eval only: the fewest
  accepted decoys while keeping TPR ≥ 0.75, checked with 3-fold cross-validation over time blocks
  (out-of-fold TPR 0.98, FPR 0.013). Frozen in `outputs/models/real_decoy_filter.json`.

### Task 3: 3D localization (`src/rgbd.py`)
- **Depth:** the Orbbec computes dense disparity between its two IR cameras (baseline 0.1004 m) and depth
  Z = f·B/d. Depth is registered to colour and stored in millimetres (0 = missing).
  Colour and depth frames are paired by timestamp (≤ 50 ms).
- **Object centre** (`object_centre`):
  1. Histogram of the box depths (100 bins): the tallest bar is the object; keep −0.05 / +0.15 m around it.
  2. Front surface = 5th percentile of those depths.
  3. Centre = front + cylinder radius 0.067 m (measured on an eval frame).
  4. Back-project the box centre: X = (u − cx)·Z / fx, Y = (v − cy)·Z / fy.
- **Coordinate system:** colour camera optical frame, origin at the camera, **X right, Y down, Z forward, metres**.
  Intrinsics come from `camera_info` (fx ≈ 306.0, fy ≈ 306.1, cx ≈ 318.5, cy ≈ 201.4).
  Lens distortion is not corrected (a known limitation).
- **Tracking extension** (`scripts/track_vote.py`):
  - A track counts as confirmed real after 3 accepted frames.
  - A box the filter rejects is kept as real if its centre is within 80 px of the track, its height is at least
    half the track's box height, and the track was accepted within the last 2 frames. At most 2 rescues in a row.
  - Online: it uses past frames only.

---

## 5. Evaluation protocol (training / development / evaluation)

| Subset | Data | Used for |
|---|---|---|
| Training | Roboflow *FireExtinguisher* v2 (web photos, 1 class) | Training YOLO11n |
| Development | `eval` recording: 99 frames, 99 real + 160 decoy occurrences | Every choice: filter thresholds, tracking settings, 3D constants |
| Evaluation | `test` recording: 67 frames, 45 real + 103 decoy occurrences | Final numbers, run **once** with everything frozen |

- **The test recording is locked in the code.** `src/rgbd.py` refuses to load it unless the environment variable
  `FINAL_TEST=1` is set, and only `scripts/final_test.py` sets it.
- **Detector:** one YOLO11n model (`models/best.pt`), trained on Roboflow photos only with the Ultralytics
  default settings. It never saw the eval or test recordings.
- **Tracking settings (on eval):** grid of 80/120/160 px × 1/2/3 frames. Rule: rescue the most real objects with
  0 extra decoys, then the fewest extra background boxes, then the smallest values → **80 px, 2 frames**
  (`outputs/tracking_selection.json`).
- **Reference labels** (`labels/gt_{eval,test}.json`):
  1. The detector proposed boxes (`labels/proposals*.json`, conf ≥ 0.05).
  2. Each box was reviewed visually as real / part of the real one / decoy / junk (`labels/review_decisions.json`).
     The review was AI-assisted and spot-checked by the team.
  3. Missed objects were added by hand (`labels/manual_boxes.json`).
  4. `scripts/build_labels.py` merges this into one box per object.

---

## 6. Results (final evaluation, `test` recording, run once)

Detector confidence ≥ 0.10. A detection belongs to an annotated object when the centre of its box lies inside the
reference box. TPR = real objects accepted / all real objects. Decoy FPR = decoys accepted / all decoys.

| | Baseline | Final system | Final + tracking (extension) |
|---|---|---|---|
| Real accepted / missed | 45 / 0 | **43 / 2** | 45 / 0 |
| Decoys accepted | 62 / 103 | **0 / 103** | 0 / 103 |
| **Real-object TPR** (target ≥ 0.75) | 1.00 | **0.96** | 1.00 |
| **Decoy FPR** (target ≤ 0.15) | 0.60 | **0.00** | 0.00 |
| Background boxes accepted | 22 | 2 | 2 |
| Height above floor (3D check) | 0.26 ± 0.07 m | **0.49 ± 0.05 m** | 0.48 ± 0.06 m |

- **Baseline:** same detector, every detection accepted as real, 3D from the median depth of the whole box.
- **Speed** (`scripts/timing.py`, Apple M4 Max, MPS): detection 14.6 + filter 0.4 + 3D 0.2 = **15.2 ms per frame (≈ 66 FPS)**.
- **3D check:** there is no 3D ground truth. The extinguisher stands still on the floor, so its height above the
  floor plane (RANSAC) must stay constant while the camera moves.
- **The 2 misses:** the camera was turning, and both colour frames were paired with the same depth frame,
  taken 28 and 45 ms earlier. In that depth frame the object has already moved out of the box. The box and the ring
  around it then see floor, so there is no depth contrast and the filter rejects the object. Tracking keeps both.

Result files: `outputs/results_test.json`, `outputs/results_tracked_test.json`. Figures: `outputs/figures/`.
Demo video: `outputs/demo_test.mp4`.

---

## 7. Limitations

- **Timing between colour and depth.** Pairing allows up to 50 ms. While the camera turns quickly, the depth frame
  shows an older moment; this caused both test misses.
- **One room.** The filter fits this room's decoys (prints on panels and blocks). A life-size print on a free-standing
  board, or a screen, could fool it.
- **Camera frame only.** Positions are relative to the moving camera. A fixed room frame would need odometry or the IMU.
- **No lens undistortion** before back-projection; the error grows towards the image edges.
- **Labels.** The labels were AI-assisted and spot-checked, not fully hand-made.

---

## 8. Repository structure

```
src/
  rgbd.py           frame pairing, intrinsics, back-projection, object depth and centre (Task 3), test lock
  contracts.py      data formats shared between the tasks
scripts/
  detect.py                 YOLO + 3D per paired frame → detections_xyz_{split}.json
  detect_all.py             YOLO on all colour images → detections_{split}.json (Task 1 output)
  real_vs_decoy_main.py     Task 2 pipeline; calls check_filter_inputs, extract_depth_cues,
                            evaluate_filter, fit_real_decoy_filter, extract_test_depth_cues,
                            apply_real_decoy_filter, evaluate_test_filter
  evaluate.py               baseline vs final: TPR, FPR, background, height above floor
  track_vote.py             tracking extension
  evaluate_tracking.py      final vs final + tracking
  select_tracking.py        tracking settings, chosen on eval
  final_test.py             the single run on the test recording
  timing.py                 time per stage
  figures_3d.py, figures_eval.py    slide figures
  build_labels.py           reference labels from the reviewed proposals
  check_data.py             sanity checks: pairing, back-projection, alignment
  collect_frames.py         copies paired frames for labelling
  baseline.py, floor.py, histogram.py, radius.py, height.py, track.py   3D exploration
                            (height.py and track.py read test, so they stop at the test lock)
  Note: check_data.py and collect_frames.py also stop at the test lock (they were used before
  the protocol). detect_all.py reads both recordings directly (it writes outputs/detections_{split}.json).
  baseline_detector.py, evaluate_detector_fixed.py, prepare_dataset_for_training.py
                            early Task 1 versions, not used for the results
notebooks/          Colab training notebook (early training setup, not used for the final model)
labels/             reference labels and the review files they are built from
models/best.pt      final detector
outputs/            detections, results, figures, frozen Task 2 thresholds, demo video
doc/                system architecture diagram
```

---

## 9. Setup

**Python 3.11.** Install the exact versions used for the results:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data** (not in git). Place the course data and the Roboflow export like this:

```
data/
├── Project_1/
│   └── raw/
│       ├── eval/    camera_color_image_raw/, camera_depth_image_raw/, camera_*_camera_info/, ...
│       └── test/    same folders
└── FireExtinguisher.v2i.yolov11/    Roboflow export: train/, valid/, test/, data.yaml
```

Use the lowercase folder name `data`. macOS ignores the case, Linux and Colab do not.

---

## 10. Reproduce the results

Run every command from the repository root. Steps 1–4 use only the eval recording.
Step 5 is the only one that opens the test recording; run it once, at the end.

**1. Detector (optional: `models/best.pt` is already included)**
```bash
yolo train model=yolo11n.pt data=data/FireExtinguisher.v2i.yolov11/data.yaml epochs=100 patience=20 \
     imgsz=640 batch=16 device=mps seed=0 name=yolo11n_final      # device=0 for an NVIDIA GPU, device=cpu otherwise
cp runs/detect/yolo11n_final/weights/best.pt models/best.pt
```

**2. Detection + 3D on eval** → `outputs/detections_xyz_eval.json` (already included)
```bash
python scripts/detect.py eval
```

**3. Task 2 filter: fit the thresholds on eval and apply them to eval**
```bash
python scripts/real_vs_decoy_main.py development     # → outputs/models/real_decoy_filter.json
python scripts/apply_real_decoy_filter.py eval        # → outputs/scored_detections_xyz_eval.json
```

**4. Development results and tracking settings (eval)**
```bash
python scripts/evaluate.py eval            # baseline vs final on eval
python scripts/track_vote.py eval
python scripts/evaluate_tracking.py eval
python scripts/select_tracking.py          # → outputs/tracking_selection.json (80 px, 2 frames)
```

**5. Final check on test (once)**
```bash
python scripts/final_test.py
```
This runs, in order: detection + 3D, the Task 2 filter with frozen thresholds, baseline vs final
(`outputs/results_test.json`), tracking (`outputs/results_tracked_test.json`), speed, and the figures
(`outputs/figures/`).
