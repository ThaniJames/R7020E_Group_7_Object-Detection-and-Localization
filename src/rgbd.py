#RGB-D loading, colour/depth pairing and pixel -> 3D conversion

from dataclasses import dataclass
from pathlib import Path
import os
import re

import numpy as np
import cv2

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "Project_1" / "raw"
DEPTH_SCALE = 0.001  # depth PNG stores millimetres


# Camera calibration from the camera_info file, needed to turn a pixel + depth into 3D (metres).
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    D: np.ndarray

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

#  Timestamp in nanoseconds from a filename, ex: camera_color_image_1727164479163392418.png -> 1727164479163392418
def _stamp(path: Path):
    return int(path.stem.rsplit("_", 1)[1])

# Read a camera_info .txt file into Intrinsics.
def read_camera_info(path: Path):
    text = path.read_text()
    K = [float(v) for v in re.search(r"K: \[([^\]]*)\]", text).group(1).split(",")]
    D = [float(v) for v in re.search(r"D: \[([^\]]*)\]", text).group(1).split(",")]
    w = int(re.search(r"^width: (\d+)", text, re.M).group(1))
    h = int(re.search(r"^height: (\d+)", text, re.M).group(1))
    return Intrinsics(K[0], K[4], K[2], K[5], w, h, np.array(D))

# Calibration of one recording ("test" or "eval"), stream "color" or "depth".
def load_intrinsics(split: str, stream: str = "color"):
    info_dir = DATA_ROOT / split / f"camera_{stream}_camera_info"
    return read_camera_info(sorted(info_dir.glob("*.txt"))[0])

# One colour frame and the depth frame closest in time.
@dataclass
class FramePair:
    color_path: Path
    depth_path: Path
    stamp_ns: int # colour timestamp
    gap_ms: float # colour - depth timestamp difference

    def color(self) -> np.ndarray:
        return cv2.imread(str(self.color_path), cv2.IMREAD_COLOR)

    #Depth image in metres (PNG millimetres x 0.001), float32, shape (400, 640); 0 = no measurement.
    def depth_m(self):
        raw = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) * DEPTH_SCALE

# Match every colour frame to the nearest-in-time depth frame; drop pairs further apart than max_gap_ms.
def pair_frames(split: str, max_gap_ms: float = 50.0):
    if split == "test" and os.environ.get("FINAL_TEST") != "1":
        raise RuntimeError("test recording is locked: only scripts/final_test.py may use it")
    colors = sorted((DATA_ROOT / split / "camera_color_image_raw").glob("*.png"), key=_stamp)
    depths = sorted((DATA_ROOT / split / "camera_depth_image_raw").glob("*.png"), key=_stamp)
    d_stamps = np.array([_stamp(p) for p in depths])
    pairs = []
    for c in colors:
        t = _stamp(c)
        i = int(np.argmin(np.abs(d_stamps - t)))
        gap_ms = abs(int(d_stamps[i]) - t) / 1e6
        if gap_ms <= max_gap_ms:
            pairs.append(FramePair(c, depths[i], t, gap_ms))
    return pairs

# Pixel coordinates + depth (m) to Nx3 points in the colour optical frame.
def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: Intrinsics):
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=-1)

# All valid depth pixels to Nx3 points.
def depth_to_points(depth_m: np.ndarray, K: Intrinsics, mask: np.ndarray | None = None):
    valid = depth_m > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    return backproject(u.astype(np.float32), v.astype(np.float32), depth_m[v, u], K)

# Baseline 3D position: median depth inside the box, back-projected at the box centre.
def box_position_median(depth_m: np.ndarray, box_xyxy, K: Intrinsics):
    x1, y1, x2, y2 = (int(round(b)) for b in box_xyxy)
    patch = depth_m[max(y1, 0):y2, max(x1, 0):x2]
    z = patch[patch > 0]
    if z.size == 0:
        return None
    uc, vc = (x1 + x2) / 2, (y1 + y2) / 2
    return backproject(np.float32(uc), np.float32(vc), np.float32(np.median(z)), K)

#  Depths (m) of the pixels in the box that belong to the object: find the tallest depth-histogram bar then keep pixels from front m in front of it to back m behind it.
def object_depths(depth_m: np.ndarray, box_xyxy, front: float = 0.05, back: float = 0.15, bins: int = 100):
    x1, y1, x2, y2 = (int(round(b)) for b in box_xyxy)
    patch = depth_m[max(y1, 0):y2, max(x1, 0):x2]
    z = patch[patch > 0]
    if z.size == 0:
        return z
    counts, edges = np.histogram(z, bins=bins)
    peak = counts.argmax() # index of the tallest bar
    peak_depth = (edges[peak] + edges[peak + 1]) / 2 # centre of that bar
    keep = (z > peak_depth - front) & (z < peak_depth + back)
    return z[keep]

# 3D centre of a cylindrical object: front surface + radius back-projected at the box centre.
def object_centre(depth_m: np.ndarray, box_xyxy, K: Intrinsics, radius: float = 0.067):
    z = object_depths(depth_m, box_xyxy) 
    if z.size == 0:
        return None
    front = np.percentile(z, 5)  # nearly the closest point, ignoring noise
    x1, y1, x2, y2 = box_xyxy
    uc, vc = (x1 + x2) / 2, (y1 + y2) / 2             # box centre pixel
    return backproject(np.float32(uc), np.float32(vc), np.float32(front + radius), K)
