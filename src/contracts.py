#Contract file

from dataclasses import dataclass

import numpy as np


# ---- Task 3 needs these from Task 1 ----

@dataclass
class Detection:
    stamp: int # colour frame timestamp, nanoseconds
    box: tuple # (x1, y1, x2, y2) in pixels
    conf: float # detector confidence, 0.0 to 1.0

#Task 1: all detections for one recording ("test" or "eval")
def detect_all(split: str) -> list[Detection]:
    raise NotImplementedError("Task 1")


# ---- Task 3 needs these from Task 2 ----

@dataclass
class ScoredDetection(Detection):
    real_score: float   # 0.0 = printed decoy, 1.0 = real extinguisher

#Task 2: real/decoy score for every detection of one recording ("test" or "eval")
def score_all(split: str) -> list[ScoredDetection]:
    raise NotImplementedError("Task 2")


# ---- Task 2 needs these from Task 3 ----

#Task 3: depths (m) of only the object's pixels inside box, without the floor and wall behind it.
def object_depths(depth: np.ndarray, box: tuple) -> np.ndarray:
    raise NotImplementedError("Task 3")
