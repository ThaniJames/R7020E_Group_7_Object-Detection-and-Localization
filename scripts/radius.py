import sys
sys.path.insert(0, "src")
import numpy as np
from rgbd import pair_frames, load_intrinsics, box_position_median, object_centre

K = load_intrinsics("eval")
p = [p for p in pair_frames("eval") if p.stamp_ns == 1727164272332739264][0]
depth = p.depth_m()

row = depth[300, 232:282]
print(np.round(row, 2))

on_object = (row > 1.0) & (row < 1.25)
width_px = on_object.sum()
Z = np.median(row[on_object])
print("width:", width_px, "px =", round(width_px * Z / K.fx, 3), "m")


box = (232, 185, 282, 400)
print("baseline :", box_position_median(depth, box, K).round(3))
print("centre   :", object_centre(depth, box, K).round(3))

box = (190, 150, 330, 400)
print("LOOSE baseline:", box_position_median(depth, box, K).round(3))
print("LOOSE centre:  ", object_centre(depth, box, K).round(3))
