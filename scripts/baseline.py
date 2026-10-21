import sys
sys.path.insert(0, "src")
from rgbd import pair_frames, load_intrinsics, box_position_median

K = load_intrinsics("eval")
pairs = pair_frames("eval")
print(len(pairs), "pairs")

p = [p for p in pairs if p.stamp_ns == 1727164272332739264][0]
print(p.depth_path.name, round(p.gap_ms, 1), "ms")

depth = p.depth_m()
print(box_position_median(depth, (232, 185, 282, 400), K))
