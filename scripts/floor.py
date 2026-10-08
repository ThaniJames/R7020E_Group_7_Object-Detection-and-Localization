import sys
sys.path.insert(0, "src")
import numpy as np
import open3d as o3d
from rgbd import pair_frames, load_intrinsics, depth_to_points, object_centre
o3d.utility.random.seed(0) 

K = load_intrinsics("eval")
p = [p for p in pair_frames("eval") if p.stamp_ns == 1727164272332739264][0]
depth = p.depth_m()

points = depth_to_points(depth, K) 
print(len(points), "points")

cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
plane, inliers = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)

a, b, c, d = plane
print("plane:", np.round(plane, 3))
print("points on floor:", len(inliers), "of", len(points))
print("camera height above floor:", round(abs(d), 2), "m")


centre = object_centre(depth, (232, 185, 282, 400), K)
height = -(a * centre[0] + b * centre[1] + c * centre[2] + d)
print("extinguisher centre:", centre.round(3))
print("height above floor: ", round(height, 2), "m")
