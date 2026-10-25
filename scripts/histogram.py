import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "data/Project_1/raw/eval/camera_depth_image_raw/camera_depth_image_1727164272306629143.png"
depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

box = depth[150:400, 190:330] 
z = box[box > 0] / 1000

plt.hist(z, bins=100) # 100 bars between the smallest and largest depth
plt.axvline(np.median(z), color="red", label=f"median {np.median(z):.2f} m")
plt.axvline(z.mean(), color="orange", label=f"mean {z.mean():.2f} m")
plt.xlabel("depth (m)")
plt.ylabel("number of pixels")
plt.legend()
plt.savefig("outputs/histogram.png")
plt.show()

counts, edges = np.histogram(z, bins=100) # same histogram, but as numbers instead of a picture
peak = counts.argmax() # index of the tallest bar
peak_depth = (edges[peak] + edges[peak + 1]) / 2   # centre of that bar, in metres

keep = (z > peak_depth - 0.05) & (z < peak_depth + 0.15)
obj = z[keep]

print("peak at       ", round(peak_depth, 3), "m")
print("kept pixels   ", len(obj), "of", len(z))
print("object median ", round(np.median(obj), 3), "m")
