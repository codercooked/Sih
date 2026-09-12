import cv2
import numpy as np

heatmap = np.zeros((0, 0), dtype=np.float32)
norm_heat = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print("norm_heat is None?", norm_heat is None)
if norm_heat is not None:
    print(norm_heat.shape)
