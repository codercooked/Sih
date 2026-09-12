import cv2
import numpy as np
heatmap = np.zeros((10, 10), dtype=np.float32)
norm_heat = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print(norm_heat)
