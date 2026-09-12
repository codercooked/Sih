import cv2
import numpy as np

norm_heat = np.zeros((10, 10), dtype=np.uint8)
color_heatmap = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)

print("Type:", color_heatmap.dtype)
print("Contains None?", None in color_heatmap)

