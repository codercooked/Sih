import cv2
import numpy as np

frame = np.zeros((10, 10, 3), dtype=np.uint8)
overlay = np.zeros((10, 10, 3), dtype=np.uint8)
mask = np.ones((10, 10), dtype=bool)

alpha = 0.5
try:
    cv2.addWeighted(frame[mask], 1 - alpha, overlay[mask], alpha, 0)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
