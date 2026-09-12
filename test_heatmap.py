import numpy as np

frame = np.zeros((10, 10, 3), dtype=np.uint8)
overlay = np.zeros((10, 10, 3), dtype=np.uint8)
mask = np.zeros((10, 10), dtype=bool)

alpha = 0.5
blended = frame.copy()
blended[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(np.uint8)
print("Success")
