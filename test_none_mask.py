import numpy as np

frame = np.zeros((10, 10, 3), dtype=np.uint8)
overlay = np.zeros((10, 10, 3), dtype=np.uint8)
blended = frame.copy()

mask = None
alpha = 0.5

try:
    blended[mask] = (frame[mask] * (1 - alpha) + overlay[mask] * alpha).astype(np.uint8)
    print("Success")
except Exception as e:
    print(f"Mask None: {type(e).__name__}: {e}")

