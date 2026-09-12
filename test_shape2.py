import numpy as np
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
mask = np.zeros((480, 640), dtype=bool)
try:
    frame[mask]
except Exception as e:
    print(f"Shape Mismatch: {type(e).__name__}: {e}")
