import numpy as np

frame = np.zeros((480, 640, 3), dtype=np.uint8)
mask = np.zeros((720, 1280), dtype=bool)

try:
    print(frame[mask])
except Exception as e:
    print(f"Error 1: {type(e).__name__}: {e}")

