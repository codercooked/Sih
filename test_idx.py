import numpy as np
frame = np.zeros((10, 10, 3), dtype=np.uint8)
try:
    frame[[None]]
except Exception as e:
    print(f"Index: {type(e).__name__}: {e}")
