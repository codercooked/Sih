import numpy as np
try:
    arr = np.array([(1, 2, 3)], dtype=object)
    arr.astype(np.uint8)
except Exception as e:
    print(f"Tuple: {type(e).__name__}: {e}")
