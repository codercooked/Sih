import numpy as np

# Test None cast
try:
    arr = np.array([None], dtype=object)
    arr.astype(np.uint8)
except Exception as e:
    print(f"Error 1: {type(e).__name__}: {e}")

