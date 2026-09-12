import numpy as np

try:
    np.array([None], dtype=object) * 0.5
except Exception as e:
    print(f"Mul: {type(e).__name__}: {e}")

