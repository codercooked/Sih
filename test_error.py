import numpy as np
try:
    np.int_(None)
except Exception as e:
    print(f"np.int_: {type(e).__name__}: {e}")

try:
    int(None)
except Exception as e:
    print(f"int: {type(e).__name__}: {e}")

