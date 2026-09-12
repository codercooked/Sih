import cv2
import numpy as np
try:
    arr = np.array([None], dtype=object)
    cv2.addWeighted(arr, 0.5, arr, 0.5, 0)
except Exception as e:
    print(f"addWeighted None: {type(e).__name__}: {e}")
