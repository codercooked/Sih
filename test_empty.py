import numpy as np
frame = np.zeros((10, 10, 3), dtype=np.uint8)
mask = np.zeros((10, 10), dtype=bool)
alpha = 0.5

print(frame[mask])
res = (frame[mask] * (1 - alpha) + frame[mask] * alpha).astype(np.uint8)
print("Res:", res)
