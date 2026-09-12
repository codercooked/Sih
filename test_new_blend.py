import cv2
import numpy as np

frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
overlay = np.zeros((10, 10, 3), dtype=np.uint8)
overlay[5:8, 5:8] = 200 # heat
mask = overlay[:,:,0] > 0

alpha = 0.5
blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
blended[~mask] = frame[~mask]

print(blended[6, 6]) # Should be blended: 100*0.5 + 200*0.5 = 150
print(blended[0, 0]) # Should be frame: 100
