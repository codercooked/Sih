import cv2
import numpy as np
import sys
import os

# Add scratch dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.heatmap import HeatmapAccumulator

acc = HeatmapAccumulator(640, 480)
acc.add_point(100, 100)
acc.add_point(None, None) # Let's see what happens!
acc.update()

frame = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    res = acc.blend(frame)
    print("Blend successful. Shape:", res.shape)
except Exception as e:
    import traceback
    traceback.print_exc()

