import cv2
try:
    color_heatmap = cv2.applyColorMap(None, cv2.COLORMAP_JET)
    print("Colormap returned!")
except Exception as e:
    print(f"Colormap: {type(e).__name__}: {e}")
