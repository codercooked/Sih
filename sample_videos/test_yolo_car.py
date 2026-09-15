import cv2
import numpy as np
from ultralytics import YOLO

# Create a cartoon car
img = np.ones((500, 800, 3), dtype=np.uint8) * 200

# Body
cv2.rectangle(img, (200, 200), (600, 350), (0, 0, 150), -1) # Dark red body
# Top cabin
cv2.rectangle(img, (300, 100), (500, 200), (0, 0, 150), -1)
# Wheels
cv2.circle(img, (300, 350), 50, (30, 30, 30), -1)
cv2.circle(img, (500, 350), 50, (30, 30, 30), -1)
# Window
cv2.rectangle(img, (320, 120), (480, 200), (200, 200, 200), -1)

# License Plate (large and readable)
plate_x, plate_y = 250, 300
cv2.rectangle(img, (plate_x, plate_y), (plate_x + 160, plate_y + 40), (255, 255, 255), -1)
cv2.rectangle(img, (plate_x, plate_y), (plate_x + 160, plate_y + 40), (0, 0, 0), 2)
cv2.putText(img, "MH01AB1234", (plate_x + 10, plate_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

model = YOLO('yolov8n.pt')
results = model(img)
for r in results:
    for box in r.boxes:
        print(f"Detected class: {int(box.cls[0])} (conf: {box.conf[0]:.2f})")
