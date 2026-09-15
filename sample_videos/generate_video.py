import urllib.request
import cv2
import numpy as np

url = "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=800"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
resp = urllib.request.urlopen(req)
arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)

# Ensure the image loaded successfully
if img is not None:
    # Just draw a massive fake white license plate with bold black text directly onto the car!
    # This guarantees YOLO sees the car AND EasyOCR sees the plate.
    plate_x = img.shape[1] // 2 - 150
    plate_y = img.shape[0] - 120
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + 300, plate_y + 80), (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + 300, plate_y + 80), (0, 0, 0), 3)
    cv2.putText(img, "DL8CA4932", (plate_x + 20, plate_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    
    car = img
    
    width, height = 1280, 720
    fps = 30
    duration = 5

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('09_license_plate_demo.mp4', fourcc, fps, (width, height))

    for i in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 150 # Grey background
        
        car_h, car_w = car.shape[:2]
        progress = i / (fps * duration)
        x = int(width - progress * (width + car_w))
        y = height // 2 - car_h // 2 + 100 
        
        x1, x2 = max(0, x), min(width, x + car_w)
        y1, y2 = max(0, y), min(height, y + car_h)
        
        car_x1 = max(0, -x)
        car_x2 = car_x1 + (x2 - x1)
        car_y1 = max(0, -y)
        car_y2 = car_y1 + (y2 - y1)
        
        if x1 < x2 and y1 < y2:
            frame[y1:y2, x1:x2] = car[car_y1:car_y2, car_x1:car_x2]
            
        out.write(frame)

    out.release()
    print("Success")
else:
    print("Failed to load image")
