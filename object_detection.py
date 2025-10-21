# fast_object_detection.py

import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (lightweight)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Resize factor to speed up detection (0.5 = half resolution)
resize_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to smaller size for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    # Run detection
    results = model(small_frame)[0]

    # Draw boxes and labels (scale back to original size)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # scale coordinates back
        x1 = int(x1 / resize_factor)
        y1 = int(y1 / resize_factor)
        x2 = int(x2 / resize_factor)
        y2 = int(y2 / resize_factor)

        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf*100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Fast Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
