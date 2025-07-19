from ultralytics import YOLO
import cv2
import numpy as np
from boxmot import DeepOCSORT  # Importing DeepOCSORT from boxmot
from pathlib import Path

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize DeepOCSORT
tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cpu',
    fp16=False,
)

# Video capture from laptop camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows for DirectShow backend

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured from the camera. Exiting.")
        break

    # Use YOLO to detect objects
    results = model(frame)

    # Extract detections (bounding boxes, confidences, and class IDs)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = float(box.conf[0].item())       # Confidence score
            class_id = int(box.cls[0].item())      # Class ID
            detections.append([x1, y1, x2, y2, conf, class_id])

    # Convert detections to a 2D NumPy array (N, 6) or an empty array if no detections
    detections_numpy = np.array(detections, dtype=np.float32) if len(detections) > 0 else np.zeros((0, 6), dtype=np.float32)

    # Debug: Print the shape of detections_numpy
    print(f"Detections shape: {detections_numpy.shape}")
    print(detections_numpy)

    # Ensure the detection array has valid dimensions before updating the tracker
    if len(detections_numpy.shape) == 2 and detections_numpy.shape[1] == 6:
        tracks = tracker.update(detections_numpy,frame)

        # Draw tracking results
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
