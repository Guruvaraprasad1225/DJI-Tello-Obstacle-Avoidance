from ultralytics import YOLO
import cv2
model = YOLO("yolov8s.pt")
VideoAccess = cv2.VideoCapture(0)

while True:
      ret,frame = VideoAccess.read()
      results = model(frame)
      frame = results[0].plot()
      cv2.imshow("YOLO Webcam", frame)
      cv2.waitKey(1)
      if cv2.waitKey(10) & 0xFF == ord('q'):
        break

VideoAccess.release()
cv2.destroyAllWindows()