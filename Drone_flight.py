from djitellopy import Tello
import time
import keyboard
import cv2
from ultralytics import YOLO

# Initialize the Tello drone and YOLO model
tello = Tello()
video_out = 'Video.mp4'
model_path = r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt"
model = YOLO(model_path)

try:
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    # tello.stream_on()
    # print("Drone is streaming")
    tello.streamon()
    print("Drone is streaming")
    tello.takeoff()
    print("Drone is flying.")

    cap = tello.get_frame_read()
    frame = cap.frame
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out, fourcc, 30.0, (width, height))

    while True:
        frame = cap.frame
        results = model.predict(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Drone Feed", frame)
        out.write(frame)

        if keyboard.is_pressed('x'):
            print("Landing...")
            if tello.get_height() > 0:
                tello.land()
            break

        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        if tello.stream_on:
            tello.streamoff()
    except Exception as e:
        print(f"Stream error: {e}")
    if 'out' in locals():
        out.release()
    cv2.destroyAllWindows()
    try:
        if tello.get_height() > 0:
            tello.land()
    except Exception:
        print("Drone is already on the ground.")
    print("Drone has landed.")
