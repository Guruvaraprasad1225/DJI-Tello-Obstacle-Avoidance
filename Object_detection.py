import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import os

# Global variables
video_out = 'video.mp4'
model_path = r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt"  # Corrected path
tello = Tello()

def initialize_drone():
    tello.connect()
    print(f"Drone battery level: {tello.get_battery()}%")
    tello.streamon()
    tello.takeoff()

def get_frame():
    frame_read = tello.get_frame_read()
    while True:
        frame = frame_read.frame
        if frame is not None:
            yield frame
        else:
            print("No frame received. Rechecking...")

def main():
    global out  # Declare 'out' as global to ensure it's accessible in the cleanup function
    try:
        initialize_drone()
        model = YOLO(model_path, task='detect')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out, fourcc, 20.0, (960, 720))

        for frame in get_frame():
            results = model(frame)
            # Render the detections onto the frame
            if results.xyxy[0].nelement() == 0:
                print("No detections")
            else:
                for result in results.xyxy[0]:
                    cv2.rectangle(frame, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{results.names[int(result[5])]} {result[4]:.2f}', (int(result[0]), int(result[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)  # Save the processed frame to the output video
            cv2.imshow('YOLOv8 Tello Drone Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup()

def cleanup():
    try:
        out.release()  # Ensure 'out' is properly closed
        tello.land()
        tello.streamoff()
    except Exception as e:
        print(f"Cleanup error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
