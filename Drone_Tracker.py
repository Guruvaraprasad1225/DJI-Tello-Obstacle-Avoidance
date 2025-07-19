import argparse
import sys
import cv2
from ultralytics import YOLO
from djitellopy import Tello
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT
import time
import math

# Configuration
video_out = 'video.mp4'
model_path = './yolo8s.pt'
tracker_weights = './osnet_x0_25_msmt17.pt'
mode = 'camera'

# Initialize Tello Drone or Camera
tello = Tello()
tello_connected = False
object_tracking = {}

# Initialize drone connection
def initialize_drone():
    global tello_connected
    try:
        tello.connect()
        battery_level = tello.get_battery()
        print(f"Battery: {battery_level}%")
        
        # Only proceed if battery level is above a threshold, e.g., 10%
        if battery_level < 10:
            print("Battery too low to initiate flight.")
            tello_connected = False
            if mode == 'tello':
                sys.exit("Exiting due to low battery.")
            return
        
        # Start streaming video and take off after a brief pause
        tello.streamon()
        time.sleep(2)
        #tello.takeoff()
        tello_connected = True
        print("Tello drone initialized and ready.")
        
    except Exception as e:
        print("Failed to connect to Tello drone. Falling back to local camera.", e)
        tello_connected = False
        if mode == 'tello':
            sys.exit(1)

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out_writer = cv2.VideoWriter(video_out, fourcc, 20, (640, 480), isColor=True)
video_out_writer.set(cv2.CAP_PROP_FPS, 15)  # Reduce to 15 FPS

# Get Frame from Tello or camera
def get_frame():
    frame_source = tello.get_frame_read(with_queue=False, max_queue_len=0) if tello_connected else cv2.VideoCapture(0)
    while True:
        process_start_time = time.time()
        frame = frame_source.frame if tello_connected else frame_source.read()[1]
        if frame is not None:
            yield frame, process_start_time
        else:
            print("No frame Received")

# Keyboard Input to Control the Drone
def control_drone():
    key = cv2.waitKey(1) & 0xFF  # Wait for key press
    if key == 27:  # ESC to exit
        return 'exit'
    elif key == 81:  # Left Arrow
        tello.send_rc_control(-20, 0, 0, 0)  # Move left
    elif key == 83:  # Right Arrow
        tello.send_rc_control(20, 0, 0, 0)  # Move right
    elif key == 82:  # Up Arrow
        tello.send_rc_control(0, 20, 0, 0)  # Move forward
    elif key == 84:  # Down Arrow
        tello.send_rc_control(0, -20, 0, 0)  # Move backward
    elif key == ord('x'):  # 'x' to land or exit
        tello.send_rc_control(0, 0, 0, 0)  # Stop movement
        return 'exit'
    return None

# Process Frame for Detection
def process_frame(frame, process_start_time, model, tracker):
    start_time = time.time()
    results = model(frame, verbose=False)
    dets = [(*box.xyxy.tolist()[0], box.conf.item(), box.cls.item()) for box in results[0].boxes]
    dets = np.array(dets)

    if len(dets.shape) == 2:
        tracker.update(dets, frame)
        tracker.plot_results(frame, show_trajectories=True)
        track_objects(frame, tracker)

    end_time = time.time()
    processing_time = end_time - start_time
    return frame

# Track Objects in the Frame
def track_objects(frame, tracker):
    active_tracks = tracker.active_tracks
    for a in tracker.active_tracks:
        if len(a.history_observations) < 12:
            continue
        box = a.history_observations[-1][:4]
        track_id = a.id
        annotate_frame(box, track_id, frame)
        add_trajectory_point(track_id, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), frame)

# Annotate Frame with Distance and Speed
def annotate_frame(box, track_id, frame):
    x1, y1, x2, y2 = map(int, box)
    distance = get_dist(box, frame, track_id)
    speed, ttc = calculate_speed_and_distance(track_id, distance)
    speed_text = f'Speed: {speed:.2f} cm/s | TTC: {ttc:.2f} s'
    cv2.putText(frame, speed_text, ((x1 + x1) // 2, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Get Distance from Bounding Box
def get_dist(rectangle, image, track_id):
    focal = 4
    object_width = 40
    pixels = rectangle[2] - rectangle[0]
    dist = (object_width * focal) / pixels
    dist_text = f'Distance:{dist:.2f}cm'
    cv2.putText(image, dist_text, (int(rectangle[0]), int(rectangle[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return dist

# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tello', help='tello or camera')
    parser.add_argument('--video', type=str, default=video_out, help='video file path')
    args = parser.parse_args()

    mode = args.mode
    video_out = args.video

    initialize_drone()

    model = YOLO(r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt", task='detect')
    tracker = DeepOCSORT(
        model_weights=Path(tracker_weights),
        device='cpu',
        fp16=False,
    )

    try:
        for frame, start_time in get_frame():
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and handle keyboard control
            processed_frame = process_frame(frame, start_time, model, tracker)

            # Capture keyboard input for drone control
            exit_command = control_drone()
            if exit_command == 'exit':
                break

            # Write the processed frame to video and display
            video_out_writer.write(processed_frame)
            cv2.imshow("YOLOv8 Tello Drone Tracking", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            if mode == 'tello' and not tello_connected:
                break
    except Exception as e:
        print(f'Error during running: {e}')
    # finally:
    #     cleanup()
