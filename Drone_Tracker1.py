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
model_path = Path('./yolo8s.pt')
tracker_weights = Path('./osnet_x0_25_msmt17.pt')
mode = 'tello'

# Initialize Tello Drone or Camera
tello = Tello()
tello_connected = False
object_tracking = {}
trajectory_points = {}

def initialize_drone():
    global tello_connected
    try:
        tello.connect()
        tello_connected = True
        print(f"Battery: {tello.get_battery()}%")
        tello.streamon()
    except Exception as e:
        print("Failed to connect to Tello drone:", e)
        tello_connected = False
        if mode == 'tello':
            sys.exit(1)

# Choose a widely compatible codec for the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out_writer = cv2.VideoWriter(video_out, fourcc, 20, (960, 720), isColor=True)

def get_frame():
    frame_source = tello.get_frame_read() if tello_connected else cv2.VideoCapture(0)
    if tello_connected:
        tello.takeoff()
    while True:
        frame = frame_source.frame if tello_connected else frame_source.read()[1]
        if frame is not None:
            yield frame
        else:
            print("No frame received")
            time.sleep(0.1)

def process_frame(frame, model, tracker):
    start_time = time.time()
    results = model(frame, verbose=False)
    dets = [(box.xyxy.tolist(), box.conf.item(), box.cls.item()) for box in results[0].boxes]
    dets = np.array([d[:4] + [d[4], d[5]] for d in dets]) if dets else np.empty((0, 6))

    if dets.size > 0:
        tracker.update(dets, frame)
        tracker.plot_results(frame, show_trajectories=True)
        track_objects(frame, tracker)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Frame processing time: {processing_time:.2f} seconds")

def track_objects(frame, tracker):
    for a in tracker.active_tracks:
        if len(a.history_observations) < 12:
            continue
        box = a.history_observations[-1][:4]
        track_id = a.id
        annotate_frame(box, track_id, frame)
        add_trajectory_point(track_id, (int((box[0] + box[2])/2), int((box[1] + box[3])/2)), frame)

def annotate_frame(box, track_id, frame):
    x1, y1, x2, y2 = map(int, box)
    distance = get_dist(box, frame, track_id)
    speed, ttc = calculate_speed_and_distance(track_id, distance)
    speed_text = f'Speed: {speed:.2f} cm/s | TTC: {ttc:.2f} s'
    cv2.putText(frame, speed_text, ((x1 + x2) // 2, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_dist(rectangle, image, track_id):
    focal = 450
    object_width = 40
    pixels = rectangle[2] - rectangle[0]
    dist = (object_width * focal) / pixels
    dist_text = f'Distance: {dist:.2f} cm'
    cv2.putText(image, dist_text, (rectangle[0], rectangle[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if track_id not in object_tracking:
        object_tracking[track_id] = []
    object_tracking[track_id].append({'time': time.time(), 'distance': dist})

    if len(object_tracking[track_id]) > 25:
        object_tracking[track_id] = object_tracking[track_id][-25:]

    return dist

def calculate_speed_and_distance(track_id, current_distance):
    tracking_info = object_tracking[track_id]
    if len(tracking_info) > 1:
        previous_info = tracking_info[-2]
        current_info = tracking_info[-1]
        t1, d1 = previous_info['time'], previous_info['distance']
        t2, d2 = current_info['time'], current_info['distance']
        speed = (d2 - d1) / (t2 - t1)
        ttc = d2 / speed if speed != 0 else float('inf')
        return abs(speed), ttc
    return 0, float('inf')

def add_trajectory_point(track_id, point, frame):
    trajectory_points.setdefault(track_id, []).append(point)

def cleanup():
    video_out_writer.release()
    if tello_connected or mode == 'tello':
        tello.streamoff()
        tello.land()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tello', help='Operate in tello or camera mode')
    args = parser.parse_args()

    mode = args.mode

    if not initialize_drone():
        print("Exiting: Failed to initialize drone.")
        sys.exit(1)

    # Load model locally without trying to download
    if not model_path.exists() or not tracker_weights.exists():
        print("Model files are missing, please check the paths.")
        sys.exit(1)
    model = YOLO(model_path, task='detect')
    tracker = DeepOCSORT(model_weights=tracker_weights, device='cpu', fp16=False)

    try:
        for frame in get_frame():
            process_frame(frame, model, tracker)
            video_out_writer.write(frame)
            cv2.imshow("Tello Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f'Error during operation: {e}')
    finally:
        cleanup()
