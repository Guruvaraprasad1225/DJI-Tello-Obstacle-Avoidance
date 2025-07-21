import os
import argparse
import sys
import cv2
from ultralytics import YOLO
from djitellopy import Tello
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT
from datetime import datetime
from pynput import keyboard
import time
import math
import keyboard
import threading

# Global variables
video_out = 'video.mp4'
model_path = ' ./yolo8s.pt'
tracker_weights = './osnet_x0_25_msmt17.pt'
mode = 'tello'
frame = None
object_tracking = {}
trajectory_points = {}
tello_connected = (mode == 'tello')
listener = None
decision_count = 0
experiment_dir = None
log_file_path = ''
#controls = None

# Initialize Tello Drone or Camera
tello = Tello()
tello_connected = False

#Initialize Drone Connection
def initialize_drone():
    global tello_connected
    
    try:
        tello.connect()
        tello_connected = True
        print(f"battery: {tello.get_battery()}%")
        tello.streamon()
        time.sleep(2)
        #tello.takeoff()

    except Exception as e: 
        print("Failed to connect to Tello drone. Falling back to local camera.",e)
        tello_connected = False
        attempt +=1
        time.sleep(5)
    if not tello_connected and mode == 'tello':
            print("Failed to connect to Tello drone after several attempts.")
            sys.exit(1)

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# (640, 480)
video_out_writer = cv2.VideoWriter(video_out, fourcc, 20, (960, 720), isColor=True)

# Get Frame from Tello or camera
def get_frame():
    #frame_source = tello.get_frame_read(with_queue = False, max_queue_len=0) if tello_connected else cv2.VideoCapture(0)
    frame_source = tello.get_frame_read() if tello_connected else cv2.VideoCapture(0)
    while True:
        process_start_time = time.time()
        frame = frame_source.frame if tello_connected else frame_source.read()[1]
        #cv2.imshow('video',frame)
        if tello_connected:
            frame = frame_source.frame
        else:
            success, frame = frame_source.read()
            if not success:
                print("Failed to capture frame from camera.")
                break
        if frame is not None:
            # print(frame)
            yield frame,process_start_time
        else:
            print("No frame Received")
            frame_source.release()
            time.sleep(2)  # Wait a bit before retrying
            frame_source = tello.get_frame_read() if tello_connected else cv2.VideoCapture(0)

#Log Setup
def setup_experiment_directory():
    global log_file_path,experiment_dir
    # Setup a unique directory for each experiment based on the current datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('./experiments', current_time)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file_path = os.path.join(experiment_dir, 'decision_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Experiment started at {}\n".format(current_time))
    return experiment_dir

def log_decision(decision=None, distance=None, frame=None, history=None):
    global decision_count, log_file_path, experiment_dir
    if log_file_path is None:
        print("Error: log_file_path is not set.")
        return
    
    decision_count += 1
    log_entry = f"{decision_count}: Decision to move {decision} with distance {distance} at {datetime.now().strftime('%H:%M:%S')}\n"
    with open(log_file_path, 'a') as f:
        f.write(log_entry)
    if isinstance(history, list):
        with open(str(os.path.dirname(log_file_path)) + '/history{}.txt'.format(decision_count), 'w') as f:
            for hist in history:
                f.write(str(hist))


    # Save the current frame with decision
    if frame is not None and experiment_dir is not None:
        frame_path = os.path.join(experiment_dir, f"frame_{decision_count}_{decision}_{distance}.jpg")
        cv2.imwrite(frame_path, frame)

def avoid_objects():
        # Enhanced object avoidance logic using TTC (Time-To-Collision) to make decisions
        for a in tracker.active_tracks:
            # Check if there is tracking info for the current object
            if a.id in object_tracking:
                tracking_info = object_tracking[a.id]
                # Ensure there is enough data to make an informed decision
                if not tracking_info or len(tracking_info) < 18:
                    continue
                
                last_info = tracking_info[-1]
                ttc = last_info['ttc']
                box = last_info['bbox']
                x_center = (box[2] + box[0]) / 2  # Calculate the horizontal center of the object
                frame_center_x = frame.shape[1] / 2  # Middle of the frame

                # Act if the object is projected to collide within 5 seconds
                if ttc < 3:  
                    # save tracking info for future use

                    print(f"Object close to collision. TTC: {ttc:.2f}s")
                    # Consider an object for avoidance if it is within 10% of the frame center horizontally
                    if abs(x_center - frame_center_x) < frame.shape[1] * 0.3:
                        if x_center < frame_center_x:
                            # Object is on the left, maneuver the drone to the right
                            print("Object on left, moving right")
                            move_drone('right', 80,tracking_info[-10:],frame)
                        else:
                            # Object is on the right, maneuver the drone to the left
                            print("Object on right, moving left")
                            move_drone('left', 80,tracking_info[-10:],frame)
                    else:
                        # Object not close enough to the center to be an immediate threat, adjust strategy accordingly
                        # This may include slowing down or slightly adjusting the path without a sharp turn
                        if x_center < frame_center_x:
                            # If object is to the left but not dangerously close, consider a mild right adjustment
                            print("Object slightly on left, mild adjustment to the right")
                            move_drone('right', 40,tracking_info[-10:],frame) # smaller movement than the urgent avoidance
                        else:
                            # If object is to the right but not dangerously close, consider a mild left adjustment
                            print("Object slightly on right, mild adjustment to the left")
                            move_drone('left', 40,tracking_info[-10:],frame)# smaller movement than the urgent avoidance
                            

# Drone Movement
def move_drone(direction, distance,history=None,frame=None):
    log_decision(direction,distance, frame,history)
    if tello_connected:
        # getattr(tello, f"move_{direction}")(distance)
        # go_xyz_speed
        time.sleep(0.25) # wait for 1.5 seconds
        print("Moving", direction, distance)
        print("\n\n")
        tello.get_current_state()
        if direction == 'right':
            # tello.go_xyz_speed(0,120, 0, 30)
            tello.move_right(distance)
        elif direction == 'left':
            #tello.go_xyz_speed(0,-120, 0, 30)
            tello.move_left(distance)

#Process Frame for Detection
def process_frame(frame,process_start_time,model,tracker):
    # Call the process frame from DroneTracker
    avoid_objects()
    start_time = time.time()
    results = model(frame, verbose=False)
    # dets = [(box.xyxy.tolist()[0], box.conf.item(), box.cls.item()) for box in results[0].boxes]
    dets = [(*box.xyxy.tolist()[0], box.conf.item(), box.cls.item()) for box in results[0].boxes]
    # print('1',dets)
    dets = np.array(dets)
    # print('2',dets)

    print(f"dets shape: {dets.shape}")

    if len(dets.shape) ==2:
        tracker.update(dets,frame)
        tracker.plot_results(frame,show_trajectories = True)
        #frame = draw_updated_boxes(frame, tracker.active_tracks)
        track_objects(frame,tracker)
    
    end_time = time.time()
    process_end_time = time.time()
    processing_time = end_time-start_time
    lag_time = process_end_time - process_start_time
    print(f"Frame processing time: {processing_time:.2f}seconds")
    print(f"Total lag time (including detection + tracking): {lag_time:.2f} seconds")
    return frame  # Returning the processed frame for further use

#Track Objects in the Frame
def track_objects(frame,tracker):
    active_tracks = tracker.active_tracks
    for a in tracker.active_tracks:
        if len(a.history_observations) < 12:
            continue
        box = a.history_observations[-1][:4]    
        track_id = a.id
        print(track_id)
        annotate_frame(box, track_id,frame)
        add_trajectory_point(track_id, (int((box[0] + box[2])/2),int ((box[1]+box[3])/2)),frame)

#Track Objects in the Frame
def track_objects(frame,tracker):
    active_tracks = tracker.active_tracks
    for a in tracker.active_tracks:
        if len(a.history_observations) < 12:
            continue
        box = a.history_observations[-1][:4]    
        track_id = a.id
        print(track_id)
        annotate_frame(box, track_id,frame)
        add_trajectory_point(track_id, (int((box[0] + box[2])/2),int ((box[1]+box[3])/2)),frame)

# Annotate Frame with Distance and Speed
def annotate_frame(box,track_id,frame):
    x1,y1,x2,y2 = map(int,box)
    distance = get_dist(box,frame,track_id)
    speed, ttc = calculate_speed_and_distance(track_id,distance)
    speed_text = f'Speed: {speed:.2f} cm/s | TTC: {ttc:.2f} s'
    cv2.putText(frame,speed_text, ((x1+x1)//2,y2-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

#Get Distance from Bounding Box
def get_dist(rectangle,image,track_id):
    focal = 450
    object_width = 40
    pixels =rectangle[2]-rectangle[0]
    dist = (object_width*focal)/pixels
    dist_text =f'Distance:{dist:.2f}cm'
    cv2.putText(image, dist_text, (int(rectangle[0]), int(rectangle[1] + 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    if track_id not in object_tracking:
        object_tracking[track_id]=[]
    object_tracking[track_id].append({
        'time':time.time(),
        'bbox':rectangle,
        'distance': dist,
        'speed': None,
        'ttc':None
    })

    if len (object_tracking[track_id])>25:
        object_tracking[track_id] = object_tracking[track_id][-25:]

    return dist

#Calculate Speed and Distance to Collision
def calculate_speed_and_distance(track_id, current_distance):
    tracking_info = object_tracking[track_id]
    if len(tracking_info)>1:
        previous_info = tracking_info[-2]
        current_info = tracking_info[-1]
        t1,d1 = previous_info['time'],previous_info['distance']
        t2,d2 = current_info['time'],current_info['distance']
        speed = -1*(d2-d1)/(t2-t1)
        ttc = d2/speed if speed>0 else float ('inf')
        current_info['speed'] =speed
        current_info['ttc'] =ttc
        return speed,ttc
    return 0, float('inf')
    
# def draw_updated_boxes(frame, tracks):
#     for track in tracks:
#         box = track.latest_box  # Update this based on how your tracking library provides the latest box
#         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#     return frame

#Draw Trajectory Points
trajectory_points ={}

def add_trajectory_point(track_id,point,frame):
    if track_id not in trajectory_points:
        trajectory_points[track_id] = []
    trajectory_points[track_id].append(point)
    drawPoints(frame,trajectory_points[track_id])

def drawPoints(img,points):
    for point in points:
        cv2.circle(img,point,5,(0,0,255),-1)
        x_meters = (points[-1][0] - 500)/100
        y_meters = (points[-1][1] - 500)/100
        text = f'({x_meters:.2f}m,{y_meters:.2f}m)'
        cv2.putText(img,text,(points[-1][0]+10,points[-1][1]+30),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)

# Function to send keep-alive commands
def keep_alive(drone):
    while True:
        time.sleep(10)  # Send command every 10 seconds
        try:
            drone.send_control_command("command")
            print("Keep-alive command sent.")
        except Exception as e:
            print("Error sending keep-alive command:", e)

#Cleanup Function
def cleanup():
    video_out_writer.release()
    if tello_connected or mode =='tello':
        try:
            tello.streamoff()
            tello.land()
        except Exception as e:
            print(f"Error stopping video stream:{e}")
    cv2.destroyAllWindows()

    #Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the drone and track objects using camera.")
    parser.add_argument('--mode',type=str, default='tello', help='Choose "tello" for drone or "camera" for webcam.')
    parser.add_argument('--video', type=str, default=video_out,help='video file path')
    args = parser.parse_args()

    mode = args.mode
    video_out = args.video

    if mode == 'tello':
        initialize_drone()

        #start keep-alive threading
        keep_alive_thread = threading.Thread(target=keep_alive, args=(tello,))
        keep_alive_thread.daemon = True # Ensure the thread stops when the program exits
        keep_alive_thread.start()

    model=YOLO(r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt",task='detect')
    tracker = DeepOCSORT(
        model_weights= Path(tracker_weights),
        device = 'cpu',
        fp16=False,
     # Instantiate the NavigatorFromTracker with the parsed arguments
    # navigator = NavigatorFromTracker(video_path, args.model_path, args.tracker_weights,mode= args.mode)
    # navigator.run()
    )
    tello.takeoff()
    setup_experiment_directory()
    count = 0
    try:
        for frame,start_time in get_frame():
            #Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            count+=1
            #process_frame(frame,start_time,model,tracker)
            processed_frame = process_frame(frame, start_time, model, tracker)
            video_out_writer.write(processed_frame)
            cv2.imshow(f"YOLOv8 Tello Drone Tracking",processed_frame)#(f"YOLOv8 Tello Drone Tracking {count}",processed_frame)
            # if keyboard.is_pressed('x'):  # Use keyboard.is_pressed to check for 'x' key
            #     print("Emergency landing initiated.")
            #     tello.land()
            #     break

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
    except Exception as e:
        print(f'Error during running: {e}')
    finally:
        cleanup()