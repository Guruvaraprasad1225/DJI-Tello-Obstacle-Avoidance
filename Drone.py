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

#Initialize Tello Drone or Camera
tello = Tello()
tello_connected = False
object_tracking = {}

# Initialize drone connection
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
        if mode == 'tello':
            sys.exit(1)

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# (640, 480)
video_out_writer = cv2.VideoWriter(video_out, fourcc, 20, (960, 720), isColor=True)
# video_out_writer.set(cv2.CAP_PROP_FPS, 15)  # Reduce to 15 FPS
# Get Frame from Tello or camera
def get_frame():
    frame_source = tello.get_frame_read(with_queue = False, max_queue_len=0) if tello_connected else cv2.VideoCapture(0)
    '''if tello_connected:
        print('AA')
        #tello.takeoff()'''
    while True:
        process_start_time = time.time()
        frame = frame_source.frame if tello_connected else frame_source.read()[1]
        #cv2.imshow('video',frame)
        if frame is not None:
            # print(frame)
            yield frame,process_start_time
        else:
            print("No frame Received")


#Process Frame for Detection
def process_frame(frame,process_start_time,model,tracker):
    start_time = time.time()
    results = model(frame, verbose=False)
    # dets = [(box.xyxy.tolist()[0], box.conf.item(), box.cls.item()) for box in results[0].boxes]
    dets = [(*box.xyxy.tolist()[0], box.conf.item(), box.cls.item()) for box in results[0].boxes]
    print('1',dets)
    dets = np.array(dets)
    print('2',dets)

    print(f"dets shape: {dets.shape}")

    if len(dets.shape) ==2:
        tracker.update(dets,frame)
        tracker.plot_results(frame,show_trajectories = True)
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

#Cleanup Function
def cleanup():
    video_out_writer.release()
    if tello_connected or mode =='tello':
        try:
            tello.streamoff()
        except Exception as e:
            print(f"Error stopping video stream:{e}")
    cv2.destroyAllWindows()

#Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str, default='tello', help='tello or camera')
    parser.add_argument('--video', type=str, default=video_out,help='video file path')
    args = parser.parse_args()

    mode = args.mode
    video_out = args.video

    initialize_drone()

    model=YOLO(r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt",task='detect')
    tracker = DeepOCSORT(
        model_weights= Path(tracker_weights),
        device = 'cpu',
        fp16=False,
    )
    count = 0
    try:
        for frame,start_time in get_frame():
            #Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            count+=1
            #process_frame(frame,start_time,model,tracker)
            processed_frame = process_frame(frame, start_time, model, tracker)
            video_out_writer.write(processed_frame)
            cv2.imshow(f"YOLOv8 Tello Drone Tracking {count}",processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            if mode == 'tello' and not tello_connected:
                break
            if count>=30:
                break
    except Exception as e:
        print(f'Error during running: {e}')
    finally:
        cleanup()