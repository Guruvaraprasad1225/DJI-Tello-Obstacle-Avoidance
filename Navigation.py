import os
import sys
import time
import cv2
import argparse
from datetime import datetime
from pynput import keyboard
from drone_tracker import DroneTracker

def setup_experiment_directory():
    # Setup a unique directory for each experiment based on the current datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('./experiments', current_time)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file_path = os.path.join(experiment_dir, 'decision_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Experiment started at {}\n".format(current_time))
    return experiment_dir, log_file_path

def log_decision(decision_count, decision=None, distance=None, frame=None, history=None, experiment_dir=None, log_file_path=None):
    decision_count += 1
    if decision:
        log_entry = f"{decision_count}: Decision to move {decision} with distance {distance} at {datetime.now().strftime('%H:%M:%S')}\n"
    with open(log_file_path, 'a') as f:
        f.write(log_entry)
    if isinstance(history, list):
        with open(str(os.path.dirname(log_file_path)) + f'/history{decision_count}.txt', 'w') as f:
            for hist in history:
                f.write(str(hist))

    # Save the current frame with decision
    if frame is not None:
        frame_path = os.path.join(experiment_dir, f"frame_{decision_count}_{decision}_{distance}.jpg")
        cv2.imwrite(frame_path, frame)
    return decision_count

def move_drone(tello, direction, distance, frame, history, tello_connected, decision_count, experiment_dir, log_file_path):
    decision_count = log_decision(decision_count, direction, distance, frame, history, experiment_dir, log_file_path)
    if tello_connected:
        time.sleep(0.25)  # wait for 1.5 seconds
        print("Moving", direction, distance)
        print("\n\n")
        tello.get_current_state()
        if direction == 'right':
            tello.go_xyz_speed(0, 120, 0, 30)
        elif direction == 'left':
            tello.go_xyz_speed(0, -120, 0, 30)
    return decision_count

def on_key_press(key, tello, controls, tello_connected):
    try:
        func = controls.get(key.char, lambda: None)
    except AttributeError:
        func = controls.get(key.name, lambda: None)
    if func and tello_connected:
        func()

def main(args):
    # Initialize decision count, experiment directory, log file
    decision_count = 0
    experiment_dir, log_file_path = setup_experiment_directory()
    
    # Initialize keyboard controls
    controls = {
        'w': lambda: args.tello.move_forward(30),
        'a': lambda: args.tello.move_left(30),
        's': lambda: args.tello.move_back(30),
        'd': lambda: args.tello.move_right(30),
        'i': lambda: args.tello.flip('f'),
        'k': lambda: args.tello.flip('b'),
        'j': lambda: args.tello.flip('l'),
        'l': lambda: args.tello.flip('r'),
        'Key.left': lambda: args.tello.rotate_counter_clockwise(10),
        'Key.right': lambda: args.tello.rotate_clockwise(10),
        'Key.up': lambda: args.tello.move_up(30),
        'Key.down': lambda: args.tello.move_down(30),
        'enter': lambda: args.tello.takeoff(),
        'Key.backspace': lambda: args.tello.land(),
        'c': lambda: args.tello.rotate_clockwise(360),
        'x': lambda: exit()
    }
    
    # Setup keyboard listener
    listener = keyboard.Listener(on_press=lambda key: on_key_press(key, args.tello, controls, args.tello_connected))
    listener.start()  # Start listening to the keyboard

    # Run object detection and avoidance
    while True:
        # Logic to process frames, detect objects, and avoid collisions
        # Example: move_drone(...) with relevant arguments for each frame processed
        pass

    listener.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the drone and track objects using camera.")
    parser.add_argument('--mode', type=str, default='camera', help='tello or camera')
    parser.add_argument('--video_path', type=str, default='video_nav.mp4', help='Path to save the video file')
    parser.add_argument('--model_path', type=str, default='yolov8s.pt', help='Path to the YOLO model file')
    parser.add_argument('--tracker_weights', type=str, default='osnet_x0_25_msmt17.pt', help='Path to the tracker weights')

    args = parser.parse_args()
    args.tello_connected = True  # Assume connection for demonstration
    args.tello = DroneTracker(args.video_path, args.model_path, args.tracker_weights, mode=args.mode)
    
    main(args)
