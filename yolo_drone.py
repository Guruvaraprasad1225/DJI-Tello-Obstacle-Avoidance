from djitellopy import Tello
from pynput import keyboard
from ultralytics import YOLO
import cv2
import time

tello = Tello()
model = YOLO("yolov8l.pt")

def connect_drone():
    tello.connect()
    print(f"batter percentage: {tello.get_battery()}%")
def KeyPress(key):
    try:
        
        if key.char == 'w':
            tello.move_up(100)
        elif key.char =='s':
            tello.move_down(100)
        elif key.char =='a':
            tello.move_left(100)
        elif key.char == 'd':
            tello.move_right(100)
        elif key.char == 'z':
            tello.move_forward(100)
        elif key.char =='e':
            tello.move_back(100)
        elif key.char == 'r':
            tello.rotate_clockwise(100)
        elif key.char =='f':
            tello.rotate_counter_clockwise(100)
    except AttributeError:
        if key == keyboard.Key.enter:
            tello.takeoff()
        elif key == keyboard.Key.space:
            tello.land()
    print(f"KeyPressed=  {Key}")
def release(key):
    if key == keyboard.Key.esc:
        return False
def main():
    connect_drone()
    tello.streamon()

    #VideoAccess = cv2.VideoCapture(0)
    listener = keyboard.Listener(on_press=KeyPress,on_release=release )
    listener.start()
    FrameRead=tello.get_frame_read()
    while True:
        '''ret,frame = VideoAccess.read()
        results = model(frame)
        frame = results[0].plot()
        cv2.imshow("YOLO Webcam", frame)'''
       
        frame= FrameRead.frame
        FrameResized=cv2.resize(frame,(640,480))
        results = model(FrameResized)
        FrameDisplay = results[0].plot()
        cv2.imshow("Tello Cam",FrameDisplay)
        if cv2.waitKey(1) == ord('q'):
            break
    tello.streamoff()
    #VideoAccess.release()
    cv2.destroyAllWindows()
    tello.land()
    tello.end()
    listener.stop()
main()