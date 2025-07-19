from djitellopy import Tello
import time    
import keyboard

tello = Tello()
# video_out='video.mp4'
# model_path = r"D:\CSU\#2Fall\Graduate Project\Project Code\yolov8s.pt" 
try:
    tello.connect()
    print(f"Battery:{tello.get_battery()}%")
    #tello.stream_on()
    print("Drone is streaming")
    tello.takeoff()
    print("Drone is flying.")
    
    while True:
        if keyboard.is_pressed('x'):
            print("Landing")
            tello.land()
        else:
            tello.send_rc_control(0,0,0,0)
            time.sleep(2)
except Exception as e:
    print(F"Error:{e}")
finally:
    if tello.is_flying:  # Check if the drone is still flying
        tello.land()
    print("Drone Landed.")