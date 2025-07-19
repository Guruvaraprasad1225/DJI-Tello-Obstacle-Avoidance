from djitellopy import Tello
from pynput import keyboard
import time

# Initialize Tello drone
tello = Tello()

def connect_drone():
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

def on_press(key):
    try:
        if key.char == 'w':
            tello.move_forward(30)
        elif key.char == 's':
            tello.move_back(30)
        elif key.char == 'a':
            tello.move_left(30)
        elif key.char == 'd':
            tello.move_right(30)
        elif key.char == 'q':
            tello.rotate_counter_clockwise(30)
        elif key.char == 'e':
            tello.rotate_clockwise(30)
        elif key.char == 'r':
            tello.move_up(30)
        elif key.char == 'f':
            tello.move_down(30)
    except AttributeError:
        if key == keyboard.Key.space:
            tello.land()
        elif key == keyboard.Key.enter:
            tello.takeoff()
    
    print(f"Key {key} pressed")

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def main():
    # Connect to the drone
    connect_drone()
    
    # Keyboard listener for controlling the drone
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
        listener.join()

    # Safely land the drone after exiting
    tello.land()
    tello.end()
main()
