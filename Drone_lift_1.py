from djitellopy import Tello
import threading
import time
import sys

# Function to send keep-alive commands to prevent auto-landing
def keep_alive(drone):
    while True:
        time.sleep(10)  # Send command every 10 seconds
        try:
            drone.send_control_command("command")  # Keep-alive command
        except Exception as e:
            print("Error sending keep-alive command:", e)

# Initialize Drone Connection
def initialize_drone():
    tello = Tello()
    try:
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")
        tello.takeoff()
        print("Drone has taken off!")
    except Exception as e:
        print("Failed to connect to Tello drone.", e)
        sys.exit(1)

    # Start keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive, args=(tello,))
    keep_alive_thread.daemon = True  # Ensure thread stops when program exits
    keep_alive_thread.start()

    return tello

# Control Drone Movement
def control_drone(drone):
    print("Control keys: w (forward), s (backward), a (left), d (right),")
    print("              h (rotate left), n (rotate right), j (down), u (up),")
    print("              l (land), x (exit program).")

    while True:
        key = input("Enter a control key: ").strip().lower()

        # Movement Controls
        try:
            if key == 'w':  # Forward
                print("Moving forward...")
                drone.move_forward(30)  # Forward 30 cm
            elif key == 's':  # Backward
                print("Moving backward...")
                drone.move_back(30)  # Backward 30 cm
            elif key == 'a':  # Left
                print("Moving left...")
                drone.move_left(30)  # Left 30 cm
            elif key == 'd':  # Right
                print("Moving right...")
                drone.move_right(30)  # Right 30 cm
            elif key == 'j':  # Down
                print("Moving down...")
                drone.move_down(30)  # Down 30 cm
            elif key == 'u':  # Up
                print("Moving up...")
                drone.move_up(30)  # Up 30 cm
            elif key == 'h':  # Rotate left
                print("Rotating left...")
                drone.rotate_counter_clockwise(30)  # Rotate 30 degrees left
            elif key == 'n':  # Rotate right
                print("Rotating right...")
                drone.rotate_clockwise(30)  # Rotate 30 degrees right

            # Land the drone
            elif key == 'l':
                print("Landing the drone...")
                drone.land()
                break

            # Exit Program
            elif key == 'x':
                print("Exiting program...")
                break

            else:
                print("Invalid key. Use 'w', 's', 'a', 'd', 'h', 'n', 'j', 'u', or 'l'.")
        except Exception as e:
            print(f"Error executing command: {e}")

# Main Function
if __name__ == "__main__":
    drone = initialize_drone()  # Initialize and store the Tello instance

    try:
        # Control the drone based on user input
        control_drone(drone)
    except KeyboardInterrupt:
        print("Program interrupted manually.")
    finally:
        # Land the drone if the program exits unexpectedly
        if drone.is_flying:
            drone.land()
            print("Drone landed and program exited cleanly.")
