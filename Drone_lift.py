from djitellopy import Tello
import sys

# Initialize Drone Connection
def initialize_drone():
    tello = Tello()
    try:
        # Connect to the Tello drone
        tello.connect()
        print(f"Battery: {tello.get_battery()}%")
        
        # Take off
        tello.takeoff()
        print("Drone has taken off!")
    except Exception as e:
        print("Failed to connect to Tello drone.", e)
        sys.exit(1)
    return tello

# Control Drone Movement
def control_drone(drone):
    print("Control keys: w (forward), s (backward), a (left), d (right), u (up), j (down), h (rotate CW), k (rotate CCW), x (quit)")

    while True:
        key = input("Enter a control key: ").strip().lower()

        if key == 'w':  # Move forward
            print("Moving forward...")
            drone.go_xyz_speed(30, 0, 0, 30)  # Forward 30 cm
        elif key == 's':  # Move backward
            print("Moving backward...")
            drone.go_xyz_speed(-30, 0, 0, 30)  # Backward 30 cm
        elif key == 'a':  # Move left
            print("Moving left...")
            drone.go_xyz_speed(0, -30, 0, 30)  # Left 30 cm
        elif key == 'd':  # Move right
            print("Moving right...")
            drone.go_xyz_speed(0, 30, 0, 30)  # Right 30 cm
        elif key == 'u':  # Move up
            print("Moving up...")
            drone.go_xyz_speed(0, 0, 30, 30)  # Up 30 cm
        elif key == 'j':  # Move down
            print("Moving down...")
            drone.go_xyz_speed(0, 0, -30, 30)  # Down 30 cm
        elif key == 'h':  # Rotate clockwise
            print("Rotating clockwise 30 degrees...")
            drone.rotate_clockwise(30)  # Rotate CW 30 degrees
        elif key == 'k':  # Rotate counter-clockwise
            print("Rotating counter-clockwise 30 degrees...")
            drone.rotate_counter_clockwise(30)  # Rotate CCW 30 degrees
        elif key == 'x':  # Exit
            print("Exiting program...")
            break
        else:
            print("Invalid key. Use 'w', 's', 'a', 'd', 'u', 'j', 'h', 'k', or 'x'.")

# Main Function
if __name__ == "__main__":
    drone = initialize_drone()  # Initialize and store the Tello instance

    try:
        # Control the drone based on user input
        control_drone(drone)
    except KeyboardInterrupt:
        print("Program interrupted manually.")
    finally:
        # Land the drone and clean up
        drone.land()
        print("Drone landed and program exited cleanly.")
