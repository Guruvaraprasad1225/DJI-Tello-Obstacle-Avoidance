from djitellopy import Tello
import pygame
import time

# Initialize pygame
pygame.init()

# Define screen for pygame (not visible but needed for key detection)
screen = pygame.display.set_mode((400, 300))

# Initialize the Tello drone object
drone = Tello()
drone.connect()

# Check battery level
battery = drone.get_battery()
print(f"Battery level: {battery}%")

# Function to send a command with retries
def send_command_with_retry(command, *args, retries=3):
    for attempt in range(retries):
        try:
            command(*args)
            time.sleep(0.5)  # Add a small delay after each command
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {command.__name__}: {e}")
            if attempt == retries - 1:
                print(f"Max retries reached for {command.__name__}")

# Function to handle key presses for controlling the drone
def handle_keypress(key):
    if key == pygame.K_UP:
        send_command_with_retry(drone.move_forward, 30)
    elif key == pygame.K_DOWN:
        send_command_with_retry(drone.move_back, 30)
    elif key == pygame.K_LEFT:
        send_command_with_retry(drone.move_left, 30)
    elif key == pygame.K_RIGHT:
        send_command_with_retry(drone.move_right, 30)
    elif key == pygame.K_w:
        send_command_with_retry(drone.move_up, 30)
    elif key == pygame.K_s:
        send_command_with_retry(drone.move_down, 30)
    elif key == pygame.K_a:
        send_command_with_retry(drone.rotate_counter_clockwise, 30)
    elif key == pygame.K_d:
        send_command_with_retry(drone.rotate_clockwise, 30)

# Takeoff if enough battery
if battery > 20:
    try:
        drone.takeoff()
        print("Drone has taken off!")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Press 'Q' to land and quit
                        running = False
                        drone.land()
                        print("Drone has landed safely.")
                    else:
                        handle_keypress(event.key)

            time.sleep(0.1)  # Small delay to prevent high CPU usage

    except Exception as e:
        print(f"An error occurred during takeoff or landing: {e}")
        drone.land()
else:
    print("Battery level is too low to fly.")

# Quit pygame
pygame.quit()
