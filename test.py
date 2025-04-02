import subprocess

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import math
import time
import board
import busio
import adafruit_adxl34x

# Initialize I2C connection
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize ADXL345
accelerometer = adafruit_adxl34x.ADXL345(i2c)

# Optional: Set the range (e.g., ±2g, ±4g, ±8g, ±16g)
accelerometer.range = adafruit_adxl34x.Range.RANGE_2_G
counter=0
dispense_allowed = False  # Flag to control dispensing
blink_detected = False

#converting acceleration to angles in degrees
def get_tilt_angles(x, y, z):
    roll = math.atan2(y, math.sqrt(x**2 + z**2)) * (180 / math.pi)
    pitch = math.atan2(x, math.sqrt(y**2 + z**2)) * (180 / math.pi)
    return roll, pitch

# Load Haarcascade eye detector
eye_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

eyes_open_start = None  # Track when the eyes first open
eyes_closed_start = None  # Track when the eyes disappear
# Initialize variables
scale_factor = 3
eyes_open_start_time = 0
open_duration = 3  # Time in seconds required for eyes open to trigger motor
blink_threshold = 0.1  # Time in seconds for eyes to be considered closed

def calculate_ear(eye):
    x, y, w, h = eye
    return w / h  # Approximate EAR (Eye Aspect Ratio) using width/height


def process_frame(frame):
    global eyes_open_start, eyes_closed_start, blink_detected
    # blink_detected = False
    dispense = False
    # Convert to grayscale
    small_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in eyes:
        x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
    if len(eyes) > 0:  # If eyes are detected
        blink_detected = False
        if eyes_open_start is None:
            eyes_open_start = time.time()  # Start tracking eye-open time
        if eyes_closed_start is not None:
            eyes_closed_start = None  # Reset closed timer when eyes are detected

        # Check if eyes have been open for the required duration
        if time.time() - eyes_open_start >= open_duration:
            dispense = True
            eyes_open_start = None  # Reset timer after activation

    else:  # If no eyes are detected
        if eyes_closed_start is None:
            eyes_closed_start = time.time()  # Start tracking closed time

        # If eyes have been closed for too long, reset open timer
        if time.time() - eyes_closed_start >= blink_threshold:
            print("Blink detected (eyes disappeared too long)")
            blink_detected = True
            eyes_open_start = None  # Reset open timer

    return frame, dispense

def main():
    # Initialize PiCamera2
    global scale_factor, dispense_allowed, blink_detected

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)  # Set resolution
    picam2.preview_configuration.main.format = "RGB888"  # Set format
    picam2.configure("preview")
    picam2.start()
    # picam2 = cv2.VideoCapture(0)
    while True:
        # frame = picam2.capture_array()  # Capture frame as NumPy array
        ret, frame = picam2.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for OpenCV

        processed_frame, dispense = process_frame(frame)
        x, y, z = accelerometer.acceleration
        roll, pitch = get_tilt_angles(x, y, z)

        roll = 0
        if 45 <= abs(roll) <= 180:  # or 45 <= abs(pitch) <= 180:
            print("correct angle")
            if dispense and dispense_allowed:
                subprocess.run(['python', 'stepper_motor.py'])
                dispense = False
    #    print("TEST")

        # Display the processed frame
        cv2.putText(processed_frame, f"Scale Factor: {scale_factor}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                   2)
        if blink_detected:
            cv2.putText(processed_frame, "Blink Detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Blink Detection", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        # Adjust scale factor using keyboard input
        if key == ord('+') and scale_factor < 10:
            scale_factor += 1
        elif key == ord('-') and scale_factor > 1:
            scale_factor -= 1
        elif key == ord('r'):
            dispense_allowed = True  # Reset dispense flag
            print("Dispense reset")
        elif key == ord('q'):
            break


cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
