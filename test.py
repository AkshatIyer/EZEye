import subprocess

import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Load Haarcascade eye detector
eye_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')



# Initialize variables
scale_factor = 1
eyes_open_start_time = 0

def calculate_ear(eye):
    x, y, w, h = eye
    return w / h  # Approximate EAR (Eye Aspect Ratio) using width/height


def process_frame(frame):
    global eyes_open_start_time

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0] // scale_factor))

    # Convert to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Simulated face region for eye detection (Replace with face detector if needed)
    x, y, w, h = 0, 0, 100, 100
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    face_roi = gray[y:y + h, x:x + w]

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_roi)
    for (ex, ey, ew, eh) in eyes:
        eye_top_left = (x + ex, y + ey)
        eye_bottom_right = (x + ex + ew, y + ey + eh)
        cv2.rectangle(frame, eye_top_left, eye_bottom_right, (0, 255, 0), 2)

        # Calculate EAR for blink detection
        ear = calculate_ear((ex, ey, ew, eh))
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        eye_status = "Blink Detected" if ear < 0.2 else "Eyes Open"
        print(eye_status + "\n")
        cv2.putText(frame, eye_status, eye_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)  # Set resolution
    picam2.preview_configuration.main.format = "RGB888"  # Set format
    picam2.configure("preview")
    picam2.start()

    while True:
        frame = picam2.capture_array()  # Capture frame as NumPy array
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for OpenCV

        processed_frame = process_frame(frame)
    #    print("TEST")

        # Display the processed frame
        cv2.imshow("Blink Detection", processed_frame)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
