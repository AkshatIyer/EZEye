import cv2
import numpy as np
import time

# Load pre-trained classifiers
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Initialize variables
scale_factor = 3.0
eyes_open_start_time = 0
is_server_called = False
scale_factor = 1

def calculate_ear(eye):
    x, y, w, h = eye
    return w / h  # Simple EAR approximation as width/height


def process_frame(frame):
    global eyes_open_start_time, is_server_called

    # Resize frame to improve processing speed
    resized_frame = cv2.resize(frame, (frame.shape[1] // int(scale_factor), frame.shape[0] // int(scale_factor)))

    # Convert to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    ear = 0
    x = 100
    y = 100
    w = 200
    h = 200
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
        cv2.putText(frame, eye_status, eye_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    return frame


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        # Display EAR and Scale Factor


        cv2.imshow("Blink Detection", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()