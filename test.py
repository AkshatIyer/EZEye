import subprocess
import cv2
import numpy as np
from picamera2 import Picamera2

# Start FFmpeg Process for RTSP Streaming
def open_ffmpeg_stream_process():
    args = (
        "ffmpeg -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i pipe:0 "
        "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p "
        "-f rtsp rtsp://192.168.1.100:8554/stream"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

ffmpeg_process = open_ffmpeg_stream_process()

# Load Haarcascade eye detector
eye_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

def calculate_ear(eye):
    x, y, w, h = eye
    return w / h  # Approximate EAR (Eye Aspect Ratio)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        ear = calculate_ear((ex, ey, ew, eh))
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

def main():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)  # Set resolution
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        processed_frame = process_frame(frame)

        try:
            ffmpeg_process.stdin.write(processed_frame.astype(np.uint8).tobytes())
            ffmpeg_process.stdin.flush()
        except BrokenPipeError:
            print("FFmpeg stream closed.")
            break

if __name__ == "__main__":
    main()
