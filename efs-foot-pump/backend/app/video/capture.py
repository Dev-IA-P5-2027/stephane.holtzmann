import os
import cv2

RTSP_URL = "rtsp://192.168.1.253:554/stream"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def get_capture():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir le flux RTSP avec OpenCV")
        print("✅ VLC marche, donc réseau OK")
        print("➡️ Prochaine option : passer par une URL HTTP/MJPEG si l'app téléphone le propose")
    else:
        print("✅ Flux RTSP connecté avec OpenCV")

    return cap