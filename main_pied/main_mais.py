import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    raise SystemExit

start_time = time.monotonic()
last_timestamp_ms = -1

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire la frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                points = []

                for i, lm in enumerate(hand_landmarks):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        str(i),
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )

                for start_idx, end_idx in HAND_CONNECTIONS:
                    x1, y1 = points[start_idx]
                    x2, y2 = points[end_idx]
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                index_tip = hand_landmarks[8]
                ix = int(index_tip.x * w)
                iy = int(index_tip.y * h)

                cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    "INDEX",
                    (ix + 10, iy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        cv2.putText(
            frame,
            "Press Q or ESC to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Hand Tracking Skeleton", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print("Fermeture demandee.")
            break

cap.release()
cv2.destroyAllWindows()