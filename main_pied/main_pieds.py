import cv2
import math
import mediapipe as mp
from collections import deque

CAMERA_INDEX = 0
HISTORY = 8  # lissage simple

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def angle_between(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    n1 = math.sqrt(x1 * x1 + y1 * y1)
    n2 = math.sqrt(x2 * x2 + y2 * y2)
    if n1 == 0 or n2 == 0:
        return None
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))

def pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h)), lm.visibility

def ankle_angle(knee_p, ankle_p, heel_p, toe_p):
    leg_vec = (knee_p[0] - ankle_p[0], knee_p[1] - ankle_p[1])
    foot_vec = (toe_p[0] - heel_p[0], toe_p[1] - heel_p[1])
    return angle_between(leg_vec, foot_vec)

def smooth_push(history, value):
    if value is not None:
        history.append(value)
    if not history:
        return None
    return sum(history) / len(history)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Impossible d'ouvrir la caméra {CAMERA_INDEX}")
    raise SystemExit

left_hist = deque(maxlen=HISTORY)
right_hist = deque(maxlen=HISTORY)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(thickness=1)
        )

        lms = res.pose_landmarks.landmark

        # Gauche
        left_knee, vk = pt(lms, mp_pose.PoseLandmark.LEFT_KNEE, w, h)
        left_ankle, va = pt(lms, mp_pose.PoseLandmark.LEFT_ANKLE, w, h)
        left_heel, vh = pt(lms, mp_pose.PoseLandmark.LEFT_HEEL, w, h)
        left_toe, vt = pt(lms, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, w, h)

        # Droite
        right_knee, rvk = pt(lms, mp_pose.PoseLandmark.RIGHT_KNEE, w, h)
        right_ankle, rva = pt(lms, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h)
        right_heel, rvh = pt(lms, mp_pose.PoseLandmark.RIGHT_HEEL, w, h)
        right_toe, rvt = pt(lms, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, w, h)

        if min(vk, va, vh, vt) > 0.5:
            a_left = ankle_angle(left_knee, left_ankle, left_heel, left_toe)
            a_left_s = smooth_push(left_hist, a_left)

            cv2.line(frame, left_knee, left_ankle, (255, 255, 255), 2)
            cv2.line(frame, left_heel, left_toe, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Cheville G: {a_left_s:.1f} deg" if a_left_s is not None else "Cheville G: n/a",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        if min(rvk, rva, rvh, rvt) > 0.5:
            a_right = ankle_angle(right_knee, right_ankle, right_heel, right_toe)
            a_right_s = smooth_push(right_hist, a_right)

            cv2.line(frame, right_knee, right_ankle, (255, 255, 255), 2)
            cv2.line(frame, right_heel, right_toe, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Cheville D: {a_right_s:.1f} deg" if a_right_s is not None else "Cheville D: n/a",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Analyse pompe cheville", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()