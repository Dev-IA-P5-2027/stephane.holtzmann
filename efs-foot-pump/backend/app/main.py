from app.video.capture import get_capture
from app.detection.pose_detector import PoseDetector
from app.detection.foot_tracker import FootTracker
from app.core.movement_logic import MovementLogic
from app.utils.drawing import draw_text

import cv2


def run():
    cap = get_capture()

    pose_detector = PoseDetector()
    tracker = FootTracker()
    logic = MovementLogic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = pose_detector.process(frame)
        positions = tracker.extract(landmarks, frame)

        movement = logic.compute(positions)

        draw_text(frame, movement)

        cv2.imshow("EFS Foot Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()