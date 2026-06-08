import cv2

def draw_text(frame, text):
    cv2.putText(
        frame,
        text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )