class FootTracker:
    def extract(self, landmarks, frame):
        if landmarks is None:
            return None

        h, w, _ = frame.shape

        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        return {
            "left_y": int(left_ankle.y * h),
            "right_y": int(right_ankle.y * h)
        }