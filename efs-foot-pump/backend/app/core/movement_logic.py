class MovementLogic:
    def __init__(self):
        self.prev_left = None
        self.prev_right = None
        self.threshold = 10

    def compute(self, positions):
        if positions is None:
            return "NONE"

        movement = "NONE"

        left = positions["left_y"]
        right = positions["right_y"]

        if self.prev_left is not None:
            if abs(left - self.prev_left) > self.threshold:
                movement = "LEFT"

        if self.prev_right is not None:
            if abs(right - self.prev_right) > self.threshold:
                movement = "RIGHT"

        self.prev_left = left
        self.prev_right = right

        return movement