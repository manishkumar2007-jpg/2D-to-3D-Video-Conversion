import numpy as np


class TemporalSmoother:

    def __init__(self, alpha=0.7):

        self.alpha = alpha
        self.previous = None

    def apply(self, current):

        current = current.astype(
            np.float32
        )

        if self.previous is None:

            self.previous = current.copy()

            return current

        smoothed = (
            self.alpha * self.previous
            + (1.0 - self.alpha) * current
        )

        self.previous = smoothed.copy()

        return smoothed

    def reset(self):

        self.previous = None