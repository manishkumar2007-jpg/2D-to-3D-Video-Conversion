import cv2
import numpy as np


def create_anaglyph(left, right):

    left_gray = cv2.cvtColor(
        left,
        cv2.COLOR_BGR2GRAY
    )

    right_gray = cv2.cvtColor(
        right,
        cv2.COLOR_BGR2GRAY
    )

    output = np.zeros_like(left)

    # OpenCV uses BGR
    output[:, :, 2] = left_gray       # Red
    output[:, :, 1] = right_gray      # Green
    output[:, :, 0] = right_gray      # Blue

    return output