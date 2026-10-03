import cv2
import numpy as np


def depth_to_disparity(
    depth,
    maximum_disparity=20,
    strength=1.0,
    invert=False
):
    depth = depth.astype(np.float32)

    depth = cv2.normalize(
        depth,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX
    )

    if invert:
        depth = 1.0 - depth

    disparity = depth * maximum_disparity * strength

    return disparity


def create_stereo_pair(
    frame,
    disparity
):

    height, width = frame.shape[:2]

    x, y = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Left eye
    map_left_x = x + disparity
    map_left_y = y

    # Right eye
    map_right_x = x - disparity
    map_right_y = y

    left = cv2.remap(
        frame,
        map_left_x,
        map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    right = cv2.remap(
        frame,
        map_right_x,
        map_right_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return left, right