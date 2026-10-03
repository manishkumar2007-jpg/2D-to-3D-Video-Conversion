import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class DepthEstimator:

    def __init__(self):

        self.device = 0 if torch.cuda.is_available() else -1

        print("Loading depth model...")

        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device
        )

        print("Depth model loaded.")

    def estimate(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        result = self.pipe(image)

        depth = np.array(result["depth"])

        depth = depth.astype(np.float32)

        depth = cv2.normalize(
            depth,
            None,
            0.0,
            1.0,
            cv2.NORM_MINMAX
        )

        return depth