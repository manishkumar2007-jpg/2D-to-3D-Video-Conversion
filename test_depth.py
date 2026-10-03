import cv2
from core.depth_estimator import DepthEstimator


estimator = DepthEstimator()

image = cv2.imread("test.jpg")

depth = estimator.estimate(image)

depth_display = (depth * 255).astype("uint8")

cv2.imshow("Original", image)
cv2.imshow("Depth Map", depth_display)

cv2.waitKey(0)
cv2.destroyAllWindows()