import cv2

from core.depth_estimator import DepthEstimator
from core.stereo import depth_to_disparity, create_stereo_pair
from core.anaglyph import create_anaglyph


VIDEO_PATH = "videos/sample.mp4"

MAX_DISPARITY = 15
STRENGTH = 1.0
INVERT_DEPTH = False


print("Starting 2D → 3D test...")

# -----------------------------
# Load depth model
# -----------------------------
print("Loading depth model...")

depth_estimator = DepthEstimator()

print("Depth model ready.")


# -----------------------------
# Open video
# -----------------------------
print(f"Opening video: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

print("Video opened successfully.")

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {fps}")
print(f"Video resolution: {width} x {height}")
print(f"Frame count: {frame_count}")


# -----------------------------
# Read first frame
# -----------------------------
success, frame = cap.read()

if not success or frame is None:
    cap.release()
    raise RuntimeError(
        "Video opened, but OpenCV could not read the first frame."
    )

print("First frame read successfully.")


# Resize for faster processing
frame = cv2.resize(frame, (640, 360))

print("Estimating depth...")

depth = depth_estimator.estimate(frame)

print("Depth estimation successful.")


# -----------------------------
# Depth → Disparity
# -----------------------------
print("Creating disparity...")

disparity = depth_to_disparity(
    depth,
    maximum_disparity=MAX_DISPARITY,
    strength=STRENGTH,
    invert=INVERT_DEPTH
)

print("Disparity created.")


# -----------------------------
# Stereo rendering
# -----------------------------
print("Creating left/right views...")

left, right = create_stereo_pair(
    frame,
    disparity
)

print("Stereo views created.")


# -----------------------------
# Red/Cyan anaglyph
# -----------------------------
print("Creating red/cyan anaglyph...")

anaglyph = create_anaglyph(
    left,
    right
)

print("Anaglyph created.")


# -----------------------------
# Display
# -----------------------------
cv2.imshow("Original", frame)
cv2.imshow("Left Eye", left)
cv2.imshow("Right Eye", right)
cv2.imshow("2D to 3D Anaglyph", anaglyph)

print()
print("3D preview is now displayed.")
print("Press any key inside an OpenCV window to close.")

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print("Test completed.")