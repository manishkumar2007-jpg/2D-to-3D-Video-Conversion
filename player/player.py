import sys
import os
import cv2

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication, QFileDialog

from core.depth_estimator import DepthEstimator
from core.stereo import depth_to_disparity, create_stereo_pair
from core.anaglyph import create_anaglyph
from core.temporal_smoothing import TemporalSmoother
from core.settings import load_settings


def select_video():
    """Open a file selection window."""

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select 2D Video",
        os.path.join(PROJECT_ROOT, "videos"),
        "Video Files (*.mp4 *.avi *.mov *.mkv)"
    )

    return video_path


def main():

    print("=" * 50)
    print("       2D TO 3D ANAGLYPH VIDEO PLAYER")
    print("=" * 50)

    # -----------------------------------------
    # Load saved calibration settings
    # -----------------------------------------

    settings_path = os.path.join(PROJECT_ROOT, "settings.json")

    settings = load_settings(settings_path)

    strength = settings["3d_strength"]
    maximum_disparity = settings["maximum_disparity"]
    invert_depth = settings["depth_inversion"]
    smoothing = settings["temporal_smoothing"]

    print("\nLoaded settings:")
    print("3D Strength       :", strength)
    print("Maximum Disparity :", maximum_disparity)
    print("Depth Inversion   :", invert_depth)
    print("Temporal Smoothing:", smoothing)

    # -----------------------------------------
    # Select video
    # -----------------------------------------

    video_path = select_video()

    if not video_path:
        print("No video selected.")
        return

    print("\nSelected video:")
    print(video_path)

    # -----------------------------------------
    # Open video
    # -----------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\nVideo information:")
    print("Resolution:", width, "x", height)
    print("FPS:", fps)

    # -----------------------------------------
    # Load AI depth estimator
    # -----------------------------------------

    print("\nLoading depth AI...")

    depth_estimator = DepthEstimator()

    print("Depth AI ready.")

    # -----------------------------------------
    # Temporal smoothing
    # -----------------------------------------

    smoother = TemporalSmoother(alpha=smoothing)

    print("\nStarting 3D playback...")
    print("Controls:")
    print("SPACE = Pause / Resume")
    print("Q     = Quit")

    paused = False

    # -----------------------------------------
    # Main video loop
    # -----------------------------------------

    while True:

        if not paused:

            success, frame = cap.read()

            if not success:
                print("\nVideo finished.")
                break

            # ---------------------------------
            # Resize for faster processing
            # ---------------------------------

            frame = cv2.resize(
                frame,
                (640, 360)
            )

            # ---------------------------------
            # AI depth estimation
            # ---------------------------------

            depth = depth_estimator.estimate(frame)

            # ---------------------------------
            # Temporal smoothing
            # ---------------------------------

            depth = smoother.apply(depth)

            # ---------------------------------
            # Depth → disparity
            # ---------------------------------

            disparity = depth_to_disparity(
                depth,
                maximum_disparity=maximum_disparity,
                strength=strength,
                invert=invert_depth
            )

            # ---------------------------------
            # Create stereo views
            # ---------------------------------

            left, right = create_stereo_pair(
                frame,
                disparity
            )

            # ---------------------------------
            # Create red/cyan image
            # ---------------------------------

            anaglyph = create_anaglyph(
                left,
                right
            )

            # ---------------------------------
            # Display
            # ---------------------------------

            cv2.imshow(
                "2D to 3D Player - Red/Cyan",
                anaglyph
            )

        # -------------------------------------
        # Keyboard controls
        # -------------------------------------

        key = cv2.waitKey(
            max(1, int(1000 / fps))
        ) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            paused = not paused

            if paused:
                print("Paused")
            else:
                print("Playing")

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    cap.release()
    cv2.destroyAllWindows()

    print("\n3D player closed.")


if __name__ == "__main__":
    main()