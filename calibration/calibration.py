import sys
import cv2

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QFileDialog
)

from core.depth_estimator import DepthEstimator
from core.stereo import depth_to_disparity, create_stereo_pair
from core.anaglyph import create_anaglyph
from core.temporal_smoothing import TemporalSmoother
from core.settings import save_settings, load_settings


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_FOLDER = PROJECT_ROOT / "videos"

DEFAULT_VIDEO = VIDEO_FOLDER / "sample.mp4"

SETTINGS_PATH = PROJECT_ROOT / "settings.json"

SUPPORTED_FORMATS = (
    "*.mp4 *.avi *.mkv *.mov"
)

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


# =========================================================
# VIDEO SELECTION
# =========================================================

def select_video():

    VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)

    # Use sample.mp4 if it exists
    if DEFAULT_VIDEO.exists():
        return str(DEFAULT_VIDEO)

    # Search for available videos
    videos = []

    for extension in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        videos.extend(VIDEO_FOLDER.glob(extension))

    if len(videos) == 1:
        return str(videos[0])

    # If multiple videos or no videos, open picker
    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Calibration Video",
        str(VIDEO_FOLDER),
        f"Video Files ({SUPPORTED_FORMATS})"
    )

    return video_path


# =========================================================
# CALIBRATION WINDOW
# =========================================================

class CalibrationWindow(QWidget):

    def __init__(self, video_path):

        super().__init__()

        self.setWindowTitle("2D to 3D - Calibration")
        self.resize(1100, 700)

        # -----------------------------
        # Load saved settings
        # -----------------------------

        self.settings = load_settings()

        self.strength = float(
            self.settings["3d_strength"]
        )

        self.maximum_disparity = int(
            self.settings["maximum_disparity"]
        )

        self.invert_depth = bool(
            self.settings["depth_inversion"]
        )

        self.smoothing = float(
            self.settings["temporal_smoothing"]
        )

        # -----------------------------
        # Video
        # -----------------------------

        self.video_path = video_path

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = 30

        # -----------------------------
        # Processing
        # -----------------------------

        print("Loading depth model...")

        self.depth_estimator = DepthEstimator()

        print("Depth model loaded.")

        self.smoother = TemporalSmoother(
            alpha=self.smoothing
        )

        # -----------------------------
        # State
        # -----------------------------

        self.paused = False

        # -----------------------------
        # GUI
        # -----------------------------

        self.create_ui()

        # -----------------------------
        # Timer
        # -----------------------------

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

        interval = int(1000 / self.fps)

        self.timer.start(max(interval, 30))

    # =====================================================
    # GUI
    # =====================================================

    def create_ui(self):

        main_layout = QHBoxLayout()

        # -----------------------------------------
        # Preview
        # -----------------------------------------

        self.preview_label = QLabel("Loading preview...")

        self.preview_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.preview_label.setMinimumSize(
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT
        )

        self.preview_label.setStyleSheet(
            """
            QLabel {
                background-color: black;
                color: white;
                border: 2px solid #555;
            }
            """
        )

        # -----------------------------------------
        # Controls
        # -----------------------------------------

        controls = QVBoxLayout()

        title = QLabel("3D Calibration")

        title.setStyleSheet(
            """
            QLabel {
                font-size: 22px;
                font-weight: bold;
            }
            """
        )

        controls.addWidget(title)

        # -----------------------------------------
        # 3D Strength
        # -----------------------------------------

        self.strength_value = QLabel()

        strength_title = QLabel("3D Strength")

        self.strength_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(300)

        self.strength_slider.setValue(
            int(self.strength * 100)
        )

        self.strength_slider.valueChanged.connect(
            self.update_strength
        )

        controls.addWidget(strength_title)
        controls.addWidget(self.strength_slider)
        controls.addWidget(self.strength_value)

        # -----------------------------------------
        # Maximum Disparity
        # -----------------------------------------

        self.disparity_value = QLabel()

        disparity_title = QLabel(
            "Maximum Disparity"
        )

        self.disparity_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.disparity_slider.setMinimum(0)
        self.disparity_slider.setMaximum(50)

        self.disparity_slider.setValue(
            self.maximum_disparity
        )

        self.disparity_slider.valueChanged.connect(
            self.update_disparity
        )

        controls.addWidget(disparity_title)
        controls.addWidget(self.disparity_slider)
        controls.addWidget(self.disparity_value)

        # -----------------------------------------
        # Temporal Smoothing
        # -----------------------------------------

        self.smoothing_value = QLabel()

        smoothing_title = QLabel(
            "Temporal Smoothing"
        )

        self.smoothing_slider = QSlider(
            Qt.Orientation.Horizontal
        )

        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(100)

        self.smoothing_slider.setValue(
            int(self.smoothing * 100)
        )

        self.smoothing_slider.valueChanged.connect(
            self.update_smoothing
        )

        controls.addWidget(smoothing_title)
        controls.addWidget(self.smoothing_slider)
        controls.addWidget(self.smoothing_value)

        # -----------------------------------------
        # Depth Inversion
        # -----------------------------------------

        self.invert_checkbox = QCheckBox(
            "Invert Depth"
        )

        self.invert_checkbox.setChecked(
            self.invert_depth
        )

        self.invert_checkbox.stateChanged.connect(
            self.update_inversion
        )

        controls.addWidget(self.invert_checkbox)

        # -----------------------------------------
        # Play / Pause
        # -----------------------------------------

        self.play_button = QPushButton("Pause")

        self.play_button.clicked.connect(
            self.toggle_play
        )

        controls.addWidget(self.play_button)

        # -----------------------------------------
        # Save Settings
        # -----------------------------------------

        self.save_button = QPushButton(
            "Save Settings"
        )

        self.save_button.clicked.connect(
            self.save_current_settings
        )

        controls.addWidget(self.save_button)

        # -----------------------------------------
        # Reset Defaults
        # -----------------------------------------

        self.reset_button = QPushButton(
            "Reset Defaults"
        )

        self.reset_button.clicked.connect(
            self.reset_settings
        )

        controls.addWidget(self.reset_button)

        controls.addStretch()

        # -----------------------------------------
        # Combine Layouts
        # -----------------------------------------

        main_layout.addWidget(
            self.preview_label,
            stretch=3
        )

        main_layout.addLayout(
            controls,
            stretch=1
        )

        self.setLayout(main_layout)

        # Initial values
        self.update_strength(
            self.strength_slider.value()
        )

        self.update_disparity(
            self.disparity_slider.value()
        )

        self.update_smoothing(
            self.smoothing_slider.value()
        )

    # =====================================================
    # SLIDER UPDATES
    # =====================================================

    def update_strength(self, value):

        self.strength = value / 100.0

        self.strength_value.setText(
            f"{self.strength:.2f}"
        )

    def update_disparity(self, value):

        self.maximum_disparity = value

        self.disparity_value.setText(
            f"{self.maximum_disparity} pixels"
        )

    def update_smoothing(self, value):

        self.smoothing = value / 100.0

        self.smoothing_value.setText(
            f"{self.smoothing:.2f}"
        )

        self.smoother.alpha = self.smoothing

    def update_inversion(self, state):

        self.invert_depth = (
            state == Qt.CheckState.Checked.value
        )

        self.smoother.reset()

    # =====================================================
    # PLAY / PAUSE
    # =====================================================

    def toggle_play(self):

        self.paused = not self.paused

        if self.paused:
            self.play_button.setText("Play")
        else:
            self.play_button.setText("Pause")

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def process_frame(self):

        if self.paused:
            return

        success, frame = self.cap.read()

        if not success:

            # Restart video automatically
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            self.smoother.reset()

            success, frame = self.cap.read()

            if not success:
                return

        # Resize
        frame = cv2.resize(
            frame,
            (PREVIEW_WIDTH, PREVIEW_HEIGHT)
        )

        # -----------------------------------------
        # Depth Estimation
        # -----------------------------------------

        depth = self.depth_estimator.estimate(
            frame
        )

        # -----------------------------------------
        # Temporal Smoothing
        # -----------------------------------------

        depth = self.smoother.apply(depth)

        # -----------------------------------------
        # Depth to Disparity
        # -----------------------------------------

        disparity = depth_to_disparity(
            depth,
            maximum_disparity=self.maximum_disparity,
            strength=self.strength,
            invert=self.invert_depth
        )

        # -----------------------------------------
        # Stereo Generation
        # -----------------------------------------

        left, right = create_stereo_pair(
            frame,
            disparity
        )

        # -----------------------------------------
        # Anaglyph Generation
        # -----------------------------------------

        anaglyph = create_anaglyph(
            left,
            right
        )

        # -----------------------------------------
        # Display
        # -----------------------------------------

        rgb = cv2.cvtColor(
            anaglyph,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb.shape

        bytes_per_line = channels * width

        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    # =====================================================
    # SAVE SETTINGS
    # =====================================================

    def save_current_settings(self):

        settings = {
            "3d_strength": self.strength,
            "maximum_disparity": self.maximum_disparity,
            "depth_inversion": self.invert_depth,
            "temporal_smoothing": self.smoothing
        }

        save_settings(
            settings,
            str(SETTINGS_PATH)
        )

        QMessageBox.information(
            self,
            "Settings Saved",
            "Calibration settings saved successfully."
        )

    # =====================================================
    # RESET SETTINGS
    # =====================================================

    def reset_settings(self):

        self.strength_slider.setValue(100)
        self.disparity_slider.setValue(15)
        self.smoothing_slider.setValue(70)

        self.invert_checkbox.setChecked(False)

        self.smoother.reset()

    # =====================================================
    # CLOSE
    # =====================================================

    def closeEvent(self, event):

        self.timer.stop()

        self.cap.release()

        event.accept()


# =========================================================
# APPLICATION
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    video_path = select_video()

    if not video_path:

        QMessageBox.warning(
            None,
            "Video Not Found",
            "No video was selected. Add a video to the videos folder."
        )

        sys.exit(0)

    try:

        window = CalibrationWindow(video_path)

        window.show()

        sys.exit(app.exec())

    except Exception as error:

        QMessageBox.critical(
            None,
            "Calibration Error",
            str(error)
        )

        print("Calibration Error:", error)

        sys.exit(1)
