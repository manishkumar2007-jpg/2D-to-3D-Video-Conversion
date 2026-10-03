import sys
import cv2

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
    QGridLayout,
    QMessageBox
)

from core.depth_estimator import DepthEstimator
from core.stereo import depth_to_disparity, create_stereo_pair
from core.anaglyph import create_anaglyph
from core.temporal_smoothing import TemporalSmoother
from core.settings import save_settings, load_settings


VIDEO_PATH = "videos/sample.mp4"

# Preview resolution.
# Keeping this small makes the calibration program faster.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360


class CalibrationWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("2D to 3D - Calibration")
        self.resize(1100, 700)

        # -----------------------------
        # Load saved settings
        # -----------------------------
        self.settings = load_settings()

        self.strength = float(self.settings["3d_strength"])
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
        self.cap = cv2.VideoCapture(VIDEO_PATH)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {VIDEO_PATH}"
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
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        interval = int(1000 / self.fps)

        # We use a slightly larger interval because
        # depth estimation is computationally expensive.
        self.timer.start(max(interval, 30))

    # =========================================================
    # GUI
    # =========================================================

    def create_ui(self):

        main_layout = QHBoxLayout()

        # -----------------------------------------
        # Preview
        # -----------------------------------------

        self.preview_label = QLabel("Loading preview...")

        self.preview_label.setAlignment(
            Qt.AlignCenter
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
        # Strength
        # -----------------------------------------

        self.strength_value = QLabel()

        strength_title = QLabel("3D Strength")

        self.strength_slider = QSlider(
            Qt.Horizontal
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
        # Maximum disparity
        # -----------------------------------------

        self.disparity_value = QLabel()

        disparity_title = QLabel(
            "Maximum Disparity"
        )

        self.disparity_slider = QSlider(
            Qt.Horizontal
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
        controls.addWidget(
            self.disparity_slider
        )
        controls.addWidget(
            self.disparity_value
        )

        # -----------------------------------------
        # Temporal smoothing
        # -----------------------------------------

        self.smoothing_value = QLabel()

        smoothing_title = QLabel(
            "Temporal Smoothing"
        )

        self.smoothing_slider = QSlider(
            Qt.Horizontal
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
        controls.addWidget(
            self.smoothing_slider
        )
        controls.addWidget(
            self.smoothing_value
        )

        # -----------------------------------------
        # Depth inversion
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

        controls.addWidget(
            self.invert_checkbox
        )

        # -----------------------------------------
        # Play / Pause
        # -----------------------------------------

        self.play_button = QPushButton(
            "Pause"
        )

        self.play_button.clicked.connect(
            self.toggle_play
        )

        controls.addWidget(
            self.play_button
        )

        # -----------------------------------------
        # Save
        # -----------------------------------------

        self.save_button = QPushButton(
            "Save Settings"
        )

        self.save_button.clicked.connect(
            self.save_current_settings
        )

        controls.addWidget(
            self.save_button
        )

        # -----------------------------------------
        # Reset
        # -----------------------------------------

        self.reset_button = QPushButton(
            "Reset Defaults"
        )

        self.reset_button.clicked.connect(
            self.reset_settings
        )

        controls.addWidget(
            self.reset_button
        )

        controls.addStretch()

        # -----------------------------------------
        # Combine layouts
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

        # Show initial values
        self.update_strength(
            self.strength_slider.value()
        )

        self.update_disparity(
            self.disparity_slider.value()
        )

        self.update_smoothing(
            self.smoothing_slider.value()
        )

    # =========================================================
    # Slider updates
    # =========================================================

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
            state == Qt.Checked
        )

        # Reset temporal history because the
        # depth interpretation has changed.
        self.smoother.reset()

    # =========================================================
    # Play / Pause
    # =========================================================

    def toggle_play(self):

        self.paused = not self.paused

        if self.paused:
            self.play_button.setText("Play")
        else:
            self.play_button.setText("Pause")

    # =========================================================
    # Process frame
    # =========================================================

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
            (
                PREVIEW_WIDTH,
                PREVIEW_HEIGHT
            )
        )

        # -----------------------------
        # Depth estimation
        # -----------------------------

        depth = self.depth_estimator.estimate(
            frame
        )

        # -----------------------------
        # Temporal smoothing
        # -----------------------------

        depth = self.smoother.apply(
            depth
        )

        # -----------------------------
        # Depth → disparity
        # -----------------------------

        disparity = depth_to_disparity(
            depth,
            maximum_disparity=
                self.maximum_disparity,
            strength=self.strength,
            invert=self.invert_depth
        )

        # -----------------------------
        # Stereo
        # -----------------------------

        left, right = create_stereo_pair(
            frame,
            disparity
        )

        # -----------------------------
        # Anaglyph
        # -----------------------------

        anaglyph = create_anaglyph(
            left,
            right
        )

        # -----------------------------
        # Display
        # -----------------------------

        rgb = cv2.cvtColor(
            anaglyph,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb.shape

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            image
        )

        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # =========================================================
    # Save settings
    # =========================================================

    def save_current_settings(self):

        settings = {
            "3d_strength": self.strength,
            "maximum_disparity":
                self.maximum_disparity,
            "depth_inversion":
                self.invert_depth,
            "temporal_smoothing":
                self.smoothing
        }

        save_settings(
            settings,
            "settings.json"
        )

        QMessageBox.information(
            self,
            "Settings Saved",
            "Calibration settings saved successfully."
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset_settings(self):

        self.strength_slider.setValue(100)
        self.disparity_slider.setValue(15)
        self.smoothing_slider.setValue(70)

        self.invert_checkbox.setChecked(
            False
        )

        self.smoother.reset()

    # =========================================================
    # Close
    # =========================================================

    def closeEvent(self, event):

        self.timer.stop()

        self.cap.release()

        event.accept()


# =============================================================
# Application
# =============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = CalibrationWindow()

    window.show()

    sys.exit(
        app.exec()
    )