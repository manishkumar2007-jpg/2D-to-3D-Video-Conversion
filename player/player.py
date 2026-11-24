import sys
import cv2
import time
import queue
import threading

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from core.depth_estimator import DepthEstimator
from core.stereo import depth_to_disparity, create_stereo_pair
from core.anaglyph import create_anaglyph
from core.temporal_smoothing import TemporalSmoother
from core.settings import load_settings


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_FOLDER = PROJECT_ROOT / "videos"


# =========================================================
# VIDEO PROCESSING THREAD
# =========================================================

class VideoProcessor(threading.Thread):

    def __init__(self, video_path, frame_queue, stop_event):

        super().__init__(daemon=True)

        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = stop_event

        self.finished = False
        self.error = None

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        self.frame_duration = 1.0 / self.fps

        self.settings = load_settings()

        self.strength = float(
            self.settings["3d_strength"]
        )

        self.maximum_disparity = int(
            self.settings["maximum_disparity"]
        )

        self.depth_inversion = bool(
            self.settings["depth_inversion"]
        )

        self.smoothing = float(
            self.settings["temporal_smoothing"]
        )

        self.estimator = DepthEstimator()

        self.smoother = TemporalSmoother(
            alpha=self.smoothing
        )

    def run(self):

        try:

            while not self.stop_event.is_set():

                ret, frame = self.cap.read()

                if not ret:
                    break

                frame = cv2.resize(
                    frame,
                    (640, 360)
                )

                # AI depth estimation
                depth = self.estimator.estimate(frame)

                # Temporal smoothing
                depth = self.smoother.apply(depth)

                # Depth to disparity
                disparity = depth_to_disparity(
                    depth,
                    self.maximum_disparity,
                    self.strength,
                    self.depth_inversion
                )

                # Stereo generation
                left, right = create_stereo_pair(
                    frame,
                    disparity
                )

                # Anaglyph generation
                output = create_anaglyph(
                    left,
                    right
                )

                # Wait until the display queue is empty.
                # Do not discard processed frames.
                while (
                    not self.stop_event.is_set()
                    and self.frame_queue.full()
                ):
                    time.sleep(0.005)

                if self.stop_event.is_set():
                    break

                self.frame_queue.put(output)

        except Exception as error:

            self.error = str(error)

        finally:

            self.finished = True
            self.cap.release()


# =========================================================
# MAIN VIDEO PLAYER
# =========================================================

class VideoPlayer(QMainWindow):

    def __init__(self, video_path):

        super().__init__()

        self.setWindowTitle("2D to 3D Video Player")
        self.resize(1000, 650)

        self.video_path = video_path

        self.frame_queue = queue.Queue(maxsize=1)

        self.stop_event = threading.Event()

        self.playing = True
        self.audio_started = False
        self.closed = False

        self.next_frame_time = time.monotonic()

        # -----------------------------------------
        # VIDEO PROCESSOR
        # -----------------------------------------

        self.processor = VideoProcessor(
            video_path,
            self.frame_queue,
            self.stop_event
        )

        self.frame_duration = self.processor.frame_duration

        # -----------------------------------------
        # AUDIO PLAYER
        # -----------------------------------------

        self.audio_output = QAudioOutput()

        self.audio_player = QMediaPlayer()

        self.audio_player.setAudioOutput(
            self.audio_output
        )

        self.audio_player.setSource(
            QUrl.fromLocalFile(
                str(Path(video_path).resolve())
            )
        )

        self.audio_output.setVolume(1.0)

        # -----------------------------------------
        # VIDEO DISPLAY
        # -----------------------------------------

        self.video_label = QLabel()

        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.video_label.setStyleSheet(
            "background-color: black;"
        )

        self.video_label.setMinimumSize(640, 360)

        # -----------------------------------------
        # BUTTONS
        # -----------------------------------------

        self.pause_button = QPushButton("Pause")

        self.pause_button.clicked.connect(
            self.toggle_playback
        )

        self.quit_button = QPushButton("Quit")

        self.quit_button.clicked.connect(
            self.close
        )

        button_layout = QHBoxLayout()

        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.quit_button)

        # -----------------------------------------
        # MAIN LAYOUT
        # -----------------------------------------

        layout = QVBoxLayout()

        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        # -----------------------------------------
        # KEYBOARD SHORTCUTS
        # -----------------------------------------

        self.pause_shortcut = QShortcut(
            QKeySequence("Space"),
            self
        )

        self.pause_shortcut.activated.connect(
            self.toggle_playback
        )

        self.quit_shortcut = QShortcut(
            QKeySequence("Q"),
            self
        )

        self.quit_shortcut.activated.connect(
            self.close
        )

        # -----------------------------------------
        # TIMER
        # -----------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_video
        )

        self.timer.start(5)

        self.processor.start()

    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    def display_frame(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb_frame.shape

        bytes_per_line = channels * width

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    # =====================================================
    # VIDEO AND AUDIO SYNCHRONIZATION
    # =====================================================

    def update_video(self):

        if not self.playing or self.closed:
            return

        if time.monotonic() < self.next_frame_time:
            return

        try:

            frame = self.frame_queue.get_nowait()

        except queue.Empty:

            # Pause audio if the next frame is not ready
            if self.audio_started:
                self.audio_player.pause()

            if self.processor.finished:

                self.audio_player.stop()
                self.timer.stop()

                if self.processor.error:
                    print(
                        "Processing error:",
                        self.processor.error
                    )

                return

            return

        # Display next processed frame
        self.display_frame(frame)

        # Start or resume audio
        if not self.audio_started:

            self.audio_player.play()
            self.audio_started = True

        else:

            self.audio_player.play()

        # Schedule next frame
        self.next_frame_time = (
            time.monotonic() + self.frame_duration
        )

    # =====================================================
    # PAUSE / RESUME
    # =====================================================

    def toggle_playback(self):

        if self.closed:
            return

        if self.playing:

            self.playing = False

            self.audio_player.pause()

            self.pause_button.setText("Resume")

        else:

            self.playing = True

            self.next_frame_time = time.monotonic()

            self.pause_button.setText("Pause")

    # =====================================================
    # CLOSE PLAYER
    # =====================================================

    def closeEvent(self, event):

        self.closed = True

        self.timer.stop()

        self.audio_player.stop()

        self.stop_event.set()

        if self.processor.is_alive():
            self.processor.join(timeout=2)

        event.accept()


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    VIDEO_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    video_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Video",
        str(VIDEO_FOLDER),
        "Video Files (*.mp4 *.avi *.mkv *.mov)"
    )

    if not video_path:
        sys.exit(0)

    try:

        window = VideoPlayer(video_path)

        window.show()

        sys.exit(app.exec())

    except Exception as error:

        print("Player Error:", error)
        sys.exit(1)
