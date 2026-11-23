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
from core.stereo import (
    depth_to_disparity,
    create_stereo_pair
)
from core.anaglyph import create_anaglyph
from core.temporal_smoothing import TemporalSmoother
from core.settings import load_settings


# --------------------------------------------------
# VIDEO PROCESSING THREAD
# --------------------------------------------------

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
            raise RuntimeError("Unable to open video file.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        self.frame_duration = 1.0 / self.fps

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Load existing settings
        self.settings = load_settings("settings.json")

        self.strength = self.settings.get("3d_strength", 1.0)

        self.maximum_disparity = self.settings.get(
            "maximum_disparity", 30
        )

        self.depth_inversion = self.settings.get(
            "depth_inversion", False
        )

        self.temporal_enabled = self.settings.get(
            "temporal_smoothing", True
        )

        # Initialize AI model
        self.estimator = DepthEstimator()

        self.smoother = TemporalSmoother(alpha=0.7)

    def run(self):

        try:

            while not self.stop_event.is_set():

                ret, frame = self.cap.read()

                if not ret:
                    break

                # Reduce resolution for better CPU performance
                frame = cv2.resize(frame, (640, 360))

                # AI depth estimation
                depth = self.estimator.estimate(frame)

                # Temporal smoothing
                if self.temporal_enabled:
                    depth = self.smoother.apply(depth)

                # Convert depth into disparity
                disparity = depth_to_disparity(
                    depth,
                    self.maximum_disparity,
                    self.strength,
                    self.depth_inversion
                )

                # Generate stereo images
                left, right = create_stereo_pair(
                    frame,
                    disparity
                )

                # Generate red-cyan anaglyph
                output = create_anaglyph(left, right)

                # Wait if the queue already contains a frame.
                # IMPORTANT: We do not discard old frames.
                while (
                    not self.stop_event.is_set()
                    and self.frame_queue.full()
                ):
                    time.sleep(0.005)

                if self.stop_event.is_set():
                    break

                self.frame_queue.put(output)

        except Exception as e:

            self.error = str(e)

        finally:

            self.finished = True
            self.cap.release()


# --------------------------------------------------
# MAIN VIDEO PLAYER
# --------------------------------------------------

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

        self.frame_duration = 1.0 / 30.0

        self.last_frame = None

        # ------------------------------------------
        # VIDEO PROCESSOR
        # ------------------------------------------

        self.processor = VideoProcessor(
            video_path,
            self.frame_queue,
            self.stop_event
        )

        self.frame_duration = self.processor.frame_duration

        # ------------------------------------------
        # AUDIO PLAYER
        # ------------------------------------------

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

        # ------------------------------------------
        # VIDEO DISPLAY
        # ------------------------------------------

        self.video_label = QLabel()

        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.video_label.setStyleSheet(
            "background-color: black;"
        )

        self.video_label.setMinimumSize(640, 360)

        # ------------------------------------------
        # STATUS LABEL
        # ------------------------------------------

        self.status_label = QLabel(
            "Initializing video..."
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        # ------------------------------------------
        # CONTROL BUTTONS
        # ------------------------------------------

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

        # ------------------------------------------
        # MAIN LAYOUT
        # ------------------------------------------

        layout = QVBoxLayout()

        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)

        container = QWidget()

        container.setLayout(layout)

        self.setCentralWidget(container)

        # ------------------------------------------
        # KEYBOARD SHORTCUTS
        # ------------------------------------------

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

        # ------------------------------------------
        # TIMER
        # ------------------------------------------

        self.timer = QTimer(self)

        self.timer.timeout.connect(
            self.update_video
        )

        self.timer.start(5)

        # Start processing
        self.processor.start()

    # --------------------------------------------------
    # DISPLAY FRAME
    # --------------------------------------------------

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

    # --------------------------------------------------
    # VIDEO AND AUDIO SYNCHRONIZATION
    # --------------------------------------------------

    def update_video(self):

        if not self.playing or self.closed:
            return

        current_time = time.monotonic()

        # Wait until the current frame's display interval ends
        if current_time < self.next_frame_time:
            return

        # Check whether the next processed frame is ready
        try:

            frame = self.frame_queue.get_nowait()

        except queue.Empty:

            # No next frame available:
            # Pause audio so it cannot run ahead.
            if self.audio_started:
                self.audio_player.pause()

            # Check again shortly
            if self.processor.finished:

                if self.processor.error:

                    self.status_label.setText(
                        "Processing error: " +
                        self.processor.error
                    )

                else:

                    self.status_label.setText(
                        "Video playback completed."
                    )

                self.audio_player.stop()
                self.timer.stop()

                return

            return

        # ------------------------------------------
        # A NEW FRAME IS READY
        # ------------------------------------------

        self.last_frame = frame

        self.display_frame(frame)

        # Start or resume audio
        if not self.audio_started:

            self.audio_player.play()

            self.audio_started = True

        else:

            self.audio_player.play()

        # Schedule the next frame
        self.next_frame_time = (
            time.monotonic() + self.frame_duration
        )


    # --------------------------------------------------
    # PAUSE / RESUME
    # --------------------------------------------------

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

    # --------------------------------------------------
    # WINDOW CLOSE / QUIT
    # --------------------------------------------------

    def closeEvent(self, event):

        self.closed = True

        self.timer.stop()

        self.audio_player.stop()

        self.stop_event.set()

        if self.processor.is_alive():

            self.processor.join(timeout=2)

        event.accept()


# --------------------------------------------------
# APPLICATION START
# --------------------------------------------------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    video_path = None

    # Allow video path through terminal argument
    if len(sys.argv) > 1:

        video_path = sys.argv[1]

    else:

       from pathlib import Path

       video_folder = Path(__file__).resolve().parent.parent / "videos"

       video_path, _ = QFileDialog.getOpenFileName(
          None,
          "Select Video",
          str(video_folder),
          "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
    if not video_path:

        sys.exit(0)

    try:

        window = VideoPlayer(video_path)

        window.show()

        sys.exit(app.exec())

    except Exception as e:

        print("Player Error:", e)

        sys.exit(1)
