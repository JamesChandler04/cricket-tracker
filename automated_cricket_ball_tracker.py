import sys
import os
import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
import automations
from log_bridge import bridge


class TrackingWorker(QThread):
    finished = pyqtSignal(list, list)  # top_down_frames, side_on_frames
    error = pyqtSignal(str)

    def __init__(self, top_down_path: str, side_on_path: str,
                 top_down_tracker, side_on_tracker):
        super().__init__()
        self.top_down_path = top_down_path
        self.side_on_path = side_on_path
        self.top_down_tracker = top_down_tracker
        self.side_on_tracker = side_on_tracker

    def run(self):
        try:
            top_down_video = automations.Video(self.top_down_path)
            side_on_video = automations.Video(self.side_on_path)

            top_down_ball_data = self.top_down_tracker.get_ball_data(top_down_video)
            side_on_ball_data = self.side_on_tracker.get_ball_data(side_on_video)

            print("\nTracking Results:")
            if not top_down_ball_data:
                print("Warning: No ball data found in top down video.")
            else:
                for data_point in top_down_ball_data:
                    print(f"Top Down - Frame {data_point.frame_number}: Ball Position = {data_point.data.centre}, Seam Angle = {data_point.data.seam_angle:.2f} degrees")

            if not side_on_ball_data:
                print("Warning: No ball data found in side on video.")
            else:
                for data_point in side_on_ball_data:
                    print(f"Side On - Frame {data_point.frame_number}: Ball Position = {data_point.data.centre}")

            # Collect saved frame images from output folders
            top_down_frames = sorted(glob.glob(f"{automations.top_down_tracking_folder}/frame_*.jpg"))
            side_on_frames  = sorted(glob.glob(f"{automations.side_on_tracking_folder}/frame_*.jpg"))

            self.finished.emit(top_down_frames, side_on_frames)
        except Exception as e:
            self.error.emit(str(e))


class FrameViewer(QWidget):
    """A labelled image viewer with prev/next navigation."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._frames: list[str] = []
        self._index: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title
        lbl = QLabel(title)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        lbl.setFont(font)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 340)
        self.image_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #555;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # Nav row
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)

        self.prev_btn = QPushButton("◀  Prev")
        self.prev_btn.setFixedHeight(32)
        self.prev_btn.clicked.connect(self.prev_frame)

        self.counter_label = QLabel("–")
        self.counter_label.setAlignment(Qt.AlignCenter)
        self.counter_label.setMinimumWidth(100)

        self.next_btn = QPushButton("Next  ▶")
        self.next_btn.setFixedHeight(32)
        self.next_btn.clicked.connect(self.next_frame)

        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.counter_label)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        self._set_empty()

    def load_frames(self, paths: list[str]):
        self._frames = paths
        self._index = 0
        if paths:
            self._show_current()
        else:
            self._set_empty()

    def _set_empty(self):
        self.image_label.setText("No frames")
        self.image_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #555; color: #666;")
        self.counter_label.setText("–")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def _show_current(self):
        path = self._frames[self._index]
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText(f"Could not load:\n{os.path.basename(path)}")
            return

        scaled = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

        total = len(self._frames)
        # Extract frame number from filename for display
        name = os.path.basename(path)
        frame_num = name.replace("frame_", "").replace(".jpg", "").lstrip("0") or "0"
        self.counter_label.setText(f"Frame {frame_num}  ({self._index + 1}/{total})")
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(self._index < total - 1)

    def prev_frame(self):
        if self._index > 0:
            self._index -= 1
            self._show_current()

    def next_frame(self):
        if self._index < len(self._frames) - 1:
            self._index += 1
            self._show_current()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frames:
            self._show_current()


class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.top_down_video_path = None
        self.side_on_video_path = None
        self._worker = None

        sys.stdout = bridge
        bridge.message_received.connect(self.append_log)

        self.top_down_tracker = automations.TopDownBallFinder()
        self.side_on_tracker = automations.SideOnBallFinder()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cricket Ball Tracker")
        self.setMinimumSize(1600, 900)
        self.setStyleSheet("background-color: #f0f0f0;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("Cricket Ball Tracker")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # File pickers
        top_down_video_layout = QHBoxLayout()
        top_down_video_label = QLabel("Top Down View:")
        top_down_video_label.setMinimumWidth(150)
        self.top_down_video_display = QLabel("No file selected")
        self.top_down_video_display.setStyleSheet(
            "background-color: white; padding: 10px; border: 1px solid #ccc;")
        top_down_video_btn = QPushButton("Browse")
        top_down_video_btn.clicked.connect(self.select_top_down_video)
        top_down_video_btn.setFixedWidth(100)
        top_down_video_layout.addWidget(top_down_video_label)
        top_down_video_layout.addWidget(self.top_down_video_display)
        top_down_video_layout.addWidget(top_down_video_btn)
        main_layout.addLayout(top_down_video_layout)

        side_on_video_layout = QHBoxLayout()
        side_on_video_label = QLabel("Side On View:")
        side_on_video_label.setMinimumWidth(150)
        self.side_on_video_display = QLabel("No file selected")
        self.side_on_video_display.setStyleSheet(
            "background-color: white; padding: 10px; border: 1px solid #ccc;")
        side_on_video_btn = QPushButton("Browse")
        side_on_video_btn.clicked.connect(self.select_side_on_video)
        side_on_video_btn.setFixedWidth(100)
        side_on_video_layout.addWidget(side_on_video_label)
        side_on_video_layout.addWidget(self.side_on_video_display)
        side_on_video_layout.addWidget(side_on_video_btn)
        main_layout.addLayout(side_on_video_layout)

        # ── Middle section: log (left) + frame viewers (right, hidden until done) ──
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(16)

        # Log panel
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(6)

        log_label = QLabel("Processing Log")
        log_label_font = QFont()
        log_label_font.setPointSize(11)
        log_label_font.setBold(True)
        log_label.setFont(log_label_font)
        log_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4;"
            "font-family: Consolas, Monaco, monospace; font-size: 11px;"
            "border: 1px solid #555; padding: 6px;"
        )
        self.log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout.addWidget(self.log_display, stretch=1)

        clear_btn_layout = QHBoxLayout()
        clear_btn_layout.addStretch()
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(self.log_display.clear)
        clear_btn_layout.addWidget(clear_btn)
        log_layout.addLayout(clear_btn_layout)

        middle_layout.addWidget(log_widget, stretch=1)

        # Frame viewers (hidden until tracking completes)
        self.viewers_widget = QWidget()
        self.viewers_widget.setVisible(False)
        viewers_layout = QHBoxLayout(self.viewers_widget)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(16)

        self.top_down_viewer = FrameViewer("Top Down View")
        self.side_on_viewer  = FrameViewer("Side On View")

        # Vertical divider
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color: #ccc;")

        viewers_layout.addWidget(self.top_down_viewer)
        viewers_layout.addWidget(divider)
        viewers_layout.addWidget(self.side_on_viewer)

        middle_layout.addWidget(self.viewers_widget, stretch=2)

        main_layout.addLayout(middle_layout, stretch=1)

        # Start button
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.setFixedHeight(40)
        start_btn_font = QFont()
        start_btn_font.setPointSize(12)
        self.start_btn.setFont(start_btn_font)
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_tracking)
        main_layout.addWidget(self.start_btn)

        central_widget.setLayout(main_layout)

    # ── File selection ────────────────────────────────────────────────────────

    def select_top_down_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Top Down View Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if file_path:
            self.top_down_video_path = file_path
            self.top_down_video_display.setText(file_path.split("/")[-1])
            self.top_down_video_display.setStyleSheet(
                "background-color: white; padding: 10px; border: 1px solid #ccc;")

    def select_side_on_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Side On View Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if file_path:
            self.side_on_video_path = file_path
            self.side_on_video_display.setText(file_path.split("/")[-1])
            self.side_on_video_display.setStyleSheet(
                "background-color: white; padding: 10px; border: 1px solid #ccc;")

    # ── Tracking ──────────────────────────────────────────────────────────────

    def start_tracking(self):
        if not self.top_down_video_path or not self.side_on_video_path:
            self.top_down_video_display.setText("Please select both videos.")
            self.top_down_video_display.setStyleSheet(
                "background-color: #ffcccc; padding: 10px; border: 1px solid #ccc;")
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Processing…")
        self.viewers_widget.setVisible(False)
        self.log_display.clear()
        self.append_log("Starting tracking…")

        self._worker = TrackingWorker(
            self.top_down_video_path, self.side_on_video_path,
            self.top_down_tracker, self.side_on_tracker)
        self._worker.finished.connect(self.on_tracking_finished)
        self._worker.error.connect(self.on_tracking_error)
        self._worker.start()

    def on_tracking_finished(self, top_down_frames: list, side_on_frames: list):
        self.append_log("\nDone.")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Tracking")

        self.top_down_viewer.load_frames(top_down_frames)
        self.side_on_viewer.load_frames(side_on_frames)
        self.viewers_widget.setVisible(True)

        td = len(top_down_frames)
        so = len(side_on_frames)
        self.append_log(f"Showing {td} top-down frame{'s' if td != 1 else ''} and "
                        f"{so} side-on frame{'s' if so != 1 else ''}.")

    def on_tracking_error(self, message: str):
        self.append_log(f"\nERROR: {message}")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Tracking")

    # ── Log helpers ───────────────────────────────────────────────────────────

    def append_log(self, text: str):
        scrollbar = self.log_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.log_display.append(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    application = Application()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
