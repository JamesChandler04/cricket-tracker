import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import automations
from log_bridge import bridge


class TrackingWorker(QThread):
    """
    Runs the ball-tracking in a background thread so the UI stays responsive.
    All print() calls inside automations.py flow through the redirected stdout
    and arrive at the log panel via the bridge signal.
    """
    finished = pyqtSignal()
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
            for data_point in top_down_ball_data:
                print(f"Top Down - Frame {data_point.frame_number}: Ball Position = {data_point.data.centre}, Seam Angle = {data_point.data.seam_angle:.2f} degrees")
            for data_point in side_on_ball_data:
                print(f"Side On - Frame {data_point.frame_number}: Ball Position = {data_point.data.centre}")

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.top_down_video_path = None
        self.side_on_video_path = None
        self._worker = None

        # Redirect stdout to the bridge before any automations code runs
        sys.stdout = bridge
        bridge.message_received.connect(self.append_log)

        self.top_down_tracker = automations.TopDownBallFinder()
        self.side_on_tracker = automations.SideOnBallFinder()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Cricket Ball Tracker")
        self.setFixedSize(1600, 900)
        self.setStyleSheet("background-color: #f0f0f0;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("Upload Cricket Videos")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)

        # Top-down video section
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

        # Side-on video section
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

        # Logging section
        log_label = QLabel("Processing Log")
        log_label_font = QFont()
        log_label_font.setPointSize(11)
        log_label_font.setBold(True)
        log_label.setFont(log_label_font)
        main_layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4;"
            "font-family: Consolas, Monaco, monospace; font-size: 11px;"
            "border: 1px solid #555; padding: 6px;"
        )
        self.log_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_display, stretch=1)

        # Clear log button (small, right-aligned)
        clear_btn_layout = QHBoxLayout()
        clear_btn_layout.addStretch()
        clear_btn = QPushButton("Clear Log")
        clear_btn.setFixedWidth(100)
        clear_btn.clicked.connect(self.log_display.clear)
        clear_btn_layout.addWidget(clear_btn)
        main_layout.addLayout(clear_btn_layout)

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

    # Select video file dialogs

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

    def start_tracking(self):
        if not self.top_down_video_path or not self.side_on_video_path:
            self.top_down_video_display.setText("Please select both videos.")
            self.top_down_video_display.setStyleSheet(
                "background-color: #ffcccc; padding: 10px; border: 1px solid #ccc;")
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Processing...")
        self.log_display.clear()
        self.append_log("Starting tracking...")

        self._worker = TrackingWorker(
            self.top_down_video_path, self.side_on_video_path,
            self.top_down_tracker, self.side_on_tracker)
        self._worker.finished.connect(self.on_tracking_finished)
        self._worker.error.connect(self.on_tracking_error)
        self._worker.start()

    # Logging and UI updates

    def on_tracking_finished(self):
        self.append_log("\nDone.")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Tracking")

    def on_tracking_error(self, message: str):
        self.append_log(f"\nERROR: {message}")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start Tracking")

    def append_log(self, text: str):
        scrollbar = self.log_display.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.log_display.append(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

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
