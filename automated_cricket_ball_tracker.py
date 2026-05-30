import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import automations

class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_video_path = None
        self.side_video_path = None
        self.init_ui()

        self.top_down_tracker = automations.TopDownBallFinder()
        self.side_on_tracker = automations.SideOnBallFinder()

    def init_ui(self):
        self.setWindowTitle("Cricket Ball Tracker")
        self.setFixedSize(1600, 800)
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

        # Main video section
        top_down_video_layout = QHBoxLayout()
        top_down_video_label = QLabel("Top Down View:")
        top_down_video_label.setMinimumWidth(150)
        self.top_down_video_display = QLabel("No file selected")
        self.top_down_video_display.setStyleSheet("background-color: white; padding: 10px; border: 1px solid #ccc;")
        top_down_video_btn = QPushButton("Browse")
        top_down_video_btn.clicked.connect(self.select_top_down_video)
        top_down_video_btn.setFixedWidth(100)
        top_down_video_layout.addWidget(top_down_video_label)
        top_down_video_layout.addWidget(self.top_down_video_display)
        top_down_video_layout.addWidget(top_down_video_btn)
        main_layout.addLayout(top_down_video_layout)

        # Side video section
        side_on_video_layout = QHBoxLayout()
        side_on_video_label = QLabel("Side On View:")
        side_on_video_label.setMinimumWidth(150)
        self.side_on_video_display = QLabel("No file selected")
        self.side_on_video_display.setStyleSheet("background-color: white; padding: 10px; border: 1px solid #ccc;")
        side_on_video_btn = QPushButton("Browse")
        side_on_video_btn.clicked.connect(self.select_side_on_video)
        side_on_video_btn.setFixedWidth(100)
        side_on_video_layout.addWidget(side_on_video_label)
        side_on_video_layout.addWidget(self.side_on_video_display)
        side_on_video_layout.addWidget(side_on_video_btn)
        main_layout.addLayout(side_on_video_layout)

        main_layout.addStretch()

        # Start button
        start_btn = QPushButton("Start Tracking")
        start_btn.setFixedHeight(40)
        start_btn_font = QFont()
        start_btn_font.setPointSize(12)
        start_btn.setFont(start_btn_font)
        start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        start_btn.clicked.connect(self.start_tracking)
        main_layout.addWidget(start_btn)

        central_widget.setLayout(main_layout)

    def select_top_down_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main View Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.top_down_video_path = file_path
            filename = file_path.split("/")[-1]
            self.top_down_video_display.setText(filename)

    def select_side_on_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Side On View Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.side_on_video_path = file_path
            filename = file_path.split("/")[-1]
            self.side_on_video_display.setText(filename)

    def start_tracking(self):
        if not self.top_down_video_path or not self.side_on_video_path:
            self.top_down_video_display.setText("Please select both videos.")
            self.top_down_video_display.setStyleSheet("background-color: #ffcccc; padding: 10px; border: 1px solid #ccc;")
            return
        
        top_down_video = automations.Video(self.top_down_video_path)
        side_on_video = automations.Video(self.side_on_video_path)

        top_down_ball_data = self.top_down_tracker.get_ball_data(top_down_video)
        side_on_ball_data = self.side_on_tracker.get_ball_data(side_on_video)

        print("\nTracking Results:")

        for data_point in top_down_ball_data:
            print(f"Top Down - Frame {data_point[0]}: Seam Angle = {data_point[1]:.2f} degrees")
        
        for data_point in side_on_ball_data:
            print(f"Side On - Frame {data_point[0]}: Ball Position = ({data_point[1].x}, {data_point[1].y})")

        

def main():
    app = QApplication(sys.argv)
    application = Application()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
