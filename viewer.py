import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QFont
import PyQt5
import sys

df = pd.read_excel("output_folder/new data.xlsx", sheet_name=None)
speed = df['Parameters'].loc[df['Parameters']['Parameter'] == 'Initial Speed (km/h)', 'Value'].values[0]
seam_angle = df['Parameters'].loc[df['Parameters']['Parameter'] == 'Seam Angle (degrees)', 'Value'].values[0]
trajectory = df['Parameters'].loc[df['Parameters']['Parameter'] == 'Initial Trajectory (degrees)', 'Value'].values[0]
swing = df['Parameters'].loc[df['Parameters']['Parameter'] == 'Final Swing (m)', 'Value'].values[0]

# print(f"Initial Speed: {speed}")
# print(f"Seam Angle: {seam_angle}")
# print(f"Initial Trajectory: {trajectory}")
# print(f"Final Swing: {swing}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Delivery Parameters Viewer")
        self.setFixedSize(1400, 800)

        # Prepare title/value pairs and layout them evenly across the window.
        items = [
            ("Initial Speed", f"{speed:.2f} km/h"),
            ("Seam Angle", f"{seam_angle:.2f} degrees"),
            ("Initial Trajectory", f"{trajectory:.2f} degrees"),
            ("Final Swing", f"{swing * 100:.2f} cm"),
        ]

        total_width = self.width()
        cols = len(items)
        column_width = total_width / cols

        title_font = QFont()
        title_font.setPointSize(10)

        value_font = QFont()
        value_font.setPointSize(14)
        value_font.setBold(True)

        top_y = 50
        title_height = 24
        value_height = 32

        self.labels = []
        padding = 12
        for i, (title, value) in enumerate(items):
            center_x = int((i + 0.5) * column_width)
            rect_w = int(column_width * 0.8)
            rect_h = title_height + value_height + padding * 2
            rect_x = center_x - rect_w // 2
            rect_y = top_y - padding // 2

            # Container widget with blue background
            container = QWidget(self)
            container.setGeometry(rect_x, rect_y, rect_w, rect_h)
            container.setStyleSheet("background-color: #1976D2; border-radius: 8px;")

            # Title label inside container
            title_label = PyQt5.QtWidgets.QLabel(container)
            title_label.setText(title)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setGeometry(padding, padding, rect_w - padding * 2, title_height)
            title_label.setStyleSheet("color: white; background: transparent;")

            # Value label inside container (below title)
            value_label = PyQt5.QtWidgets.QLabel(container)
            value_label.setText(value)
            value_label.setFont(value_font)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setGeometry(padding, padding + title_height, rect_w - padding * 2, value_height)
            value_label.setStyleSheet("color: white; background: transparent;")

            self.labels.append((container, title_label, value_label))


app = QApplication([])

window = MainWindow()
window.show()

app.exec()