import sys
import os
import glob
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QSizePolicy, QFrame,
    QDialog, QDialogButtonBox, QScrollArea, QRubberBand, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint, QSize
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor
import cv2
import yaml
import automations
from log_bridge import bridge

CONFIG_PATH = "config.yml"


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_bounding_box(view: str, x_start: int, y_start: int, x_end: int, y_end: int):
    """Write bounding box coords for 'top_down' or 'side_on' into config.yml."""
    config = load_config()
    if "bounding_boxes" not in config:
        config["bounding_boxes"] = {}
    config["bounding_boxes"][view] = {
        "top_left":     {"x": x_start, "y": y_start},
        "bottom_right": {"x": x_end,   "y": y_end},
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_bounding_box(view: str):
    """Return (x_start, y_start, x_end, y_end) or None if not set."""
    config = load_config()
    bb = config.get("bounding_boxes", {}).get(view, {})
    tl = bb.get("top_left", {})
    br = bb.get("bottom_right", {})
    xs, ys = tl.get("x"), tl.get("y")
    xe, ye = br.get("x"), br.get("y")
    if None in (xs, ys, xe, ye):
        return None
    return xs, ys, xe, ye


# ── Bounding Box Dialog ───────────────────────────────────────────────────────

class SelectableImageLabel(QLabel):
    """QLabel that lets the user draw a rectangle by click-and-drag."""

    selection_changed = pyqtSignal(QRect)   # emits rect in *label* coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._origin = QPoint()
        self._rect   = QRect()
        self._drawing = False
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._origin  = event.pos()
            self._rect    = QRect(self._origin, QSize())
            self._drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._rect = QRect(self._origin, event.pos()).normalized()
            self.update()
            self.selection_changed.emit(self._rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._rect    = QRect(self._origin, event.pos()).normalized()
            self._drawing = False
            self.update()
            self.selection_changed.emit(self._rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._rect.isNull():
            painter = QPainter(self)
            pen = QPen(QColor(0, 220, 100), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._rect)
            fill = QColor(0, 220, 100, 40)
            painter.fillRect(self._rect, fill)

    def clear_selection(self):
        self._rect = QRect()
        self.update()

    @property
    def selection_rect(self) -> QRect:
        return self._rect


class BoundingBoxDialog(QDialog):
    """
    Dialog for setting the bounding box for one camera view.

    view_key: 'top_down' or 'side_on'
    """

    def __init__(self, view_key: str, parent=None):
        super().__init__(parent)
        self.view_key     = view_key
        self._orig_pixmap = None   # full-res pixmap of the loaded frame
        self._img_rect    = QRect()  # where the pixmap is actually drawn inside the label

        title = "Top Down" if view_key == "top_down" else "Side On"
        self.setWindowTitle(f"Set Bounding Box – {title} View")
        self.setMinimumSize(900, 680)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Instructions
        info = QLabel(
            "Load a video or image, then click and drag on the frame to draw the "
            "search region. Press Confirm to save."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(info)

        # Load button row
        load_row = QHBoxLayout()
        load_btn = QPushButton("Load Video / Image…")
        load_btn.setFixedHeight(32)
        load_btn.clicked.connect(self._load_file)
        load_row.addWidget(load_btn)

        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.setFixedHeight(32)
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_selection)
        load_row.addWidget(self.clear_btn)

        load_row.addStretch()

        self.coords_label = QLabel("No selection")
        self.coords_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #333;")
        load_row.addWidget(self.coords_label)

        layout.addLayout(load_row)

        # Image area
        self.img_label = SelectableImageLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumHeight(480)
        self.img_label.setText("No image loaded")
        self.img_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #555; color: #666;")
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self.img_label, stretch=1)

        # Show existing bounding box if set
        existing = get_bounding_box(view_key)
        if existing:
            xs, ys, xe, ye = existing
            self.coords_label.setText(
                f"Current: ({xs}, {ys}) → ({xe}, {ye})")

        # Dialog buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("Confirm")
        btn_box.accepted.connect(self._confirm)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── file loading ──────────────────────────────────────────────────────────

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Video or Image", "",
            "Video / Image Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                QMessageBox.warning(self, "Error", "Could not read first frame from video.")
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        else:
            qimg = QImage(path)
            if qimg.isNull():
                QMessageBox.warning(self, "Error", "Could not load image.")
                return

        self._orig_pixmap = QPixmap.fromImage(qimg)
        self._update_display()
        self.img_label.clear_selection()
        self.clear_btn.setEnabled(True)
        self.coords_label.setText("Draw a rectangle on the image")

    def _update_display(self):
        if self._orig_pixmap is None:
            return
        scaled = self._orig_pixmap.scaled(
            self.img_label.width(),
            self.img_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        # Record where the image actually sits inside the label (centred)
        lw, lh = self.img_label.width(), self.img_label.height()
        iw, ih = scaled.width(), scaled.height()
        ox = (lw - iw) // 2
        oy = (lh - ih) // 2
        self._img_rect = QRect(ox, oy, iw, ih)
        self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    # ── selection ─────────────────────────────────────────────────────────────

    def _clear_selection(self):
        self.img_label.clear_selection()
        self.coords_label.setText("Selection cleared")

    def _on_selection_changed(self, rect: QRect):
        """Convert label-space rect to original-image-space coords and display."""
        orig_rect = self._label_rect_to_image_rect(rect)
        if orig_rect is None:
            return
        self.coords_label.setText(
            f"({orig_rect.left()}, {orig_rect.top()}) → "
            f"({orig_rect.right()}, {orig_rect.bottom()})  "
            f"[{orig_rect.width()} × {orig_rect.height()} px]"
        )

    def _label_rect_to_image_rect(self, label_rect: QRect):
        """Map a rect in label pixel coords → original image pixel coords."""
        if self._orig_pixmap is None or self._img_rect.isNull():
            return None
        if self._img_rect.width() == 0 or self._img_rect.height() == 0:
            return None

        scale_x = self._orig_pixmap.width()  / self._img_rect.width()
        scale_y = self._orig_pixmap.height() / self._img_rect.height()

        # Clamp to the image area
        r = label_rect.intersected(self._img_rect)
        x1 = int((r.left()   - self._img_rect.left()) * scale_x)
        y1 = int((r.top()    - self._img_rect.top())  * scale_y)
        x2 = int((r.right()  - self._img_rect.left()) * scale_x)
        y2 = int((r.bottom() - self._img_rect.top())  * scale_y)

        x1 = max(0, min(x1, self._orig_pixmap.width()))
        y1 = max(0, min(y1, self._orig_pixmap.height()))
        x2 = max(0, min(x2, self._orig_pixmap.width()))
        y2 = max(0, min(y2, self._orig_pixmap.height()))

        return QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()

    # ── confirm ───────────────────────────────────────────────────────────────

    def _confirm(self):
        sel = self.img_label.selection_rect
        if sel.isNull() or sel.width() < 5 or sel.height() < 5:
            QMessageBox.information(
                self, "No Selection",
                "Please draw a bounding box on the image before confirming.")
            return

        orig_rect = self._label_rect_to_image_rect(sel)
        if orig_rect is None:
            QMessageBox.warning(self, "Error", "Could not map selection to image coordinates.")
            return

        save_bounding_box(
            self.view_key,
            orig_rect.left(), orig_rect.top(),
            orig_rect.right(), orig_rect.bottom(),
        )
        self.accept()


# ── Worker ────────────────────────────────────────────────────────────────────

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
            # Clear output folders from any previous run
            for folder in (automations.top_down_tracking_folder, automations.side_on_tracking_folder):
                if os.path.isdir(folder):
                    for f in glob.glob(os.path.join(folder, "*")):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                else:
                    os.makedirs(folder, exist_ok=True)

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

            top_down_frames = sorted(glob.glob(f"{automations.top_down_tracking_folder}/frame_*.jpg"))
            side_on_frames  = sorted(glob.glob(f"{automations.side_on_tracking_folder}/frame_*.jpg"))

            self.finished.emit(top_down_frames, side_on_frames)
        except Exception as e:
            self.error.emit(str(e))


# ── Frame Viewer ──────────────────────────────────────────────────────────────

class FrameViewer(QWidget):
    """A labelled image viewer with prev/next navigation."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._frames: list[str] = []
        self._index: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(title)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        lbl.setFont(font)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 340)
        self.image_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #555;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)

        self.prev_btn = QPushButton("◀  Prev")
        self.prev_btn.setFixedHeight(32)
        self.prev_btn.clicked.connect(self.prev_frame)

        self.counter_label = QLabel("-")
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
        self.counter_label.setText("-")
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


# ── Main Window ───────────────────────────────────────────────────────────────

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

        # ── File pickers + bounding box buttons ──────────────────────────────
        # Top down row
        top_down_video_layout = QHBoxLayout()
        top_down_video_label = QLabel("Top Down View:")
        top_down_video_label.setMinimumWidth(150)
        self.top_down_video_display = QLabel("No file selected")
        self.top_down_video_display.setStyleSheet(
            "background-color: white; padding: 10px; border: 1px solid #ccc;")
        top_down_video_btn = QPushButton("Browse")
        top_down_video_btn.clicked.connect(self.select_top_down_video)
        top_down_video_btn.setFixedWidth(100)

        self.top_down_bb_btn = QPushButton("Set Search Region…")
        self.top_down_bb_btn.setFixedWidth(160)
        self.top_down_bb_btn.setToolTip("Draw a bounding box to restrict where the ball is searched for")
        self.top_down_bb_btn.clicked.connect(lambda: self._open_bbox_dialog("top_down"))
        self._style_bb_button(self.top_down_bb_btn, "top_down")

        top_down_video_layout.addWidget(top_down_video_label)
        top_down_video_layout.addWidget(self.top_down_video_display)
        top_down_video_layout.addWidget(top_down_video_btn)
        top_down_video_layout.addWidget(self.top_down_bb_btn)
        main_layout.addLayout(top_down_video_layout)

        # Side on row
        side_on_video_layout = QHBoxLayout()
        side_on_video_label = QLabel("Side On View:")
        side_on_video_label.setMinimumWidth(150)
        self.side_on_video_display = QLabel("No file selected")
        self.side_on_video_display.setStyleSheet(
            "background-color: white; padding: 10px; border: 1px solid #ccc;")
        side_on_video_btn = QPushButton("Browse")
        side_on_video_btn.clicked.connect(self.select_side_on_video)
        side_on_video_btn.setFixedWidth(100)

        self.side_on_bb_btn = QPushButton("Set Search Region…")
        self.side_on_bb_btn.setFixedWidth(160)
        self.side_on_bb_btn.setToolTip("Draw a bounding box to restrict where the ball is searched for")
        self.side_on_bb_btn.clicked.connect(lambda: self._open_bbox_dialog("side_on"))
        self._style_bb_button(self.side_on_bb_btn, "side_on")

        side_on_video_layout.addWidget(side_on_video_label)
        side_on_video_layout.addWidget(self.side_on_video_display)
        side_on_video_layout.addWidget(side_on_video_btn)
        side_on_video_layout.addWidget(self.side_on_bb_btn)
        main_layout.addLayout(side_on_video_layout)

        # ── Middle: log + frame viewers ───────────────────────────────────────
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(16)

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

        self.viewers_widget = QWidget()
        self.viewers_widget.setVisible(False)
        viewers_layout = QHBoxLayout(self.viewers_widget)
        viewers_layout.setContentsMargins(0, 0, 0, 0)
        viewers_layout.setSpacing(16)

        self.top_down_viewer = FrameViewer("Top Down View")
        self.side_on_viewer  = FrameViewer("Side On View")

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

    # ── Bounding box helpers ──────────────────────────────────────────────────

    def _style_bb_button(self, btn: QPushButton, view_key: str):
        """Green tint if a bounding box is already saved, neutral otherwise."""
        bb = get_bounding_box(view_key)
        if bb:
            xs, ys, xe, ye = bb
            btn.setStyleSheet(
                "background-color: #c8f0cc; border: 1px solid #4CAF50; padding: 4px;")
            btn.setToolTip(
                f"Search region set: ({xs}, {ys}) → ({xe}, {ye})\nClick to change.")
        else:
            btn.setStyleSheet("")
            btn.setToolTip("Draw a bounding box to restrict where the ball is searched for")

    def _open_bbox_dialog(self, view_key: str):
        dlg = BoundingBoxDialog(view_key, self)
        if dlg.exec_() == QDialog.Accepted:
            # Refresh button style to reflect newly saved box
            btn = self.top_down_bb_btn if view_key == "top_down" else self.side_on_bb_btn
            self._style_bb_button(btn, view_key)
            bb = get_bounding_box(view_key)
            if bb:
                xs, ys, xe, ye = bb
                label = "Top Down" if view_key == "top_down" else "Side On"
                self.append_log(
                    f"{label} search region saved: ({xs}, {ys}) → ({xe}, {ye})")

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