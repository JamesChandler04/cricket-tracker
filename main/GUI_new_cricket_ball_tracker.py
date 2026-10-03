"""The cricket ball tracker's main program: a PyQt5 window that calibrates the side-on
camera and runs the top-down stage, the seam angle step (manual or automatic) and the
side-on stage on worker threads.

Results are saved to a chosen folder and shown in the Log, Swing, Trajectory, Results
and Coordinates tabs.
"""

from __future__ import annotations

import builtins
import sys
import threading
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):
    # Started as a plain script (python main/<name>.py, or the Run button in
    # VS Code), so put the project folder on the path for the library imports.
    # python -m main.<name> from the project folder does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import camera_calibration, paths
from library.automated.automatic_seam_angle import AutomaticSeamAngleDetector, AveragedSeamMeasurement
from library.detect_seam_angle_file import (ManualSeamAngleDetector, SeamAngleDetector,
                                            save_ball_images, save_seam_measurement)
from library.helpers import Video
from library.log_bridge import bridge
from library.manual_tracker import SideOnTracker, TopDownTracker
from library.physics_engines.side_on_physics_engine import SideOnPhysicsEngine
from library.physics_engines.top_down_physics_engine import SelectionType, TopDownPhysicsEngine
from library.results import ResultRow, load_coordinates, load_results


SAVE_DIR = str(paths.DELIVERY_DIR)
"""Save folder the window starts with."""
TOP_DOWN_VALUE_FILE = "top_down_analysis"
"""File name of the top-down results in the save folder, without the .yaml extension."""
SWING_PLOT_FILE = "swing"
"""File name of the swing plot in the save folder; matplotlib adds the .png."""
TRAJECTORY_PLOT_FILE = "trajectory_3d"
"""File name of the 3D trajectory plot in the save folder; matplotlib adds the .png."""
CALIBRATION_PATH = str(paths.CALIBRATION_PATH)
"""Side-on camera calibration file the window starts with."""

VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
"""File-type filter for the video file dialogs."""
CALIBRATION_FILTER = "Calibration files (*.npz);;All files (*.*)"
"""File-type filter for the calibration file dialog."""

MONOSPACE = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 10)
"""Fixed-width font for the log and the numbers (Menlo on macOS, Consolas elsewhere)."""


def resolve_plot_path(save_path):
    """
    Return the file the plot methods actually wrote.

    They hand save_path straight to fig.savefig, and matplotlib appends .png
    when the path carries no extension, so the returned path can name a file
    that is not on disk.
    """
    candidate = Path(save_path)
    if candidate.exists():
        return candidate
    with_png = candidate.with_name(candidate.name + ".png")
    return with_png if with_png.exists() else candidate


class PromptRouter(QObject):
    """
    Answers input() raised anywhere in the pipeline.
    Overrides builtins.input, if no input then opens a dialog on the GUI.
    """

    prompt_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        """Set up an empty script and history, queueing dialogs to the GUI thread."""
        super().__init__(parent)
        self._scripted: list[tuple[str, str]] = []
        self._original_input = None
        self.history: list[tuple[str, str]] = []
        self.prompt_requested.connect(self._serve, Qt.QueuedConnection)

    def install(self):
        """Take over builtins.input."""
        if self._original_input is None:
            self._original_input = builtins.input
            builtins.input = self

    def remove(self):
        """Hand builtins.input back."""
        if self._original_input is not None:
            builtins.input = self._original_input
            self._original_input = None

    def script(self, needle, answer):
        """Answer the next prompt containing needle with answer, without asking."""
        self._scripted.append((needle.lower(), str(answer)))

    def clear_script(self):
        """Drop any scripted answers that were not used."""
        self._scripted.clear()

    def __call__(self, prompt=""):
        """Answer an input() prompt from the script, or else by asking in a dialog."""
        text = str(prompt)
        answer = self._take_scripted(text)
        if answer is None:
            answer = self._ask(text)
        else:
            print(f"{text.strip()} -> {answer!r}")
        self.history.append((text, answer))
        return answer

    def _take_scripted(self, prompt):
        """Pop the first scripted answer whose needle is in prompt, or return None."""
        lowered = prompt.lower()
        for index, (needle, answer) in enumerate(self._scripted):
            if needle in lowered:
                self._scripted.pop(index)
                return answer
        return None

    def _ask(self, prompt):
        """Return the answer from a GUI-thread dialog; raise SystemExit if cancelled."""
        request = {"prompt": prompt, "answer": "", "ok": False,
                   "event": threading.Event()}
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            self._serve(request)
        else:
            self.prompt_requested.emit(request)
            request["event"].wait()
        if not request["ok"]:
            raise SystemExit("Cancelled at a prompt.")
        return request["answer"]

    def _serve(self, request):
        """Open the dialog. Always runs on the GUI thread."""
        try:
            label = request["prompt"].strip() or "Input required"
            text, ok = QInputDialog.getText(None, "The pipeline is asking", label)
            request["answer"], request["ok"] = text, ok
        finally:
            request["event"].set()


class ScriptedDisplay:
    """
    Stand-in for display.Display that hands back videos chosen in the GUI.

    Display opens a tkinter dialog for the file and asks for the start frame
    through input(). Both are wrong from a worker thread, so a tracker's
    .display is replaced with one of these before it runs. The interface is the
    two methods the trackers actually call.
    """

    def __init__(self, main_path="", main_start=0, side_path="", side_start=0):
        """Set up the chosen top-down and side-on video paths and their start frames."""
        self.main_path = main_path
        self.main_start = int(main_start)
        self.side_path = side_path
        self.side_start = int(side_start)

    def load_main_video(self) -> Video:
        """Return the chosen top-down video, at its start frame."""
        return self._open(self.main_path, self.main_start, "top-down")

    def load_side_video(self) -> Video:
        """Return the chosen side-on video, at its start frame."""
        return self._open(self.side_path, self.side_start, "side-on")

    @staticmethod
    def _open(path, start_frame, label) -> Video:
        """Return the video at path set to start_frame; raise ValueError if no path."""
        if not path:
            raise ValueError(f"No {label} video was chosen.")
        print(f"Loading {label} video: {path}")
        video = Video(path)
        video.current_frame = start_frame
        return video


class TaskThread(QThread):
    """
    Runs one pipeline step off the GUI thread.

    SystemExit is how the pipeline bails out of a cv2 loop, so it is reported
    as a cancellation rather than allowed to tear the process down. Anything
    else comes back as a traceback the log panel can show.
    """

    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)

    def __init__(self, step, parent=None):
        """Set up the thread to run step, a function with no arguments."""
        super().__init__(parent)
        self._step = step

    def run(self):
        """Run the step and emit done, cancelled or failed, depending on how it ends."""
        try:
            self.done.emit(self._step())
        except SystemExit as exit_signal:
            self.cancelled.emit(str(exit_signal) or "Cancelled.")
        except Exception:
            self.failed.emit(traceback.format_exc())


def track_top_down(display, save_dir, value_file, seam_detector: SeamAngleDetector):
    """
    Click the top-down video through and take the velocity from it, then save
    the ball images and let seam_detector find the seam angle on them.
    """
    print("The seam angle is picked from the ball images after tracking, "
          "so T (seam mode) is not needed in the top-down window.")
    tracker = TopDownTracker()
    tracker.display = display
    points = tracker.get_top_down_points()

    engine = TopDownPhysicsEngine()
    video = tracker.top_down_video
    fps = video.fps

    velocity, velocities = engine.calculate_velocity(points, fps=fps, type=SelectionType.MEAN)
    print(f"Calculated velocity: {velocity:.2f} km/h")
    print("Per-frame velocities: " + ", ".join(f"{v:.1f}" for v in velocities) + " km/h")

    # Seam angle step. The crops are taken from the frames the clicks were made
    # on, so they use the same video and the same rotation as the tracker.
    ball_images = save_ball_images(display.main_path, points, save_dir,
                                   rotation=video.rotation)
    seam = seam_detector.detect(ball_images)
    seam_angle = None if seam is None else seam.seam_angle_deg
    seam_warning = None
    seam_source: str | None
    if isinstance(seam, AveragedSeamMeasurement):
        seam_source = f"average of {len(seam.frame_angles_deg)} frames, spread {seam.spread_deg:.1f} deg"
        seam_warning = seam.warning
    else:
        seam_source = None if seam is None else f"frame {seam.frame_number}"
    if seam is not None:
        print(f"Seam saved to {save_seam_measurement(seam)}")
    print(f"Calculated seam angle ({seam_detector.method}): "
          + ("not set" if seam_angle is None else f"{seam_angle:+.2f} deg ({seam_source})"))

    path = engine.save_top_down_analysis(save_dir, value_file, velocity, seam_angle,
                                         fps, len(points))
    print(f"Top down values saved to {path}")

    return {"velocity": float(velocity), "seam_angle": seam_angle,
            "seam_source": seam_source, "seam_warning": seam_warning,
            "fps": float(fps), "point_count": len(points), "path": str(path)}


def track_side_on(display, velocity_km_h, calibration_path, save_dir):
    """Click the side-on video through, reconstruct it and run the swing analysis."""
    tracker = SideOnTracker()
    tracker.display = display
    points = tracker.get_side_on_points()

    fps = tracker.side_on_video.fps
    engine = SideOnPhysicsEngine.from_calibration_file(calibration_path, fps=fps)
    calibration = engine.calibration

    print(f"calibration: focal {calibration.focal_px[0]:.1f} px, "
          f"principal point {calibration.principal_point[0]:.1f},"
          f"{calibration.principal_point[1]:.1f}")
    print(f"camera centre (X, Y, Z) = {np.round(calibration.camera_centre, 3)}")
    print(f"{len(points)} tracked points, frames "
          f"{points[0].frame}-{points[-1].frame}\n")

    trajectory = engine.reconstruct_trajectory(points, velocity_km_h)

    print(f"swing_at_last_tracked_point : "
          f"{engine.swing_at_last_tracked_point(trajectory):+.2f} cm")
    print(f"swing_at_17m                : "
          f"{engine.swing_at_17m(trajectory):+.2f} cm")

    cubes = engine.ball_coordinates(trajectory, origin="cubes")
    release = engine.ball_coordinates(trajectory, origin="release")
    print(f"\n{'frame':>6} {'X':>8} {'Y':>8} {'Z':>8}   "
          f"{'dX':>8} {'dY':>8} {'dZ':>8}")
    for i in (0, len(cubes) // 2, len(cubes) - 1):
        frame = trajectory.frames[i]
        print(f"{frame:>6} {cubes[i, 0]:>8.3f} {cubes[i, 1]:>8.3f} {cubes[i, 2]:>8.3f}   "
              f"{release[i, 0]:>8.3f} {release[i, 1]:>8.3f} {release[i, 2]:>8.3f}")

    result = engine.analyse(points, velocity_km_h)
    print("\n" + result.summary())

    written = engine.save_data_to_files(result, save_dir, points=points)
    print("\nwrote " + "\nwrote ".join(written.values()))

    return {"engine": engine, "result": result, "points": points,
            "written": written, "fps": float(fps)}


def run_calibration(video_path):
    """Drive camera_calibration.main. The prompts inside it are already scripted."""
    camera_calibration.main(video_path)


class ImageView(QLabel):
    """A QLabel that keeps a plot readable as the panel is resized."""

    def __init__(self, placeholder):
        """Set up the view with grey placeholder text until a plot is shown."""
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("color: #888780;")
        self._source = None

    def set_image(self, path):
        """Show the image at path scaled to fit, or a message if it cannot be loaded."""
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._source = None
            self.setText(f"Could not load {path}")
            return
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event):
        """Rescale the image to the new size."""
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        """Scale the image to fit the view, keeping its aspect ratio."""
        if self._source is not None:
            self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))


class ElidedLabel(QLabel):
    """One line of text, shortened in the middle when it does not fit, with all of it as the tooltip."""

    def __init__(self):
        """Set up an empty grey label whose width does not depend on its text."""
        super().__init__()
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setStyleSheet("color: #888780;")

    def set_full_text(self, text):
        """Set the full text and tooltip, and show as much of the text as fits."""
        self._full_text = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event):
        """Shorten the text again to fit the new width."""
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        """Show the full text, shortened in the middle to fit the label's width."""
        self.setText(self.fontMetrics().elidedText(self._full_text, Qt.ElideMiddle, max(self.width(), 1)))


class MainWindow(QMainWindow):
    """Controls on the left, log and plots on the right."""

    def __init__(self):
        """Set up the window, send input() prompts to dialogs and printed text to the
        log, and show any results already in the save folder.
        """
        super().__init__()
        self.setWindowTitle("Cricket ball tracker")
        self.resize(1360, 860)

        self.router = PromptRouter(self)
        self.router.install()

        self.task = None
        self.engine = None
        self.result = None
        self.side_on_data = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 960])
        self.setCentralWidget(splitter)

        sys.stdout = bridge
        bridge.message_received.connect(self.append_log)

        self.save_dir.editingFinished.connect(self.load_results)
        self.load_results()

        self.statusBar().showMessage("Ready.")
        self.append_log("Pick both videos and a calibration file, then track.")

    # ------------------------------------------------------------------ build

    def _build_controls(self):
        """Build the left panel of inputs, step buttons and result readouts."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        inputs = QGroupBox("Inputs")
        form = QFormLayout(inputs)

        self.top_down_path = QLineEdit()
        form.addRow("Top-down video", self._with_browse(self.top_down_path,
                                                        self.browse_top_down))
        self.top_down_start = QSpinBox()
        self.top_down_start.setRange(0, 10_000_000)
        form.addRow("Start frame", self.top_down_start)

        self.side_on_path = QLineEdit()
        form.addRow("Side-on video", self._with_browse(self.side_on_path,
                                                       self.browse_side_on))
        self.side_on_start = QSpinBox()
        self.side_on_start.setRange(0, 10_000_000)
        form.addRow("Start frame", self.side_on_start)

        self.calibration_path = QLineEdit(CALIBRATION_PATH)
        form.addRow("Calibration", self._with_browse(self.calibration_path,
                                                     self.browse_calibration))

        self.save_dir = QLineEdit(SAVE_DIR)
        form.addRow("Save to", self._with_browse(self.save_dir, self.browse_save_dir))

        self.velocity = QDoubleSpinBox()
        self.velocity.setRange(0.0, 250.0)
        self.velocity.setDecimals(2)
        self.velocity.setSuffix(" km/h")
        self.velocity.setToolTip("Filled in by the top-down stage. Override it here "
                                 "to run the side-on stage on its own.")
        form.addRow("Delivery speed", self.velocity)

        self.automatic_seam = QCheckBox("Automatic seam angle detection")
        self.automatic_seam.setToolTip("Ticked: automatic_seam_angle.py finds the seam on "
                                       "the ball images.\nUnticked: click the seam yourself "
                                       "in the seam angle window.")
        form.addRow("", self.automatic_seam)

        self.show_3d = QCheckBox("Open the 3D plot in its own window")
        form.addRow("", self.show_3d)
        layout.addWidget(inputs)

        steps = QGroupBox("Steps")
        buttons = QVBoxLayout(steps)
        self.calibrate_button = QPushButton("Calibrate camera...")
        self.top_down_button = QPushButton("1 - Track top-down")
        self.side_on_button = QPushButton("2 - Track side-on")
        self.both_button = QPushButton("Run both")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.top_down_button.clicked.connect(self.start_top_down)
        self.side_on_button.clicked.connect(self.start_side_on)
        self.both_button.clicked.connect(self.start_both)
        for button in (self.calibrate_button, self.top_down_button,
                       self.side_on_button, self.both_button):
            buttons.addWidget(button)
        self.step_buttons = (self.calibrate_button, self.top_down_button,
                             self.side_on_button, self.both_button)
        layout.addWidget(steps)

        results = QGroupBox("Results")
        grid = QGridLayout(results)
        self.readouts = {}
        rows = ("Velocity", "Seam angle", "Tracked points",
                "Swing at last point", "Swing at", "Baseline residual",
                "Projection residual")
        for row, name in enumerate(rows):
            grid.addWidget(QLabel(name), row, 0)
            value = QLabel("-")
            value.setFont(MONOSPACE)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(value, row, 1)
            self.readouts[name] = value
        layout.addWidget(results)

        layout.addStretch(1)
        return panel

    @staticmethod
    def _with_browse(line_edit, handler):
        """Return line_edit in a row with a small browse button that calls handler."""
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(line_edit)
        button = QPushButton("...")
        button.setFixedWidth(32)
        button.clicked.connect(handler)
        box.addWidget(button)
        return row

    def _build_views(self):
        """Build the Log, Swing, Trajectory, Results and Coordinates tabs."""
        self.tabs = QTabWidget()

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(MONOSPACE)
        self.log.setMaximumBlockCount(10_000)
        self.tabs.addTab(self.log, "Log")

        self.swing_view = ImageView("The swing plot appears here after the side-on stage.")
        self.tabs.addTab(self.swing_view, "Swing")

        trajectory_tab = QWidget()
        box = QVBoxLayout(trajectory_tab)
        self.trajectory_view = ImageView("The 3D trajectory appears here after the "
                                         "side-on stage.")
        box.addWidget(self.trajectory_view)
        self.open_3d_button = QPushButton("Open the interactive 3D view")
        self.open_3d_button.setEnabled(False)
        self.open_3d_button.clicked.connect(self.open_interactive_3d)
        box.addWidget(self.open_3d_button)
        self.tabs.addTab(trajectory_tab, "Trajectory")

        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_coordinates_tab(), "Coordinates")
        return self.tabs

    def _build_results_tab(self):
        """The numbers saved in the YAML files of the save folder."""
        tab = QWidget()
        box = QVBoxLayout(tab)
        bar = QHBoxLayout()
        self.results_folder = ElidedLabel()
        bar.addWidget(self.results_folder, 1)
        reload_button = QPushButton("Reload")
        reload_button.setToolTip("Read the files in the save folder again.")
        reload_button.clicked.connect(lambda: self.load_results())
        bar.addWidget(reload_button)
        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy the selected rows, or all of them, to paste into Excel.")
        copy_button.clicked.connect(self.copy_results)
        bar.addWidget(copy_button)
        box.addLayout(bar)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Quantity", "Value"])
        self.results_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.results_tree.setAlternatingRowColors(True)
        box.addWidget(self.results_tree)
        shortcut = QShortcut(QKeySequence.Copy, self.results_tree, self.copy_results)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        return tab

    def _build_coordinates_tab(self):
        """X, Y, Z and dX, dY, dZ for every tracked frame, from tracked_points.csv."""
        tab = QWidget()
        box = QVBoxLayout(tab)
        note = QLabel("Ball position on each tracked side-on frame. X, Y and Z are measured from "
                      "the bottom-left corner of the nearest calibration ring (X to the bowler's "
                      "right, Y down the pitch, Z up). dX, dY and dZ are measured from the first "
                      "tracked point.")
        note.setWordWrap(True)
        box.addWidget(note)

        self.coordinates_table = QTableWidget()
        self.coordinates_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.coordinates_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.coordinates_table.setAlternatingRowColors(True)
        self.coordinates_table.verticalHeader().setVisible(False)
        self.coordinates_table.setFont(MONOSPACE)
        self.coordinates_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.coordinates_table.verticalHeader().setDefaultSectionSize(
            self.coordinates_table.fontMetrics().height() + 8)
        box.addWidget(self.coordinates_table)
        shortcut = QShortcut(QKeySequence.Copy, self.coordinates_table, self.copy_coordinates)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)

        bar = QHBoxLayout()
        self.coordinates_status = ElidedLabel()
        bar.addWidget(self.coordinates_status, 1)
        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy the selected rows, or the whole table, to paste into Excel.")
        copy_button.clicked.connect(self.copy_coordinates)
        bar.addWidget(copy_button)
        box.addLayout(bar)
        return tab

    # ----------------------------------------------------------------- browse

    def browse_top_down(self):
        """Ask the user for the top-down video."""
        self._browse_into(self.top_down_path, "Choose the top-down video", VIDEO_FILTER)

    def browse_side_on(self):
        """Ask the user for the side-on video."""
        self._browse_into(self.side_on_path, "Choose the side-on video", VIDEO_FILTER)

    def browse_calibration(self):
        """Ask the user for a calibration file."""
        self._browse_into(self.calibration_path, "Choose a calibration file",
                          CALIBRATION_FILTER)

    def _browse_into(self, line_edit, title, file_filter):
        """Ask the user for a file and put its path in line_edit."""
        path, _ = QFileDialog.getOpenFileName(self, title, line_edit.text(), file_filter)
        if path:
            line_edit.setText(path)

    def browse_save_dir(self):
        """Ask the user for a save folder and show the results already in it."""
        path = QFileDialog.getExistingDirectory(self, "Choose an output folder",
                                                self.save_dir.text())
        if path:
            self.save_dir.setText(path)
            self.load_results()

    # ------------------------------------------------------------------ steps

    def start_calibration(self):
        """Pick a video and a frame, then hand over to camera_calibration.main."""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Choose a video to calibrate on", self.top_down_path.text(),
            VIDEO_FILTER)
        if not video_path:
            return

        frame, ok = QInputDialog.getInt(self, "Calibration frame",
                                        "Which frame do you want to calibrate on?",
                                        0, 0, 10_000_000)
        if not ok:
            return

        target = Path(self.calibration_path.text() or CALIBRATION_PATH)
        directory = str(target.parent)
        name = target.stem
        written = target.parent / f"{name}.npz"
        stamp = written.stat().st_mtime if written.exists() else -1.0

        self.router.clear_script()
        self.router.script("what frame do you want", str(frame))
        self.router.script("what folder do you want", directory)
        self.router.script("what do you want to call", name)

        self.append_log(f"\nCalibrating on frame {frame} of {video_path}")
        self.append_log("Click the 16 ring corners in the OpenCV window. "
                        "z zooms, u undoes, s saves, q quits.")
        self._run(lambda: run_calibration(video_path),
                  lambda _: self._calibration_finished(written, stamp),
                  "Calibrating")

    def _calibration_finished(self, written, previous_mtime):
        """Use the new calibration file if one was saved, and say whether it was."""
        if written.exists() and written.stat().st_mtime > previous_mtime:
            self.calibration_path.setText(str(written))
            self.append_log(f"Calibration saved to {written}")
            self.statusBar().showMessage(f"Calibration saved to {written}")
        else:
            self.append_log("No calibration was saved.")
            self.statusBar().showMessage("No calibration was saved.")

    def _seam_detector(self) -> SeamAngleDetector:
        """The seam detector to run after top-down tracking, as chosen by the checkbox."""
        if self.automatic_seam.isChecked():
            return AutomaticSeamAngleDetector()
        return ManualSeamAngleDetector()

    def start_top_down(self):
        """Run the top-down stage and the seam angle step on a worker thread."""
        if not self._require(self.top_down_path, "Choose a top-down video first."):
            return
        display = ScriptedDisplay(main_path=self.top_down_path.text(),
                                  main_start=self.top_down_start.value())
        save_dir = self.save_dir.text()
        seam_detector = self._seam_detector()
        self._run(lambda: track_top_down(display, save_dir, TOP_DOWN_VALUE_FILE,
                                         seam_detector),
                  self._top_down_finished, "Tracking top-down")

    def _top_down_finished(self, data):
        """Show the top-down results and any seam warning; set the Delivery speed."""
        self.velocity.setValue(data["velocity"])
        self.readouts["Velocity"].setText(f"{data['velocity']:.2f} km/h")
        angle = data["seam_angle"]
        self.readouts["Seam angle"].setText(
            "-" if angle is None else f"{angle:+.2f} deg ({data['seam_source']})")
        self.readouts["Tracked points"].setText(str(data["point_count"]))
        self.statusBar().showMessage(f"Top-down done, {data['velocity']:.2f} km/h.")
        self.load_results(Path(data["path"]).parent)
        if data["seam_warning"]:
            QMessageBox.warning(self, "Cricket ball tracker", data["seam_warning"])

    def start_side_on(self):
        """Run the side-on stage on a worker thread, using the Delivery speed."""
        if not self._require(self.side_on_path, "Choose a side-on video first."):
            return
        if not self._require(self.calibration_path, "Choose a calibration file first."):
            return
        if self.velocity.value() <= 0.0:
            QMessageBox.information(self, "Cricket ball tracker",
                                    "Set a delivery speed, or run the top-down stage "
                                    "to measure one.")
            return

        display = ScriptedDisplay(side_path=self.side_on_path.text(),
                                  side_start=self.side_on_start.value())
        speed = self.velocity.value()
        calibration = self.calibration_path.text()
        save_dir = self.save_dir.text()
        self._run(lambda: track_side_on(display, speed, calibration, save_dir),
                  self._side_on_finished, "Tracking side-on")

    def _side_on_finished(self, data):
        """Show the side-on readouts and plots, and reload the saved results."""
        self.side_on_data = data
        self.engine, self.result = data["engine"], data["result"]
        self._show_results(self.result)
        self._draw_plots()
        self.load_results(Path(data["written"]["analysis_yaml"]).parent)
        self.statusBar().showMessage("Side-on done.")

    def start_both(self):
        """Top-down, a chance to change the speed, then side-on, in one run."""
        if not self._require(self.top_down_path, "Choose a top-down video first."):
            return
        if not self._require(self.side_on_path, "Choose a side-on video first."):
            return
        if not self._require(self.calibration_path, "Choose a calibration file first."):
            return

        display = ScriptedDisplay(self.top_down_path.text(), self.top_down_start.value(),
                                  self.side_on_path.text(), self.side_on_start.value())
        save_dir = self.save_dir.text()
        calibration = self.calibration_path.text()
        router = self.router
        seam_detector = self._seam_detector()

        def both():
            """Track top-down, ask which speed to use, then track side-on."""
            top_down = track_top_down(display, save_dir, TOP_DOWN_VALUE_FILE, seam_detector)
            speed = float(router(f"Top down velocity was calculated to be "
                                 f"{top_down['velocity']:.2f} km/h. "
                                 f"Speed to reconstruct with, in km/h:"))
            side_on = track_side_on(display, speed, calibration, save_dir)
            return {"top_down": top_down, "side_on": side_on, "velocity": speed}

        # The prompt above carries the measured speed as its default.
        router.clear_script()
        self._run(both, self._both_finished, "Running both stages")

    def _both_finished(self, data):
        """Show both stages' results, with the Delivery speed set to the speed used."""
        self._top_down_finished(data["top_down"])
        self.velocity.setValue(data["velocity"])
        self._side_on_finished(data["side_on"])
        self.statusBar().showMessage("Both stages done.")

    # ---------------------------------------------------------------- results

    def _show_results(self, result):
        """Show the side-on point count, swing and fit residuals in the readouts."""
        self.readouts["Tracked points"].setText(str(len(result.trajectory.frames)))
        self.readouts["Swing at last point"].setText(
            f"{result.swing_at_last_point_cm:+.2f} cm "
            f"at {result.last_point_distance_m:.2f} m")
        self.readouts["Swing at"].setText(
            f"{result.swing_at_17m_cm:+.2f} cm at {result.target_distance_m:.0f} m")
        self.readouts["Baseline residual"].setText(
            f"{result.baseline_residual_cm:.2f} cm RMS")
        self.readouts["Projection residual"].setText(
            f"{result.projection_residual_cm:.2f} cm RMS")

    def load_results(self, folder=None):
        """Fill the Results and Coordinates tabs from the files in folder, or the save folder."""
        folder = Path(folder) if folder else Path(self.save_dir.text() or SAVE_DIR)
        self.results_folder.set_full_text(f"Saved in {folder}")

        self.results_tree.clear()
        for section in load_results(folder, f"{TOP_DOWN_VALUE_FILE}.yaml"):
            item = QTreeWidgetItem([section.title, section.path.name])
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setToolTip(1, str(section.path))
            self.results_tree.addTopLevelItem(item)
            for row in section.rows:
                self._add_result_row(item, row)
            item.setExpanded(True)
        self.results_tree.resizeColumnToContents(0)

        table = load_coordinates(folder)
        self.coordinates_table.clear()
        self.coordinates_table.setColumnCount(len(table.headers))
        self.coordinates_table.setHorizontalHeaderLabels(table.headers)
        self.coordinates_table.setRowCount(len(table.rows))
        for r, row in enumerate(table.rows):
            for c, text in enumerate(row):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.coordinates_table.setItem(r, c, cell)
        self.coordinates_status.set_full_text(table.message or f"{len(table.rows)} frames, from {table.path}")

    def _add_result_row(self, parent, row: ResultRow):
        """Add row and its children under parent; collapse any with over 12 children."""
        item = QTreeWidgetItem([row.name, row.value])
        item.setFont(1, MONOSPACE)
        parent.addChild(item)
        for child in row.children:
            self._add_result_row(item, child)
        item.setExpanded(len(row.children) <= 12)

    def copy_results(self):
        """Copy the selected rows of the Results tab, or all of them, as tab-separated text."""
        rows = []
        everything = not self.results_tree.selectedItems()
        iterator = QTreeWidgetItemIterator(self.results_tree)
        while iterator.value():
            item = iterator.value()
            if everything or item.isSelected():
                rows.append(f"{item.text(0)}\t{item.text(1)}")
            iterator += 1
        QApplication.clipboard().setText("\n".join(rows))
        self.statusBar().showMessage(f"Copied {len(rows)} rows.")

    def copy_coordinates(self):
        """Copy the selected rows of the Coordinates tab, or the whole table, with its headings."""
        table = self.coordinates_table
        rows = sorted({index.row() for index in table.selectedIndexes()}) or range(table.rowCount())
        columns = range(table.columnCount())
        lines = ["\t".join(table.horizontalHeaderItem(c).text() for c in columns)]
        lines += ["\t".join(table.item(r, c).text() for c in columns) for r in rows]
        QApplication.clipboard().setText("\n".join(lines))
        self.statusBar().showMessage(f"Copied {len(lines) - 1} rows.")

    def _draw_plots(self):
        """Runs on the GUI thread, so matplotlib is never touched from a worker."""
        directory = Path(self.save_dir.text())
        directory.mkdir(parents=True, exist_ok=True)

        swing = resolve_plot_path(
            self.engine.plot_swing(self.result, save_path=str(directory / SWING_PLOT_FILE)))
        print(f"wrote {swing}")
        self.swing_view.set_image(swing)

        self.engine.plot_trajectory_3d(
            self.result, show=False,
            save_path=str(directory / TRAJECTORY_PLOT_FILE))
        trajectory = resolve_plot_path(directory / TRAJECTORY_PLOT_FILE)
        print(f"wrote {trajectory}")
        self.trajectory_view.set_image(trajectory)

        self.open_3d_button.setEnabled(True)
        self.tabs.setCurrentIndex(1)

        if self.show_3d.isChecked():
            self.open_interactive_3d()

    def open_interactive_3d(self):
        """Drag to rotate. The figure has to be built on the GUI thread to do that."""
        if self.result is None:
            return
        self.engine.plot_trajectory_3d(self.result, show=True)

    # ------------------------------------------------------------- plumbing

    def _require(self, line_edit, message):
        """Return whether line_edit has text, showing message if it is empty."""
        if line_edit.text().strip():
            return True
        QMessageBox.information(self, "Cricket ball tracker", message)
        return False

    def _run(self, step, on_done, description):
        """Start one step on a worker thread and lock the buttons while it runs."""
        if self.task is not None and self.task.isRunning():
            QMessageBox.information(self, "Cricket ball tracker",
                                    "A step is already running.")
            return

        self._set_busy(True, description)
        self.task = TaskThread(step, self)
        self.task.done.connect(on_done)
        self.task.done.connect(lambda _: self._set_busy(False, "Ready."))
        self.task.failed.connect(self._on_failed)
        self.task.cancelled.connect(self._on_cancelled)
        self.task.start()

    def _on_failed(self, message):
        """Log the traceback, unlock the buttons and show the error's last line."""
        self.append_log(message)
        self._set_busy(False, "Failed. See the log.")
        QMessageBox.critical(self, "Cricket ball tracker",
                             message.strip().splitlines()[-1])

    def _on_cancelled(self, message):
        """Log the cancellation and unlock the buttons."""
        self.append_log(f"Cancelled: {message}")
        self._set_busy(False, "Cancelled.")

    def _set_busy(self, busy, message):
        """Lock or unlock the step buttons and show message in the status bar."""
        for button in self.step_buttons:
            button.setEnabled(not busy)
        self.statusBar().showMessage(message)

    def append_log(self, text):
        """Add text to the Log tab and scroll to the end."""
        self.log.appendPlainText(text)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Give input() and sys.stdout back before the window closes."""
        self.router.remove()
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


def main():
    """Open the tracker window and run it until it is closed."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()