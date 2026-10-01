"""
Wraps the pipeline in new_cricket_ball_tracker.py without modifying any module
it drives. manual_tracker, camera_calibration, top_down_physics_engine and
side_on_physics_engine are called exactly as the script called them, apart
from the seam angle.

The seam angle is its own step here, not part of top-down tracking. Once the
top-down video has been clicked through, detect_seam_angle_file crops the ball
out of every tracked frame using the clicked centres and diameter, saves the
crops to a ball_images folder, and hands them to a seam detector. The
"Automatic seam angle detection" checkbox picks which one (see
MainWindow._seam_detector). Unticked, ManualSeamAngleDetector opens a window to
click the seam, another blocking OpenCV loop that runs on the worker thread
like the trackers. Ticked, AutomaticSeamAngleDetector from automatic_seam_angle
finds it without a window.

Four things in the pipeline do not sit naturally inside a GUI. Each is handled
by a shim installed here rather than by editing the module that causes it.

  Blocking OpenCV loops
      Both trackers and the calibration clicker own a cv2 window and spin on
      cv2.waitKey until the user presses s, q or Esc. They run on a QThread so
      the window stays responsive. sys.exit inside them arrives as SystemExit
      and is reported as a cancellation instead of killing the app.

  input() prompts
      camera_calibration asks for the frame and the save path, and Video asks
      for the frame rate when the metadata is missing. PromptRouter replaces
      builtins.input for the whole process. Prompts the GUI already knows the
      answer to are answered from the form; anything else opens a modal dialog
      on the GUI thread while the worker blocks.

  tkinter file dialogs
      display.Display opens tkinter to pick a video, which is not safe off the
      main thread. Videos are chosen here up front and injected through
      ScriptedDisplay, which duck-types Display and never touches tkinter.

  matplotlib
      Figures are only ever created on the GUI thread. The worker returns the
      SwingResult and this window does the plotting, so pyplot is never touched
      from a worker and the interactive 3D view still rotates.
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
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):
    # Started as a plain script (python main/<name>.py, or the Run button in
    # VS Code), so put the project folder on the path for the library imports.
    # python -m main.<name> from the project folder does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import camera_calibration, paths
from library.automated.automatic_seam_angle import AutomaticSeamAngleDetector
from library.detect_seam_angle_file import (ManualSeamAngleDetector, SeamAngleDetector,
                                            save_ball_images, save_seam_measurement)
from library.helpers import Video
from library.log_bridge import bridge
from library.manual_tracker import SideOnTracker, TopDownTracker
from library.physics_engines.side_on_physics_engine import SideOnPhysicsEngine
from library.physics_engines.top_down_physics_engine import SelectionType, TopDownPhysicsEngine


SAVE_DIR = str(paths.DELIVERY_DIR)
TOP_DOWN_VALUE_FILE = "top_down_analysis"
SWING_PLOT_FILE = "swing"
TRAJECTORY_PLOT_FILE = "trajectory_3d"
CALIBRATION_PATH = str(paths.CALIBRATION_PATH)

VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
CALIBRATION_FILTER = "Calibration files (*.npz);;All files (*.*)"

MONOSPACE = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 10)


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
        text = str(prompt)
        answer = self._take_scripted(text)
        if answer is None:
            answer = self._ask(text)
        else:
            print(f"{text.strip()} -> {answer!r}")
        self.history.append((text, answer))
        return answer

    def _take_scripted(self, prompt):
        lowered = prompt.lower()
        for index, (needle, answer) in enumerate(self._scripted):
            if needle in lowered:
                self._scripted.pop(index)
                return answer
        return None

    def _ask(self, prompt):
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
        self.main_path = main_path
        self.main_start = int(main_start)
        self.side_path = side_path
        self.side_start = int(side_start)

    def load_main_video(self) -> Video:
        return self._open(self.main_path, self.main_start, "top-down")

    def load_side_video(self) -> Video:
        return self._open(self.side_path, self.side_start, "side-on")

    @staticmethod
    def _open(path, start_frame, label) -> Video:
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
        super().__init__(parent)
        self._step = step

    def run(self):
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
    if seam is not None:
        print(f"Seam saved to {save_seam_measurement(seam)}")
    print(f"Calculated seam angle ({seam_detector.method}): "
          + ("not set" if seam_angle is None else f"{seam_angle:+.2f} deg"))

    path = engine.save_top_down_analysis(save_dir, value_file, velocity, seam_angle,
                                         fps, len(points))
    print(f"Top down values saved to {path}")

    return {"velocity": float(velocity), "seam_angle": seam_angle,
            "seam_frame": None if seam is None else seam.frame_number,
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
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setStyleSheet("color: #888780;")
        self._source = None

    def set_image(self, path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._source = None
            self.setText(f"Could not load {path}")
            return
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._source is not None:
            self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))


class MainWindow(QMainWindow):
    """Controls on the left, log and plots on the right."""

    def __init__(self):
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

        self.statusBar().showMessage("Ready.")
        self.append_log("Pick both videos and a calibration file, then track.")

    # ------------------------------------------------------------------ build

    def _build_controls(self):
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

        return self.tabs

    # ----------------------------------------------------------------- browse

    def browse_top_down(self):
        self._browse_into(self.top_down_path, "Choose the top-down video", VIDEO_FILTER)

    def browse_side_on(self):
        self._browse_into(self.side_on_path, "Choose the side-on video", VIDEO_FILTER)

    def browse_calibration(self):
        self._browse_into(self.calibration_path, "Choose a calibration file",
                          CALIBRATION_FILTER)

    def _browse_into(self, line_edit, title, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, title, line_edit.text(), file_filter)
        if path:
            line_edit.setText(path)

    def browse_save_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose an output folder",
                                                self.save_dir.text())
        if path:
            self.save_dir.setText(path)

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
        self.velocity.setValue(data["velocity"])
        self.readouts["Velocity"].setText(f"{data['velocity']:.2f} km/h")
        angle = data["seam_angle"]
        self.readouts["Seam angle"].setText(
            "-" if angle is None else f"{angle:+.2f} deg (frame {data['seam_frame']})")
        self.readouts["Tracked points"].setText(str(data["point_count"]))
        self.statusBar().showMessage(f"Top-down done, {data['velocity']:.2f} km/h.")

    def start_side_on(self):
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
        self.side_on_data = data
        self.engine, self.result = data["engine"], data["result"]
        self._show_results(self.result)
        self._draw_plots()
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
        self._top_down_finished(data["top_down"])
        self.velocity.setValue(data["velocity"])
        self._side_on_finished(data["side_on"])
        self.statusBar().showMessage("Both stages done.")

    # ---------------------------------------------------------------- results

    def _show_results(self, result):
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
        self.append_log(message)
        self._set_busy(False, "Failed. See the log.")
        QMessageBox.critical(self, "Cricket ball tracker",
                             message.strip().splitlines()[-1])

    def _on_cancelled(self, message):
        self.append_log(f"Cancelled: {message}")
        self._set_busy(False, "Cancelled.")

    def _set_busy(self, busy, message):
        for button in self.step_buttons:
            button.setEnabled(not busy)
        self.statusBar().showMessage(message)

    def append_log(self, text):
        self.log.appendPlainText(text)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        self.router.remove()
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()