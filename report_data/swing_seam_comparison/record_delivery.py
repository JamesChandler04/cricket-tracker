"""Records deliveries for the swing and seam comparison, one after another. For each one, the
seam is clicked on top-down ball images from the frames the ball finder found the ball in,
and then the side-on window opens to click the ball through, as in the GUI's side-on stage.
The seam angle and clicked points are saved without a speed, which calculate_swing sets
later.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import yaml

if __package__ in (None, ""):
    # Started as a plain script (python report_data/swing_seam_comparison/<name>.py, or
    # the Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.swing_seam_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library import display, paths
from library.automated import automations
from library.automated.automatic_seam_angle import (MAX_SPREAD_DEG, AutomaticSeamAngleDetector,
                                                    AveragedSeamMeasurement)
from library.detect_seam_angle_file import (BALL_IMAGE_FOLDER, BallImage, ManualSeamAngleDetector, SeamMeasurement,
                                            save_ball_images, save_seam_measurement)
from library.helpers import Coord, TopDownBallData, TopDownBallDataPoint, Video
from library.manual_tracker import SideOnTracker, TopDownTracker
from library.physics_engines.side_on_physics_engine import (ANALYSIS_YAML_NAME, BASELINE_SOURCE_POINT_COUNT,
                                                            EXTRAPOLATION_SOURCE_POINT_COUNT, POINTS_CSV_NAME,
                                                            TrackedPoint)
from report_data.ball_detection import detect_ball_frames
from report_data.ball_detection.detect_ball_frames import Detection
from report_data.speed_comparison.speed_data import RADAR_PATH, ball_and_delivery, load_radar_speeds
from report_data.swing_seam_comparison.calculate_swing import MIN_POINTS, SWING_PLOT_NAME, analyse_delivery
from report_data.swing_seam_comparison.swing_seam_data import (DELIVERY_FILE_NAME, RecordedDelivery, SeamAngle,
                                                               delivery_folder, save_delivery)

FOUND_FRAMES_FOLDER = detect_ball_frames.OUTPUT_FOLDER
"""Default folder of the ball detection test's found_frames.txt lists, one subfolder per video."""
FINDER_FILES = (paths.CONFIG_PATH, Path(automations.__file__))
"""Files with the ball finder's settings. A found_frames.txt older than either is out of
date, so the finder is run again instead.
"""
VIDEO_FOLDER = paths.ROOT / "vids"
"""Folder the file dialog for the first top-down video opens in."""
SIDE_ON_LEAD_FRAMES = 10
"""Frames before the ball's first top-down frame that the side-on video opens at, unless
another start frame is typed. The two videos' frame numbers nearly line up.
"""
SEAM_CHOICES = ("manual", "both", "automatic")
"""Values of --seam: which seam angles to measure, the first being the default."""


class ChosenVideos(display.Display):
    """Stand-in for display.Display that gives the trackers videos already chosen, at their
    start frames, instead of asking for them.
    """

    def __init__(self, top_down: Path | None = None, side_on: Path | None = None, top_down_start: int = 0,
                 side_on_start: int = 0) -> None:
        """Set up the videos and the frames they open at."""
        self.top_down = top_down
        self.side_on = side_on
        self.top_down_start = top_down_start
        self.side_on_start = side_on_start

    def load_main_video(self) -> Video:
        """Return the top-down video at its start frame."""
        return _open(self.top_down, self.top_down_start, "top-down")

    def load_side_video(self) -> Video:
        """Return the side-on video at its start frame."""
        return _open(self.side_on, self.side_on_start, "side-on")


def matching_side_on_video(top_down: Path) -> Path | None:
    """Return the side-on video with the same ball and delivery numbers as top_down, from
    the folders next to the top-down video's folder (vids/B1/Behind for vids/B1/Birdseye),
    or None unless there is exactly one.
    """
    numbers = ball_and_delivery(top_down)
    if numbers is None:
        return None
    own_folder = top_down.resolve().parent
    matches = [video for folder in sorted(own_folder.parent.iterdir()) if folder.is_dir() and folder != own_folder
               for video in detect_ball_frames.find_videos(folder) if ball_and_delivery(video) == numbers]
    return matches[0] if len(matches) == 1 else None


def next_top_down_video(current: Path) -> Path | None:
    """Return the first video after current in its folder, by name, that has ball and
    delivery numbers and has not been recorded yet, or None if there is none.
    """
    for video in detect_ball_frames.find_videos(current.parent):
        numbers = ball_and_delivery(video)
        if video.name > current.name and numbers is not None and not _recorded_file(*numbers).exists():
            return video
    return None


def read_found_frames(path: Path) -> list[Detection]:
    """Return the detections listed in a found_frames.txt written by detect_ball_frames,
    without those its track check removed, which are only comment lines.
    """
    detections = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            frame, centre_x, centre_y, radius = (int(value) for value in line.split(","))
            detections.append(Detection(frame, centre_x, centre_y, radius))
    return detections


def find_ball(video: Path, found_folder: Path) -> tuple[list[Detection], str]:
    """Return the frames the ball was found in and how they were found: from the ball
    detection test's found_frames.txt for the video if there is one newer than the ball
    finder's settings, or else by running the ball finder on the video now, which saves
    nothing.
    """
    listed = found_folder / video.stem / detect_ball_frames.LIST_NAME
    if not listed.exists():
        print(f"There is no {listed.name} for {video.name} in {found_folder}, so the ball finder is running "
              "on it now, which takes about a minute.")
    else:
        changed = [path.name for path in FINDER_FILES if path.stat().st_mtime > listed.stat().st_mtime]
        if not changed:
            return read_found_frames(listed), "ball detection test"
        print(f"{listed} is older than {' and '.join(changed)}, so the ball finder is running on {video.name} "
              "again now, which takes about a minute.")
    with tempfile.TemporaryDirectory() as scratch:
        result = detect_ball_frames.check_video(video, Path(scratch))
    return result.detections, "ball finder"


def ball_points(detections: list[Detection], video: Path) -> list[TopDownBallDataPoint]:
    """Return the detections as top-down ball data for save_ball_images, with the ball's
    diameter across its centre, leaving out any ball that runs off the edge of the frame.
    """
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    points = []
    for found in detections:
        x, y, radius = found.centre_x, found.centre_y, found.radius
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            print(f"Top-down frame {found.frame} left out: the ball runs off the edge of the frame.")
            continue
        points.append(TopDownBallDataPoint(found.frame, TopDownBallData(
            top_left=Coord(x - radius, y), bottom_right=Coord(x + radius, y), centre=Coord(x, y),
            seam_start=None, seam_end=None, seam_angle=None)))
    return points


def click_top_down(video: Path) -> tuple[list[TopDownBallDataPoint], int]:
    """Click the ball through the top-down video as in the GUI's top-down stage, and return
    the clicked frames and the video's rotation. Q or Esc in the window ends the program.
    """
    start = _ask_frame("Top-down start frame", 0)
    print("In the top-down window, press C and click across the ball in two frames, then SPACE and click "
          "the ball's centre in each frame, then S.")
    tracker = TopDownTracker()
    tracker.display = ChosenVideos(top_down=video, top_down_start=start)
    points = tracker.get_top_down_points()
    assert tracker.top_down_video is not None
    return points, tracker.top_down_video.rotation


def measure_seams(images: list[BallImage], method: str) -> dict[str, SeamAngle]:
    """Measure the seam angle on the ball images with the automatic detector, by clicking
    in the seam picker, or both, as method says, and save each seam next to the images.
    Clicking is offered when the automatic detector finds no seam.
    """
    seams: dict[str, SeamAngle] = {}
    if method in ("automatic", "both"):
        found = AutomaticSeamAngleDetector().detect(images)
        if found is not None and found.seam_angle_deg is not None:
            seams["automatic"] = _seam_angle(found)
            save_seam_measurement(found)
        elif method == "automatic" and _yes("No automatic seam angle. Click the seam by hand?", True):
            method = "manual"
    if method in ("manual", "both"):
        print("\nClick two points along the seam on the clearest ball image, then S. Q skips the clicking.")
        clicked = ManualSeamAngleDetector().detect(images)
        if clicked is not None and clicked.seam_angle_deg is not None:
            seams["manual"] = _seam_angle(clicked)
            save_seam_measurement(clicked)
    return seams


def click_side_on(video: Path, start_frame: int) -> tuple[list[TrackedPoint], float] | None:
    """Click the ball through the side-on video from start_frame, as in the GUI's side-on
    stage. Returns the clicked points and the video's frame rate, or None if the window was
    closed with Q or Esc.
    """
    tracker = SideOnTracker()
    tracker.display = ChosenVideos(side_on=video, side_on_start=start_frame)
    points = tracker.get_side_on_points()
    side_on = tracker.side_on_video
    assert side_on is not None
    closed = not side_on.cap.isOpened()  # Q and Esc release the video, S does not
    side_on.cap.release()
    cv2.destroyAllWindows()
    return None if closed else (points, float(side_on.fps))


def record(top_down: Path, side_on: Path, seam_method: str, found_folder: Path, click_top: bool,
           side_start: int | None) -> RecordedDelivery | None:
    """Record one delivery: measure its seam angle on the top-down ball images, click the
    ball through its side-on video, and save both, with the ball images and seam files.
    Returns the recording, or None if nothing was saved.
    """
    numbers = ball_and_delivery(top_down)
    assert numbers is not None
    ball, delivery = numbers
    print(f"\n=== Ball {ball}, delivery {delivery} ===")
    print(f"Top-down video: {top_down}")
    print(f"Side-on video:  {side_on}")

    with tempfile.TemporaryDirectory() as scratch:
        detections: list[Detection] = []
        found_by = "clicks"
        if not click_top:
            detections, found_by = find_ball(top_down, found_folder)
        points = ball_points(detections, top_down)
        rotation = 0
        if not points:
            if not click_top and not _yes(f"No ball to use was found in {top_down.name}. "
                                          "Click it in the top-down video instead?", True):
                print("Nothing was saved for this delivery.")
                return None
            points, rotation = click_top_down(top_down)
            found_by = "clicks"
        frames = [point.frame_number for point in points if point.data.centre is not None]
        if not frames:
            print("No ball centres were clicked, so nothing was saved for this delivery.")
            return None
        print(f"\nSeam angle from top-down frames {detect_ball_frames.frame_ranges(frames)} ({found_by})")
        images = save_ball_images(str(top_down), points, scratch, rotation=rotation)
        seams = measure_seams(images, seam_method)
        if not seams:
            print("No seam angle was measured, so nothing was saved for this delivery.")
            return None
        for seam in seams.values():
            spread = "" if seam.spread_deg is None else f", spread {seam.spread_deg:.1f} deg"
            limit = "" if seam.warning is None else f", over the {MAX_SPREAD_DEG:g} deg limit"
            print(f"Seam angle ({seam.method}): {seam.angle_deg:+.2f} deg{spread}{limit}")

        start = (side_start if side_start is not None
                 else _ask_frame(f"\nSide-on start frame, with the ball first in top-down frame {frames[0]}",
                                 max(frames[0] - SIDE_ON_LEAD_FRAMES, 0)))
        clicked = click_side_on(side_on, start)
        if clicked is None:
            print("The side-on window was closed with Q or Esc, so nothing was saved for this delivery.")
            return None
        side_on_points, fps = clicked
        if len(side_on_points) < MIN_POINTS:
            print(f"Only {len(side_on_points)} side-on points were clicked, fewer than the {MIN_POINTS} needed, "
                  "so nothing was saved for this delivery.")
            return None
        if len(side_on_points) < BASELINE_SOURCE_POINT_COUNT + EXTRAPOLATION_SOURCE_POINT_COUNT:
            print(f"Note: with {len(side_on_points)} side-on points the baseline fit (first "
                  f"{BASELINE_SOURCE_POINT_COUNT}) and the projection fit (last {EXTRAPOLATION_SOURCE_POINT_COUNT}) "
                  "share points, so the swing will read small.")

        recorded = RecordedDelivery(
            ball=ball, delivery=delivery, recorded=datetime.now().strftime("%Y-%m-%d %H:%M"),
            top_down_video=str(top_down), ball_frames=frames, ball_found_by=found_by, seams=seams,
            side_on_video=str(side_on), side_on_fps=fps, side_on_points=side_on_points)
        path = save_delivery(recorded)
        images_folder = path.parent / BALL_IMAGE_FOLDER
        if images_folder.exists():
            shutil.rmtree(images_folder)
        shutil.copytree(Path(scratch) / BALL_IMAGE_FOLDER, images_folder)
        for name in (POINTS_CSV_NAME, ANALYSIS_YAML_NAME, SWING_PLOT_NAME):  # calculate_swing's, from earlier clicks
            (path.parent / name).unlink(missing_ok=True)
    print(f"\nSaved ball {ball} delivery {delivery}: {len(side_on_points)} side-on points, frames "
          f"{side_on_points[0].frame}-{side_on_points[-1].frame}, to {path}")
    return recorded


def preview_swing(recorded: RecordedDelivery) -> None:
    """Print the delivery's swing at its radar speed, as a first check of the clicks."""
    try:
        radar = load_radar_speeds(RADAR_PATH).get((recorded.ball, recorded.delivery))
    except (OSError, ValueError, yaml.YAMLError):
        radar = None
    if radar is None:
        print("No radar speed for this delivery, so no swing preview.")
        return
    try:
        _, result = analyse_delivery(recorded, radar, paths.CALIBRATION_PATH)
    except (OSError, ValueError) as error:
        print(f"No swing preview: {error}")
        return
    print(f"Swing at the radar speed, {radar:g} km/h: {result.swing_at_last_point_cm:+.1f} cm at the last tracked "
          f"point, {result.last_point_distance_m:.1f} m on. Fit residuals {result.baseline_residual_cm:.1f} cm "
          f"(baseline) and {result.projection_residual_cm:.1f} cm (projection).")


def ask_for_video(title: str, folder: Path) -> Path | None:
    """Ask for a video in a file dialog that opens in folder. Returns None if none is chosen."""
    # Imported here so that the program can be given its videos without tkinter.
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askopenfilename(title=title, initialdir=str(folder), filetypes=[
        ("Video files", " ".join(f"*{suffix}" for suffix in detect_ball_frames.VIDEO_TYPES)), ("All files", "*.*")])
    root.destroy()
    return Path(chosen) if chosen else None


def ask_next(current: Path) -> Path | None:
    """Offer the next unrecorded video after current in its folder, and return the one to
    record next, or None to stop.
    """
    following = next_top_down_video(current)
    while following is not None:
        answer = input(f"\nNext: {following.name}. Enter to record it, S to skip it, Q to stop: ").strip().lower()
        if answer.startswith("q"):
            return None
        if not answer.startswith("s"):
            return following
        following = next_top_down_video(following)
    print("No more unrecorded videos in that folder.")
    return None


def _open(path: Path | None, start_frame: int, label: str) -> Video:
    """Return the video at path, at start_frame; raise ValueError if there is no path."""
    if path is None:
        raise ValueError(f"No {label} video was chosen.")
    print(f"Loading {label} video: {path}")
    video = Video(str(path))
    video.current_frame = min(max(start_frame, 0), max(video.total_frames - 1, 0))
    return video


def _recorded_file(ball: int, delivery: int) -> Path:
    """Return the delivery.yaml a delivery is recorded in, whether or not it exists yet."""
    return delivery_folder(ball, delivery) / DELIVERY_FILE_NAME


def _seam_angle(seam: SeamMeasurement) -> SeamAngle:
    """Return a seam found by a seam detector as a SeamAngle to save."""
    angle = seam.seam_angle_deg
    assert angle is not None
    if isinstance(seam, AveragedSeamMeasurement):
        return SeamAngle(seam.method, angle, seam.frame_number, dict(seam.frame_angles_deg), seam.spread_deg,
                         seam.warning)
    return SeamAngle(seam.method, angle, seam.frame_number)


def _ask_frame(prompt: str, default: int) -> int:
    """Ask for a frame number in the console; Enter gives default."""
    while True:
        answer = input(f"{prompt} (Enter for {default}): ").strip()
        if not answer:
            return default
        try:
            return max(int(answer), 0)
        except ValueError:
            print("Type a whole frame number.")


def _yes(question: str, default: bool) -> bool:
    """Ask a yes or no question in the console; Enter gives default."""
    answer = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not answer else answer.startswith("y")


def main(argv: list[str] | None = None) -> None:
    """Record deliveries one after another, starting from the top-down video given or
    picked in a dialog, and offering the next unrecorded video in its folder after each.
    """
    parser = argparse.ArgumentParser(description="Record deliveries for the swing and seam comparison: the seam "
                                                 "angle from the top-down video, then the ball clicked through the "
                                                 "side-on video.")
    parser.add_argument("top_down", nargs="?", type=Path,
                        help="top-down video of the first delivery to record (default: pick one in a dialog)")
    parser.add_argument("--side-on", type=Path,
                        help="side-on video of the first delivery (default: the video with the same ball and "
                             "delivery numbers in a folder next to the top-down video's)")
    parser.add_argument("--side-start", type=int,
                        help=f"frame the first delivery's side-on video opens at (default: asked, suggesting "
                             f"{SIDE_ON_LEAD_FRAMES} frames before the ball's first top-down frame)")
    parser.add_argument("--seam", choices=SEAM_CHOICES, default=SEAM_CHOICES[0],
                        help="measure the seam angle by clicking it (the default), with the automatic detector "
                             "as well, or with the automatic detector only")
    parser.add_argument("--found-frames", type=Path, default=FOUND_FRAMES_FOLDER,
                        help=f"folder of the ball detection test's found frames (default {FOUND_FRAMES_FOLDER})")
    parser.add_argument("--click-top-down", action="store_true",
                        help="click the ball through the top-down video instead of using the ball finder's frames")
    parser.add_argument("--one", action="store_true", help="record one delivery, without offering the next")
    args = parser.parse_args(argv)

    top_down = args.top_down or ask_for_video("Select the top-down video of the first delivery to record",
                                              VIDEO_FOLDER if VIDEO_FOLDER.is_dir() else paths.ROOT)
    side_on_choice: Path | None = args.side_on
    side_start: int | None = args.side_start
    try:
        while top_down is not None:
            numbers = ball_and_delivery(top_down)
            if numbers is None:
                print(f"{top_down.name} has no ball and delivery number in its name, like "
                      "Jordan_B1_Birdseye_003.mp4, so it cannot be recorded.")
                raise SystemExit(1)
            if _recorded_file(*numbers).exists() and not _yes(
                    f"Ball {numbers[0]} delivery {numbers[1]} is already recorded. Record it again?", False):
                print("Kept the earlier recording.")
            else:
                side_on = (side_on_choice or matching_side_on_video(top_down)
                           or ask_for_video(f"Select the side-on video for {top_down.name}",
                                            top_down.parent.parent))
                if side_on is None:
                    print("No side-on video was chosen, so nothing was saved for this delivery.")
                while side_on is not None:
                    recorded = record(top_down, side_on, args.seam, args.found_frames, args.click_top_down,
                                      side_start)
                    if recorded is not None:
                        preview_swing(recorded)
                        break
                    if not _yes(f"Try ball {numbers[0]} delivery {numbers[1]} again?", False):
                        break
            side_on_choice, side_start = None, None  # these options are for the first delivery only
            top_down = None if args.one else ask_next(top_down)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
