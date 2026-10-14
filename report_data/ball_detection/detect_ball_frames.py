"""Runs the automatic top-down ball finder on every video in a folder, for the ball
detection test. Each video is searched until the finder has lost the ball, and the frames
the ball was found in are saved, with the ball circled, along with a text file listing them.
Detections that do not fit the ball's straight, steady track (automations.check_track) are
removed, and their frames kept under another name for review.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if __package__ in (None, ""):
    # Started as a plain script (python report_data/ball_detection/<name>.py, or the
    # Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.ball_detection.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library import paths
from library.automated import automations
from library.helpers import TopDownBallData

if TYPE_CHECKING:
    from cv2.typing import MatLike

FOLDER = paths.REPORT_DATA_DIR / "ball_detection"
"""Folder of the ball detection test."""
OUTPUT_FOLDER = FOLDER / "found_frames"
"""Default folder for the results, with one subfolder per video, named after it."""
VIDEO_TYPES = (".mp4", ".avi", ".mov", ".mkv")
"""File extensions read as videos, in any letter case."""
FRAME_NAME = "frame_{frame:06d}.jpg"
"""File name of a saved frame the ball was found in."""
REMOVED_FRAME_NAME = "removed_frame_{frame:06d}.jpg"
"""File name a saved frame is given when the track check removes its detection, so that it
can still be reviewed.
"""
LIST_NAME = "found_frames.txt"
"""File name of the list of frames the ball was found in, in each video's subfolder."""
CIRCLE_COLOUR = (0, 255, 0)
"""BGR colour of the circle drawn round the found ball on the saved frames (green)."""
CENTRE_COLOUR = (0, 0, 255)
"""BGR colour of the dot drawn on the found ball's centre (red)."""


@dataclass
class Detection:
    """Where the finder put the ball in one frame, in px."""

    frame: int
    centre_x: int
    centre_y: int
    radius: int


@dataclass
class VideoResult:
    """What the finder found in one video."""

    video: Path
    frame_count: int  # frames in the video, frame 0 included
    fps: float
    detections: list[Detection]  # the detections kept after the track check
    last_checked: int  # number of the last frame checked, or 0 if none was
    stopped: bool  # whether the search stopped because the finder lost the ball
    seconds: float  # time taken to check the video
    removed: list[tuple[Detection, str]] = field(default_factory=list)  # removed by the track check, with why

    @property
    def frames_checked(self) -> int:
        """Return how many frames were checked: frames 1 to last_checked, as frame 0 has no
        frame before it to compare with.
        """
        return self.last_checked

    @property
    def found_frames(self) -> list[int]:
        """Return the numbers of the frames the ball was found in."""
        return [detection.frame for detection in self.detections]


def find_videos(folder: Path) -> list[Path]:
    """Return the videos in folder, not in its subfolders, sorted by name."""
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in VIDEO_TYPES)


def check_video(video: Path, output: Path) -> VideoResult:
    """Run the finder on each frame of the video against the frame before it, as
    get_ball_data does, until the finder has lost the ball (TopDownBallFinder.ball_lost)
    or the video ends, remove the detections that do not fit the ball's track
    (automations.check_track), and save the frames the ball was found in and their list
    to a subfolder of output named after the video.
    """
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Checking {video.name} ({frame_count} frames)...", flush=True)

    folder = output / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    _clear_earlier_run(folder)

    finder = automations.TopDownBallFinder()
    detections: list[Detection] = []
    frame_number = -1  # number of the frame read last
    previous: MatLike | None = None
    start = time.perf_counter()
    try:
        while not finder.ball_lost:
            read, frame = capture.read()
            if not read:
                break
            frame_number += 1
            if previous is not None:
                # find_ball draws on the frame it is given, so it gets a copy: this frame
                # is saved clean and is the background for the next frame.
                detection = _detection(frame_number, finder.track_ball(frame.copy(), previous))
                if detection is not None:
                    detections.append(detection)
                    _write_image(folder / FRAME_NAME.format(frame=detection.frame),
                                 _circled(frame, detection))
            previous = frame
    finally:
        capture.release()

    check = automations.check_track([(d.frame, d.centre_x, d.centre_y, d.radius) for d in detections])
    removed = [(detections[index], reason) for index, reason in sorted(check.removed.items())]
    for detection, _ in removed:
        (folder / FRAME_NAME.format(frame=detection.frame)).replace(
            folder / REMOVED_FRAME_NAME.format(frame=detection.frame))
    kept = [detections[index] for index in check.kept]

    result = VideoResult(video, max(frame_count, frame_number + 1), fps, kept, max(frame_number, 0),
                         finder.ball_lost, time.perf_counter() - start, removed)
    write_frame_list(folder / LIST_NAME, result, automations._load_bounding_box("top_down"))
    return result


def check_video_safely(video: Path, output: Path) -> VideoResult | str:
    """Run check_video, returning any error as text so that one bad video does not stop
    the run.
    """
    try:
        return check_video(video, output)
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def write_frame_list(path: Path, result: VideoResult, box: tuple[int, int, int, int] | None) -> None:
    """Write the frames the ball was found in to path, one per line with the ball's centre
    and radius, under a header giving the video and the finder's settings. Detections the
    track check removed follow as comment lines, with the reason for each.
    """
    stop = (f", until the ball had been missed in {automations.TOP_DOWN_MAX_MISSED_FRAMES} frames in a row"
            if result.stopped else "")
    checked = (f"frames 1-{result.last_checked} checked, each against the frame before it{stop}"
               if result.frames_checked else "no frames checked")
    lines = [
        f"# Video: {result.video}",
        f"# {result.frame_count} frames at {result.fps:.2f} fps; {checked}",
        f"# Ball found in {len(result.detections)} frames: {frame_ranges(result.found_frames)}",
    ]
    if result.removed:
        lines.append(f"# {_detections(len(result.removed))} more removed as off the ball's track: "
                     f"{frame_ranges([d.frame for d, _ in result.removed])} (listed at the end)")
    lines.append("# frame, centre_x_px, centre_y_px, radius_px")
    lines += [f"{d.frame}, {d.centre_x}, {d.centre_y}, {d.radius}" for d in result.detections]
    lines += [f"# removed {d.frame}, {d.centre_x}, {d.centre_y}, {d.radius}: {reason}" for d, reason in result.removed]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def search_area(box: tuple[int, int, int, int] | None) -> str:
    """Describe the search box from config.yml, or say that the whole frame is searched."""
    if box is None:
        return "the whole frame"
    x_start, y_start, x_end, y_end = box
    return f"x {x_start}-{x_end}, y {y_start}-{y_end} px"


def frame_ranges(frames: list[int]) -> str:
    """Return sorted frame numbers as runs of consecutive frames, like "585-589, 601", or
    "none".
    """
    if not frames:
        return "none"
    runs = [[frames[0], frames[0]]]
    for frame in frames[1:]:
        if frame == runs[-1][1] + 1:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in runs)


def run_all(videos: list[Path], output: Path, workers: int) -> list[VideoResult]:
    """Check every video, workers of them at a time, printing a line as each one finishes.
    Returns the results sorted by video name.
    """
    results: list[VideoResult] = []

    def finished(video: Path, outcome: VideoResult | str) -> None:
        """Print how a video went, and keep its result if it was checked."""
        if isinstance(outcome, str):
            print(f"{video.name}: not checked - {outcome}", flush=True)
            return
        results.append(outcome)
        stop = f", stopped at frame {outcome.last_checked}" if outcome.stopped else ""
        removed = (f", removed {frame_ranges([d.frame for d, _ in outcome.removed])} as off the ball's track"
                   if outcome.removed else "")
        print(f"{video.name}: ball in {len(outcome.detections)} of {outcome.frames_checked} frames checked "
              f"({frame_ranges(outcome.found_frames)}){removed}{stop}, {_duration(outcome.seconds)}", flush=True)

    if workers == 1:
        for video in videos:
            finished(video, check_video_safely(video, output))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_one_opencv_thread) as pool:
            futures = {pool.submit(check_video_safely, video, output): video for video in videos}
            for future in as_completed(futures):
                finished(futures[future], future.result())
    return sorted(results, key=lambda result: result.video.name)


def summarise(results: list[VideoResult]) -> list[str]:
    """Return printable lines on how many frames the ball was found in, overall and for
    each video.
    """
    checked = sum(result.frames_checked for result in results)
    found = sum(len(result.detections) for result in results)
    width = max((len(result.video.name) for result in results), default=0)
    removed = sum(len(result.removed) for result in results)
    lines = [f"Ball found in {found} of {checked} frames checked, in {len(results)} videos"
             + (f" ({_detections(removed)} more removed as off the ball's track)" if removed else "") + ":"]
    lines += [f"  {result.video.name:<{width}}  {len(result.detections):>3} found  "
              f"{frame_ranges(result.found_frames)}"
              + (f"  (removed {frame_ranges([d.frame for d, _ in result.removed])})" if result.removed else "")
              for result in results]
    return lines


def choose_folder() -> str:
    """Ask for the folder of videos with a folder dialog. Returns an empty string if none
    is chosen.
    """
    # Imported here so that the worker processes, which never show the dialog, skip it.
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select the folder of top-down videos")
    root.destroy()
    return folder


def _detection(frame: int, ball: TopDownBallData | None) -> Detection | None:
    """Return the ball find_ball returned for a frame as a Detection, or None if it found
    no ball.
    """
    if ball is None or ball.centre is None or ball.top_left is None or ball.bottom_right is None:
        return None
    # find_ball puts the box corners at the centre plus and minus the radius.
    return Detection(frame, ball.centre.x, ball.centre.y, (ball.bottom_right.x - ball.top_left.x) // 2)


def _circled(frame: MatLike, detection: Detection) -> MatLike:
    """Return a copy of the frame with the found ball circled and its centre marked, drawn
    the way find_ball draws them.
    """
    image = frame.copy()
    centre = (detection.centre_x, detection.centre_y)
    cv2.circle(image, centre, detection.radius, CIRCLE_COLOUR, automations.DETECTION_CIRCLE_THICKNESS)
    cv2.circle(image, centre, 2, CENTRE_COLOUR, 3)
    return image


def _write_image(path: Path, image: MatLike) -> None:
    """Save an image to path as a JPEG."""
    # imencode and write_bytes instead of cv2.imwrite, which fails on Windows when the
    # folder name has characters outside the system code page.
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    if not encoded_ok:
        raise ValueError(f"could not encode {path}")
    path.write_bytes(encoded.tobytes())


def _clear_earlier_run(folder: Path) -> None:
    """Delete the frames and frame list an earlier run left in folder, and nothing else."""
    for path in [*folder.glob("frame_*.jpg"), *folder.glob("removed_frame_*.jpg")]:
        path.unlink()
    (folder / LIST_NAME).unlink(missing_ok=True)


def _one_opencv_thread() -> None:
    """Keep OpenCV to one thread in a worker process, so that the workers share the
    processor evenly.
    """
    cv2.setNumThreads(1)


def _detections(count: int) -> str:
    """Return a count of detections as text, like "1 detection" or "3 detections"."""
    return f"{count} detection{'' if count == 1 else 's'}"


def _duration(seconds: float) -> str:
    """Return a time in seconds as text, like "4 min 12 s"."""
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes} min {remainder} s" if minutes else f"{remainder} s"


def main(argv: list[str] | None = None) -> None:
    """Run the finder on every video in the folder given, or in one picked in a folder
    dialog, and save what it found.
    """
    parser = argparse.ArgumentParser(description="Run the automatic top-down ball finder on every video in a folder.")
    parser.add_argument("folder", nargs="?", help="folder of top-down videos (default: pick one in a dialog)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FOLDER,
                        help=f"folder for the results, one subfolder per video (default {OUTPUT_FOLDER})")
    parser.add_argument("--workers", type=int, default=1,
                        help="how many videos to check at once, each in its own process (default 1)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be 1 or more")

    chosen = args.folder or choose_folder()
    if not chosen:
        print("No folder chosen.")
        return
    folder = Path(chosen)
    if not folder.is_dir():
        print(f"{folder} is not a folder.")
        raise SystemExit(1)
    videos = find_videos(folder)
    if not videos:
        print(f"There are no videos in {folder}.")
        raise SystemExit(1)

    output: Path = args.output
    print(f"{len(videos)} videos in {folder}")
    print(f"Search area (top_down box in config.yml): {search_area(automations._load_bounding_box('top_down'))}")
    print(f"Results go to {output}\n")
    start = time.perf_counter()
    results = run_all(videos, output, args.workers)
    print("\n" + "\n".join(summarise(results)))
    print(f"\nTook {_duration(time.perf_counter() - start)}. Frames and lists saved to {output}")


if __name__ == "__main__":
    main()
