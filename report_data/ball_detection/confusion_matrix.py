"""Counts the frames of the ball detection test in a confusion matrix: frames with and
without the ball against frames the automatic ball finder did and did not find a ball in.
It reads the found_frames.txt that detect_ball_frames.py wrote for each video, prints the
matrix with its recall, precision and specificity, and draws it as a chart.

No frame is labelled by hand. Instead, the ball is taken to be in every frame from the first
to the last frame it was found in and in no other frame checked, so every frame it was
missed in lies between two found frames, and FALSE_POSITIVES of the found frames are taken
to be something other than the ball. Only the frames the finder checked are counted.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

if __package__ in (None, ""):
    # Started as a plain script (python report_data/ball_detection/<name>.py, or the
    # Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.ball_detection.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_seam_detector import INK, INK_2, MUTED, SURFACE
from report_data.ball_detection.detect_ball_frames import FOLDER, LIST_NAME, OUTPUT_FOLDER, frame_ranges

CHART_PATH = FOLDER / "confusion_matrix.png"
"""PNG file of the confusion matrix chart."""
FALSE_POSITIVES = 0
"""Default number of found frames in which the finder found something other than the ball."""

SHADES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
          "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
"""Light to dark blue steps that shade each cell of the chart by its share of its row."""
WHITE = "#ffffff"
"""Colour of the text on the darker cells of the chart."""

CHECKED_PATTERN = re.compile(r"frames 1-(\d+) checked")
"""Header text of found_frames.txt that gives the last frame checked."""


@dataclass
class VideoFrames:
    """The frames the finder checked in one video and found the ball in, read back from
    the video's found_frames.txt.
    """

    name: str
    frames_checked: int  # frames 1 to frames_checked were checked
    found: list[int]     # frames the ball was found in, in order

    @property
    def missed(self) -> list[int]:
        """Return the frames between the first and last found frame that the ball was not
        found in.
        """
        found = set(self.found)
        return [] if not self.found else [frame for frame in range(self.found[0], self.found[-1] + 1)
                                          if frame not in found]

    @property
    def ball_frames(self) -> int:
        """Return how many of the frames checked are taken to have the ball in them: those
        from the first to the last found frame.
        """
        return 0 if not self.found else self.found[-1] - self.found[0] + 1


@dataclass
class ConfusionMatrix:
    """Frames counted by whether the ball is in them and whether the finder found a ball."""

    true_positives: int   # ball in the frame and found
    false_negatives: int  # ball in the frame but not found
    false_positives: int  # no ball in the frame, but something found
    true_negatives: int   # no ball in the frame and nothing found

    @property
    def with_ball(self) -> int:
        """Return how many frames have the ball in them."""
        return self.true_positives + self.false_negatives

    @property
    def without_ball(self) -> int:
        """Return how many frames have no ball in them."""
        return self.false_positives + self.true_negatives

    @property
    def found(self) -> int:
        """Return how many frames the finder found a ball in."""
        return self.true_positives + self.false_positives

    @property
    def total(self) -> int:
        """Return how many frames were counted."""
        return self.with_ball + self.without_ball

    @property
    def recall(self) -> float | None:
        """Return the share of the frames with the ball that it was found in, or None if
        there are none.
        """
        return _share(self.true_positives, self.with_ball)

    @property
    def precision(self) -> float | None:
        """Return the share of the finds that were the ball, or None if there are none."""
        return _share(self.true_positives, self.found)

    @property
    def specificity(self) -> float | None:
        """Return the share of the frames without the ball that nothing was found in, or
        None if there are none.
        """
        return _share(self.true_negatives, self.without_ball)

    @property
    def accuracy(self) -> float | None:
        """Return the share of all frames counted that the finder got right, or None if
        there are none.
        """
        return _share(self.true_positives + self.true_negatives, self.total)

    @property
    def f1(self) -> float | None:
        """Return the F1 score, the harmonic mean of recall and precision, or None if there
        are no finds and no frames with the ball.
        """
        return _share(2 * self.true_positives, 2 * self.true_positives + self.false_positives
                      + self.false_negatives)


def read_frame_list(path: Path) -> VideoFrames:
    """Read the frames checked and the frames the ball was found in from a found_frames.txt
    written by detect_ball_frames.py. The video is named after the file's folder.
    """
    frames_checked: int | None = None
    found: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            match = CHECKED_PATTERN.search(line)
            if match:
                frames_checked = int(match.group(1))
            elif "no frames checked" in line:
                frames_checked = 0
        elif line:
            try:
                found.append(int(line.split(",")[0]))
            except ValueError:
                raise ValueError(f"{path} has a line that does not start with a frame number: {line}") from None
    if frames_checked is None:
        raise ValueError(f"{path} does not say which frames were checked")
    found.sort()
    if found and (found[0] < 1 or found[-1] > frames_checked):
        raise ValueError(f"{path} lists a found frame outside the frames checked, 1-{frames_checked}")
    return VideoFrames(path.parent.name, frames_checked, found)


def load_results(folder: Path) -> tuple[list[VideoFrames], list[str]]:
    """Read every video's found_frames.txt in folder, one subfolder per video. Returns the
    videos, sorted by name, and the names of the subfolders without a found_frames.txt.
    """
    videos, unfinished = [], []
    for subfolder in sorted(path for path in folder.iterdir() if path.is_dir()):
        path = subfolder / LIST_NAME
        if path.exists():
            videos.append(read_frame_list(path))
        else:  # still being checked, or the run was stopped
            unfinished.append(subfolder.name)
    return videos, unfinished


def count(videos: list[VideoFrames], false_positives: int) -> ConfusionMatrix:
    """Count the frames checked in the videos into a confusion matrix: the frames from the
    first to the last found frame of each video have the ball in them, the rest do not,
    and false_positives of the found frames were not the ball.
    """
    found = sum(len(video.found) for video in videos)
    if false_positives > found:
        raise ValueError(f"{false_positives} false positives is more than the {found} frames found")
    missed = sum(len(video.missed) for video in videos)
    checked = sum(video.frames_checked for video in videos)
    true_positives = found - false_positives
    return ConfusionMatrix(true_positives, missed, false_positives,
                           checked - true_positives - missed - false_positives)


def summarise(videos: list[VideoFrames], matrix: ConfusionMatrix, false_positives: int) -> list[str]:
    """Return printable lines: the assumptions, the matrix, its measures, and the frames
    checked, found and missed in each video.
    """
    width = 3 + max(len("Ball found"), *(len(f"{value:,}") for value in (
        matrix.true_positives, matrix.false_negatives, matrix.false_positives, matrix.true_negatives)))
    lines = [
        f"Assumed: {_plural(false_positives, 'false positive')}, and the ball in every "
        "frame from the first to the last frame it was found in",
        "(so every miss lies between two found frames) and in no other frame checked.",
        "",
        f"{'':16}{'Ball found':>{width}}{'Not found':>{width}}",
        f"{'Ball in frame':16}{matrix.true_positives:>{width},}{matrix.false_negatives:>{width},}",
        f"{'Ball not in frame':16}{matrix.false_positives:>{width},}{matrix.true_negatives:>{width},}",
        "",
        f"Recall       {_percent(matrix.recall):>7}  ({matrix.true_positives:,} of the {matrix.with_ball:,} "
        "frames with the ball found)",
        f"Precision    {_percent(matrix.precision):>7}  ({matrix.true_positives:,} of the {matrix.found:,} "
        "finds were the ball)",
        f"Specificity  {_percent(matrix.specificity):>7}  ({matrix.true_negatives:,} of the "
        f"{matrix.without_ball:,} frames without the ball left alone)",
        f"Accuracy     {_percent(matrix.accuracy):>7}  ({matrix.true_positives + matrix.true_negatives:,} "
        f"of {matrix.total:,} frames)",
        f"F1 score     {'-' if matrix.f1 is None else f'{matrix.f1:.3f}':>7}",
        "",
        "Each video: frames checked, frames the ball was found in, and frames missed between them",
    ]
    name_width = max((len(video.name) for video in videos), default=0)
    for video in videos:
        missed = f": {frame_ranges(video.missed)}" if video.missed else ""
        lines.append(f"  {video.name:<{name_width}}  {video.frames_checked:>5,} checked  {len(video.found):>3} found  "
                     f"{len(video.missed):>3} missed{missed}")
    empty = [video for video in videos if not video.found]
    if empty:
        names = ", ".join(video.name for video in empty)
        lines += ["", f"No ball found in {names}, so {'its' if len(empty) == 1 else 'their'} "
                      f"{sum(video.frames_checked for video in empty):,} frames checked all count as frames "
                      "without the ball."]
    return lines


def plot_matrix(matrix: ConfusionMatrix, video_count: int, false_positives: int, path: Path) -> Figure:
    """Draw and save the chart: the four counts in a two-by-two grid, each cell shaded by its
    share of its row, with recall, precision and specificity under it.
    """
    figure = plt.figure(figsize=(7.2, 6.0), facecolor=SURFACE)
    axes = figure.add_axes((0.25, 0.2, 0.7, 0.62))
    axes.set_xlim(0, 2)
    axes.set_ylim(2, 0)  # first row at the top
    axes.set_aspect("equal")
    axes.axis("off")

    gap = 0.03  # surface gap between the cells, in cell widths
    cells = [(0, 0, matrix.true_positives, matrix.with_ball, "True positives"),
             (0, 1, matrix.false_negatives, matrix.with_ball, "False negatives"),
             (1, 0, matrix.false_positives, matrix.without_ball, "False positives"),
             (1, 1, matrix.true_negatives, matrix.without_ball, "True negatives")]
    shades = LinearSegmentedColormap.from_list("row shares", SHADES)
    for row, column, value, row_total, name in cells:
        share = _share(value, row_total)
        fill = to_hex(shades(0.0 if share is None else share))
        axes.add_patch(FancyBboxPatch((column + gap / 2, row + gap / 2), 1 - gap, 1 - gap,
                                      boxstyle="round,pad=0,rounding_size=0.025", facecolor=fill,
                                      edgecolor="none"))
        ink = _text_colour(fill)
        axes.text(column + 0.5, row + 0.40, f"{value:,}", color=ink, fontsize=22, fontweight="bold",
                  ha="center", va="center")
        axes.text(column + 0.5, row + 0.62, name, color=ink, fontsize=10.5, ha="center", va="center")
        axes.text(column + 0.5, row + 0.76, _percent(share), color=ink, fontsize=10, ha="center", va="center")

    for column, text in enumerate(("Ball found", "Ball not found")):
        axes.text(column + 0.5, -0.06, text, color=INK_2, fontsize=11, ha="center", va="bottom")
    for row, (text, row_total) in enumerate((("Ball in frame", matrix.with_ball),
                                             ("Ball not in frame", matrix.without_ball))):
        axes.text(-0.06, row + 0.47, text, color=INK_2, fontsize=11, ha="right", va="bottom")
        axes.text(-0.06, row + 0.53, f"{row_total:,} frames", color=MUTED, fontsize=9.5, ha="right", va="top")

    figure.text(0.04, 0.95, "Automatic ball detection", color=INK, fontsize=12, ha="left", va="top")
    figure.text(0.04, 0.91, f"{matrix.total:,} frames checked in {_plural(video_count, 'top-down video')}",
                color=INK_2, fontsize=10, ha="left", va="top")
    measures = (f"Recall {_percent(matrix.recall)}     Precision {_percent(matrix.precision)}     "
                f"Specificity {_percent(matrix.specificity)}")
    figure.text(0.6, 0.13, measures, color=INK_2, fontsize=10.5, ha="center", va="top")
    note = f"Note: Percentages and shading are the share of each row."
    figure.text(0.04, 0.03, note, color=MUTED, fontsize=8.5, ha="left", va="bottom")
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    return figure


def _plural(number: int, noun: str) -> str:
    """Return the number and the noun, with an s on the noun unless the number is 1."""
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _share(part: int, whole: int) -> float | None:
    """Return part as a fraction of whole, or None if whole is 0."""
    return None if whole == 0 else part / whole


def _percent(share: float | None) -> str:
    """Return a share as a percentage with just enough decimals that it only reads 0% or
    100% when it is, or "-" for None.
    """
    if share is None:
        return "-"
    value = 100 * share
    decimals = 1
    while decimals < 4 and 0 < value < 100 and round(value, decimals) in (0, 100):
        decimals += 1
    return f"{value:.{decimals}f}%"


def _text_colour(fill: str) -> str:
    """Return white or the ink colour, whichever stands out more against fill."""
    def luminance(colour: str) -> float:
        """Return a colour's relative luminance, as WCAG defines it."""
        channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour)]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    background = luminance(fill)
    on_white = 1.05 / (background + 0.05)
    on_ink = (background + 0.05) / (luminance(INK) + 0.05)
    return WHITE if on_white >= on_ink else INK


def main(argv: list[str] | None = None) -> None:
    """Count the frames the finder checked in every video's results into a confusion
    matrix, print it with its measures, and save and show the chart.
    """
    parser = argparse.ArgumentParser(description="Build the confusion matrix of the ball detection test from "
                                                 "the found_frames.txt of each video.")
    parser.add_argument("folder", nargs="?", type=Path, default=OUTPUT_FOLDER,
                        help=f"folder of detect_ball_frames.py results, one subfolder per video "
                             f"(default {OUTPUT_FOLDER})")
    parser.add_argument("--false-positives", type=int, default=FALSE_POSITIVES,
                        help=f"found frames that were not the ball (default {FALSE_POSITIVES})")
    parser.add_argument("--no-show", action="store_true", help="save the chart without opening it")
    args = parser.parse_args(argv)
    if args.false_positives < 0:
        parser.error("--false-positives cannot be negative")

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"There are no results in {folder}. Run the finder first: "
              "python -m report_data.ball_detection.detect_ball_frames")
        raise SystemExit(1)
    try:
        videos, unfinished = load_results(folder)
        if unfinished:
            print(f"Left out {', '.join(unfinished)}: no {LIST_NAME} yet (still being checked, or the run was "
                  "stopped).")
        if not videos:
            print(f"There is no {LIST_NAME} in any subfolder of {folder}.")
            raise SystemExit(1)
        matrix = count(videos, args.false_positives)
    except ValueError as error:
        print(error)
        raise SystemExit(1)

    print(f"Confusion matrix of the automatic ball finder, from {_plural(len(videos), 'video')} in {folder}")
    print("\n".join(summarise(videos, matrix, args.false_positives)))
    figure = plot_matrix(matrix, len(videos), args.false_positives, CHART_PATH)
    print(f"\nSaved {CHART_PATH.name} to {CHART_PATH.parent}")
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
