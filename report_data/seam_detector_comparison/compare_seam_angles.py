"""Compares the seam angles clicked by hand on the sample ball images with those the automatic
seam angle detector finds, in a chart, tables and a printed summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

if __package__ in (None, ""):
    # Started as a plain script (python report_data/seam_detector_comparison/<name>.py,
    # or the Run button in VS Code), so put the project folder on the path for the
    # imports. python -m report_data.seam_detector_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_seam_detector import AXIS, BAND, GRID, INK, INK_2, MUTED, SURFACE
from generate_seam_training_data import write_image
from library.automated.automatic_seam_angle import AutomaticSeamAngleDetector, AutomaticSeamMeasurement
from library.detect_seam_angle_file import (INDEX_FILE_NAME, OUTLINE, SEAM_COLOUR, BallImage, fold_line_angle,
                                            load_ball_images)
from report_data.seam_detector_comparison.click_seams import ManualSeam, load_manual_seams
from report_data.seam_detector_comparison.make_seam_images import (DETAILS_PATH, FOLDER, IMAGE_FOLDER, MANUAL_PATH,
                                                                   load_image_details)

AUTOMATIC_PATH = FOLDER / "automatic_seam_angles.csv"
"""CSV of the seams the automatic detector found on the sample images."""
COMPARISON_PATH = FOLDER / "seam_angle_comparison.csv"
"""CSV of each image's seam angle clicked by hand, found automatically and made with."""
CHART_PATH = FOLDER / "seam_angle_comparison.png"
"""PNG file of the comparison chart."""
EXAMPLES_PATH = FOLDER / "largest_differences.png"
"""PNG file of the images whose two seam angles differ most, with both seams drawn on them."""

TOLERANCE_DEG = 2.5
"""Default largest difference counted as agreeing, in degrees."""
EXAMPLE_COUNT = 8
"""Number of images in largest_differences.png."""
EXAMPLE_SIZE = 280
"""Side of each image in largest_differences.png, in px."""
EXAMPLES_PER_ROW = 4
"""Images on each row of largest_differences.png."""
AUTOMATIC_COLOUR = (0, 230, 255)
"""Yellow for the automatic seam in largest_differences.png, as a BGR colour."""

KINDS = {"photo": ("Photographed balls", "#2a78d6"), "drawn": ("Drawn balls", "#eb6834")}
"""Chart name and colour of each kind of sample image."""
OTHER_KIND = ("Other images", MUTED)
"""Chart name and colour for images missing from image_details.csv."""

AUTOMATIC_COLUMNS = ["image", "seam_found", "start_x_px", "start_y_px", "end_x_px", "end_y_px",
                     "seam_line_deg", "travel_direction_deg", "seam_angle_deg", "contrast", "time_ms"]
"""Column headings of automatic_seam_angles.csv."""
COMPARISON_COLUMNS = ["image", "kind", "made_seam_angle_deg", "manual", "manual_seam_angle_deg", "automatic",
                      "automatic_seam_angle_deg", "automatic_minus_manual_deg", "manual_minus_made_deg",
                      "automatic_minus_made_deg", "contrast"]
"""Column headings of seam_angle_comparison.csv."""


@dataclass
class AutomaticSeam:
    """The seam the automatic detector found on one image, if it found one."""

    image: str
    start: tuple[float, float] | None   # ends of the seam line, in image px
    end: tuple[float, float] | None
    seam_line_deg: float | None         # angle of the seam line in the image
    seam_angle_deg: float | None        # angle to the direction of travel
    contrast: float
    time_ms: float


@dataclass
class Comparison:
    """One image's seam angle clicked by hand, found automatically and made with."""

    image: BallImage
    kind: str
    made_deg: float | None
    manual: ManualSeam | None
    automatic: AutomaticSeam

    @property
    def manual_deg(self) -> float | None:
        """Return the seam angle clicked by hand, or None if there is none."""
        return None if self.manual is None else self.manual.seam_angle_deg

    @property
    def automatic_deg(self) -> float | None:
        """Return the seam angle found automatically, or None if none was found."""
        return self.automatic.seam_angle_deg

    @property
    def difference(self) -> float | None:
        """Return the automatic minus the manual seam angle, folded into (-90, 90] deg, or None
        if either is missing.
        """
        if self.manual_deg is None or self.automatic_deg is None:
            return None
        return fold_line_angle(self.automatic_deg - self.manual_deg)

    @property
    def midpoint(self) -> float | None:
        """Return the angle halfway between the two seam angles, or None if either is missing."""
        difference = self.difference
        if difference is None or self.manual_deg is None:
            return None
        return fold_line_angle(self.manual_deg + difference / 2)


@dataclass
class Agreement:
    """How closely one set of seam angles agrees with another."""

    count: int
    mean: float      # mean difference
    sd: float        # standard deviation of the differences
    rms: float       # root mean square difference
    largest: float   # largest difference either way
    within: int      # how many differences are within the tolerance

    @property
    def limits(self) -> tuple[float, float]:
        """Return the 95% limits of agreement: the mean difference -/+ 1.96 standard deviations."""
        return self.mean - 1.96 * self.sd, self.mean + 1.96 * self.sd


def agreement(differences: list[float], tolerance: float) -> Agreement | None:
    """Return how closely two sets of angles agree from their differences, or None if there are none."""
    if not differences:
        return None
    values = np.array(differences, dtype=np.float64)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return Agreement(len(values), float(values.mean()), sd, float(np.sqrt(np.mean(values ** 2))),
                     float(np.abs(values).max()), int(np.sum(np.abs(values) <= tolerance)))


def find_seams(images: list[BallImage]) -> dict[str, AutomaticSeam]:
    """Run the automatic seam detector on every image, by file name."""
    detector = AutomaticSeamAngleDetector()
    seams = {}
    print(f"Running the automatic seam detector on {len(images)} images")
    for number, image in enumerate(images, start=1):
        started = time.perf_counter()
        seam = detector.find_seam(image)
        time_ms = 1000 * (time.perf_counter() - started)
        name = image.path.name
        if seam is None:
            seams[name] = AutomaticSeam(name, None, None, None, None, 0.0, time_ms)
        else:
            contrast = seam.contrast if isinstance(seam, AutomaticSeamMeasurement) else 0.0
            seams[name] = AutomaticSeam(name, image.to_crop(*seam.start), image.to_crop(*seam.end),
                                        seam.raw_angle_deg, seam.seam_angle_deg, contrast, time_ms)
        if number % 10 == 0 or number == len(images):
            print(f"  {number} of {len(images)} done")
    return seams


def compare(images: list[BallImage], manual: dict[str, ManualSeam],
            automatic: dict[str, AutomaticSeam]) -> list[Comparison]:
    """Put each image's manual, automatic and made seam angles side by side, in image order."""
    details = load_image_details(DETAILS_PATH)
    rows = []
    for image in images:
        name = image.path.name
        made = details.get(name)
        rows.append(Comparison(image, "other" if made is None else made.kind,
                               None if made is None else made.seam_angle_deg, manual.get(name), automatic[name]))
    return rows


def save_automatic(rows: list[Comparison], path: Path) -> None:
    """Write automatic_seam_angles.csv: the seam found on each image."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(AUTOMATIC_COLUMNS)
        for row in rows:
            seam = row.automatic
            start = (None, None) if seam.start is None else seam.start
            end = (None, None) if seam.end is None else seam.end
            writer.writerow([seam.image, "no" if seam.seam_angle_deg is None else "yes",
                             _format(start[0], 2), _format(start[1], 2), _format(end[0], 2), _format(end[1], 2),
                             _format(seam.seam_line_deg, 3), _format(row.image.travel_direction_deg, 3),
                             _format(seam.seam_angle_deg, 3), f"{seam.contrast:.2f}", f"{seam.time_ms:.0f}"])


def save_comparison(rows: list[Comparison], path: Path) -> None:
    """Write seam_angle_comparison.csv: each image's three seam angles and their differences."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COMPARISON_COLUMNS)
        for row in rows:
            writer.writerow([row.image.path.name, row.kind, _format(row.made_deg, 3), _manual_status(row),
                             _format(row.manual_deg, 3), "no seam found" if row.automatic_deg is None else "found",
                             _format(row.automatic_deg, 3), _format(row.difference, 3),
                             _format(_error(row.manual_deg, row.made_deg), 3),
                             _format(_error(row.automatic_deg, row.made_deg), 3), f"{row.automatic.contrast:.2f}"])


def summarise(rows: list[Comparison], tolerance: float) -> list[str]:
    """Return printable lines on how closely the two seam angles agree, overall and for each
    kind of image, and how close each is to the angle the images were made with.
    """
    clicked = sum(1 for row in rows if row.manual is not None and row.manual.seen)
    unseen = sum(1 for row in rows if row.manual is not None and not row.manual.seen)
    found = sum(1 for row in rows if row.automatic_deg is not None)
    groups = [("all images", rows)] + [(name.lower(), [row for row in rows if row.kind == kind])
                                       for kind, (name, _) in _kinds(rows).items()]
    lines = [f"Seam angles on {len(rows)} sample images, agreeing if within +-{tolerance:g} deg",
             f"  By hand: {clicked} clicked, {unseen} with no seam visible, "
             f"{len(rows) - clicked - unseen} not clicked yet",
             f"  Automatic: {found} found, {len(rows) - found} with no seam found",
             "",
             "Automatic minus manual seam angle:"]
    for label, group in groups:
        differences = [row.difference for row in group if row.difference is not None]
        lines.append(_describe(label, agreement(differences, tolerance), tolerance))

    if any(row.made_deg is not None for row in rows):
        lines += ["", "Seam angle minus the angle the image was made with:"]
        methods: list[tuple[str, Callable[[Comparison], float | None]]] = [
            ("By hand", lambda row: row.manual_deg), ("Automatic", lambda row: row.automatic_deg)]
        for method, angle in methods:
            for label, group in groups:
                errors = [error for row in group if (error := _error(angle(row), row.made_deg)) is not None]
                lines.append(_describe(f"{method}, {label}", agreement(errors, tolerance), tolerance))
        if any(row.kind == "photo" for row in rows):
            lines.append("  (The photographed balls' angles come from seam_labels.yaml, measured with the "
                         "automatic detector's stripe method.)")
    return lines


def _describe(label: str, result: Agreement | None, tolerance: float) -> str:
    """Return one summary line for a set of differences."""
    if result is None:
        return f"  {label}: none to compare"
    low, high = result.limits
    return (f"  {label} (n={result.count}): mean {result.mean:+.2f}, sd {result.sd:.2f}, "
            f"95% limits {low:+.2f} to {high:+.2f}, rms {result.rms:.2f}, largest {result.largest:.2f} deg; "
            f"{result.within} of {result.count} ({result.within / result.count:.0%}) within +-{tolerance:g} deg")


def plot_comparison(rows: list[Comparison], tolerance: float, path: Path) -> Figure:
    """Draw and save the chart: the automatic against the manual seam angle, and their difference
    against their midpoint with its mean and 95% limits of agreement.
    """
    paired = [row for row in rows if row.difference is not None]
    stats = agreement([row.difference for row in paired if row.difference is not None], tolerance)
    assert stats is not None
    angles = [angle for row in paired for angle in (row.manual_deg, row.automatic_deg) if angle is not None]
    low = 5 * math.floor(min(angles) / 5) - 5
    high = 5 * math.ceil(max(angles) / 5) + 5

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=SURFACE)
    for axes in (left, right):
        _style(axes)

    # Left: each image's automatic against its manual seam angle, and the line where they are equal.
    left.plot([low, high], [low, high], color=MUTED, linewidth=1, zorder=1, label="Equal angles")
    for kind, (name, colour) in _kinds(paired).items():
        group = [row for row in paired if row.kind == kind]
        left.scatter([row.manual_deg for row in group], [row.automatic_deg for row in group], s=36, color=colour,
                     edgecolors=SURFACE, linewidths=1.2, zorder=3, label=f"{name} ({len(group)})")
    left.set_xlim(low, high)
    left.set_ylim(low, high)
    left.set_aspect("equal")
    left.set_xlabel("Seam angle clicked by hand (°)", color=INK_2, fontsize=11)
    left.set_ylabel("Seam angle found automatically (°)", color=INK_2, fontsize=11)
    left.set_title("Automatic against manual", color=INK, fontsize=12, loc="left")
    left.text(0.97, 0.04, f"n = {stats.count}\nRMS difference {stats.rms:.2f}°", transform=left.transAxes,
              color=INK_2, fontsize=10, ha="right", va="bottom")

    # Right: difference against midpoint (a Bland-Altman plot), with the mean difference and
    # the 95% limits of agreement labelled in the margin so they never cover a point.
    limit_low, limit_high = stats.limits
    reach = max(2 * tolerance, abs(limit_low) + 1, abs(limit_high) + 1, stats.largest + 0.5)
    right.axhspan(-tolerance, tolerance, color=BAND, zorder=0)
    right.axhline(0, color=AXIS, linewidth=1, zorder=1)
    right.axhline(stats.mean, color=INK_2, linewidth=1.2, zorder=2)
    for value in (limit_low, limit_high):
        right.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    for kind, (name, colour) in _kinds(paired).items():
        group = [row for row in paired if row.kind == kind]
        right.scatter([row.midpoint for row in group], [row.difference for row in group], s=36, color=colour,
                      edgecolors=SURFACE, linewidths=1.2, zorder=3)
    right.set_xlim(low, high)
    right.set_ylim(-reach, reach)
    for value, text, colour in ((limit_high, f"{limit_high:+.2f}°  +1.96 SD", MUTED),
                                (stats.mean, f"{stats.mean:+.2f}°  mean", INK_2),
                                (limit_low, f"{limit_low:+.2f}°  −1.96 SD", MUTED)):
        right.annotate(text, xy=(1, value), xycoords=("axes fraction", "data"), xytext=(4, 0),
                       textcoords="offset points", color=colour, fontsize=9, ha="left", va="center",
                       annotation_clip=False)
    right.set_xlabel("Midpoint of the two seam angles (°)", color=INK_2, fontsize=11)
    right.set_ylabel("Automatic minus manual (°)", color=INK_2, fontsize=11)
    right.set_title("Difference against midpoint", color=INK, fontsize=12, loc="left")

    handles, labels = left.get_legend_handles_labels()  # the line comes first; list it after the points
    handles = handles[1:] + handles[:1] + [Patch(facecolor=BAND, edgecolor=AXIS, linewidth=0.6)]
    labels = labels[1:] + labels[:1] + [f"±{tolerance:g}° tolerance"]
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=10,
                  labelcolor=INK_2, bbox_to_anchor=(0.5, 0.995))
    missing = _missing_note(rows)
    if missing:
        figure.text(0.01, 0.01, missing, color=MUTED, fontsize=9, ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    return figure


def _style(axes: Axes) -> None:
    """Give a chart the quiet look of the other report charts: hairline grid, no top or right edge."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelsize=10)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)


def _missing_note(rows: list[Comparison]) -> str:
    """Return a note on the images left out of the chart, or nothing if none were."""
    parts = []
    unseen = sum(1 for row in rows if row.manual is not None and not row.manual.seen)
    unclicked = sum(1 for row in rows if row.manual is None)
    not_found = sum(1 for row in rows if row.automatic_deg is None and row.manual_deg is not None)
    if unseen:
        parts.append(f"{unseen} with no seam visible by hand")
    if unclicked:
        parts.append(f"{unclicked} not clicked yet")
    if not_found:
        parts.append(f"{not_found} with no seam found automatically")
    return "Not shown: " + ", ".join(parts) + "." if parts else ""


def draw_largest_differences(rows: list[Comparison], path: Path, count: int = EXAMPLE_COUNT) -> None:
    """Save the images whose two seam angles differ most, with the seam clicked by hand (cyan)
    and the one found automatically (yellow) drawn on each.
    """
    ranked = sorted((row for row in rows if row.difference is not None),
                    key=lambda row: -abs(row.difference or 0.0))[:count]
    tiles = []
    for row in ranked:
        crop = row.image.load()
        scale = EXAMPLE_SIZE / crop.shape[1]
        tile = cv2.resize(crop, (EXAMPLE_SIZE, EXAMPLE_SIZE), interpolation=cv2.INTER_LINEAR)
        lines = [(row.automatic.start, row.automatic.end, AUTOMATIC_COLOUR)]
        if row.manual is not None:
            lines.append((row.manual.start, row.manual.end, SEAM_COLOUR))
        for start, end, colour in lines:
            if start is None or end is None:
                continue
            ends = [(round((x + 0.5) * scale - 0.5), round((y + 0.5) * scale - 0.5)) for x, y in (start, end)]
            cv2.line(tile, ends[0], ends[1], OUTLINE, 4, cv2.LINE_AA)
            cv2.line(tile, ends[0], ends[1], colour, 2, cv2.LINE_AA)
        manual, automatic, difference = row.manual_deg, row.automatic_deg, row.difference
        assert manual is not None and automatic is not None and difference is not None
        caption = np.full((44, EXAMPLE_SIZE, 3), 255, np.uint8)
        _caption(caption, f"{row.image.path.name} ({row.kind})", 16)
        _caption(caption, f"hand {manual:+.1f}  auto {automatic:+.1f}  diff {difference:+.1f}", 36)
        tiles.append(np.pad(np.vstack([tile, caption]), ((6, 6), (6, 6), (0, 0)), constant_values=255))
    if not tiles:
        return
    blank = np.full_like(tiles[0], 255)
    while len(tiles) % EXAMPLES_PER_ROW:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[start:start + EXAMPLES_PER_ROW])
                      for start in range(0, len(tiles), EXAMPLES_PER_ROW)])
    header = np.full((34, grid.shape[1], 3), 255, np.uint8)
    _caption(header, "Largest differences, in degrees.   cyan: clicked by hand   yellow: found automatically", 22)
    write_image(path, np.vstack([header, grid]))


def _caption(canvas: np.ndarray, text: str, y: int) -> None:
    """Write a line of dark grey text on canvas, 8 px from the left with its baseline at y."""
    cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)


def _kinds(rows: list[Comparison]) -> dict[str, tuple[str, str]]:
    """Return the name and chart colour of each kind of image in rows, in a fixed order."""
    present = {row.kind for row in rows}
    kinds = {kind: style for kind, style in KINDS.items() if kind in present}
    if present - set(KINDS):
        kinds["other"] = OTHER_KIND
    return kinds


def _manual_status(row: Comparison) -> str:
    """Return whether the image's seam was clicked, marked as not visible or not done yet."""
    if row.manual is None:
        return "not clicked"
    return "clicked" if row.manual.seen else "no seam visible"


def _error(angle: float | None, made: float | None) -> float | None:
    """Return angle minus made, folded into (-90, 90] deg, or None if either is missing."""
    return None if angle is None or made is None else fold_line_angle(angle - made)


def _format(value: float | None, decimals: int) -> str:
    """Format a number to decimals places, or leave it blank when there is none."""
    return "" if value is None else f"{value:.{decimals}f}"


def main(argv: list[str] | None = None) -> None:
    """Run the automatic detector on the sample images, compare it with the seams clicked by
    hand, and save the tables, summary and charts.
    """
    parser = argparse.ArgumentParser(description="Compare the seam angles clicked by hand with the "
                                                 "automatic seam angle detector's.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEG,
                        help=f"largest difference counted as agreeing, in degrees (default {TOLERANCE_DEG:g})")
    parser.add_argument("--no-show", action="store_true", help="save the chart without opening it")
    args = parser.parse_args(argv)

    if not (IMAGE_FOLDER / INDEX_FILE_NAME).exists():
        print(f"There are no sample images in {IMAGE_FOLDER} yet. Make them first: "
              "python -m report_data.seam_detector_comparison.make_seam_images")
        raise SystemExit(1)
    manual = load_manual_seams(MANUAL_PATH)
    if not manual:
        print("No seams have been clicked yet. Click them first: "
              "python -m report_data.seam_detector_comparison.click_seams")
        raise SystemExit(1)

    images = load_ball_images(IMAGE_FOLDER)
    rows = compare(images, manual, find_seams(images))
    save_automatic(rows, AUTOMATIC_PATH)
    save_comparison(rows, COMPARISON_PATH)
    lines = summarise(rows, args.tolerance)
    print("\n" + "\n".join(lines))

    saved = [AUTOMATIC_PATH.name, COMPARISON_PATH.name]
    if any(row.difference is not None for row in rows):
        figure = plot_comparison(rows, args.tolerance, CHART_PATH)
        draw_largest_differences(rows, EXAMPLES_PATH)
        saved += [CHART_PATH.name, EXAMPLES_PATH.name]
    else:
        figure = None
        print("\nNo image has both a clicked and an automatic seam, so there is no chart.")
    print(f"\nSaved {', '.join(saved)} to {FOLDER}")
    if figure is not None:
        if not args.no_show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
