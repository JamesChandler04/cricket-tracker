"""Tests how accurately the automatic seam angle detector finds known seam angles.

The detector is run on photographed balls: labelled ball images turned to set seam angles,
shifted and re-shaded.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from generate_seam_training_data import (BALL_DIAMETER, DEFAULT_FOLDERS, IMAGE_SIZE, MAX_SHIFT,
                                         load_labelled_balls, make_example, pick_rotation)
from library import paths
from library.automated.automatic_seam_angle import AutomaticSeamAngleDetector
from library.detect_seam_angle_file import BallImage, fold_line_angle

DEFAULT_OUTPUT_DIR = paths.OUTPUT_DIR / "seam_detector_test"
"""Folder the results are saved to by default."""
TOLERANCE_DEG = 2.5
"""Default largest error counted as a correct seam angle, in degrees."""
CLICK_ERROR_PX = 2.0
"""Default largest error added to each coordinate of the given ball centre, in px."""
TEST_SET = "photographed balls"
"""Name of the test set in results.csv, the chart and the summary."""

# Chart colours
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
"""Colour of the error dots on the chart."""
BAND = "#f0efec"
"""Colour of the shaded tolerance band on the chart."""


@dataclass
class TestImage(BallImage):
    """A ball image held in memory instead of in a file."""

    pixels: np.ndarray | None = None

    def load(self) -> np.ndarray:
        """Return the pixels held in memory."""
        assert self.pixels is not None
        return self.pixels


@dataclass
class Result:
    """Outcome of one test: the true and found seam angles and the image tested."""
    test_set: str
    index: int
    true_deg: float
    found_deg: float | None
    contrast: float
    image: np.ndarray
    centre: tuple[float, float]
    reference_deg: float  # angle the true and found seam angles are measured from in the image

    @property
    def error(self) -> float | None:
        """Return the found minus the true seam angle, folded into (-90, 90] deg, or
        None if no seam was found.
        """
        return None if self.found_deg is None else fold_line_angle(self.found_deg - self.true_deg)


def photographed_ball_tests(count: int, angle_range: tuple[float, float], rng: np.random.Generator,
                            folders: list[Path], click_error: float = CLICK_ERROR_PX) -> list[Result]:
    """Run the detector on photographed balls: the labelled ball images turned to seam angles
    within angle_range, shifted and re-shaded.
    """
    balls = [ball for folder in folders for ball in load_labelled_balls(folder)]
    detector = AutomaticSeamAngleDetector()
    middle = (IMAGE_SIZE - 1) / 2
    results = []
    for index in range(count):
        ball = balls[index % len(balls)]
        example = make_example(ball, rng, MAX_SHIFT, pick_rotation(ball, rng, angle_range))
        centre = (middle + example.shift_px[0], middle + example.shift_px[1])
        given = (centre[0] + rng.uniform(-click_error, click_error),
                 centre[1] + rng.uniform(-click_error, click_error))
        # The direction of travel is not known here (the blur turned with the
        # ball), so the detector measures the seam line from the image's x axis.
        image = TestImage(index, Path(f"ball_{index}.png"), given, BALL_DIAMETER, (0, 0), IMAGE_SIZE, None,
                          pixels=example.image)
        seam = detector.find_seam(image)
        found = None if seam is None else seam.raw_angle_deg
        results.append(Result(TEST_SET, index, example.seam_angle_deg, found, _contrast(seam),
                              example.image, centre, 0.0))
    return results


def _contrast(seam: object) -> float:
    """Return the contrast of a found seam, or 0 if no seam was found."""
    return float(getattr(seam, "contrast", 0.0)) if seam is not None else 0.0


def summarise(results: list[Result], angle_range: tuple[float, float], tolerance: float) -> list[str]:
    """Return printable lines on the detector's accuracy, overall and for each 10 deg band
    of true seam angle.
    """
    errors = np.array([r.error for r in results if r.error is not None])
    within = sum(1 for r in results if r.error is not None and abs(r.error) <= tolerance)
    missed = sum(1 for r in results if r.error is None)
    lines = [f"{results[0].test_set}: {within / len(results):.1%} within +-{tolerance:g} deg "
             f"({within} of {len(results)}, no seam found {missed})"]
    if len(errors):
        lines.append(f"  error: mean {errors.mean():+.2f}, spread (sd) {errors.std():.2f}, "
                     f"95% within +-{np.percentile(np.abs(errors), 95):.2f}, worst {np.abs(errors).max():.2f} deg")
    edges = np.arange(angle_range[0], angle_range[1] + 1e-9, 10.0)
    lines.append("  by seam angle:   " + "  ".join(f"{a:+.0f} to {b:+.0f}" for a, b in zip(edges[:-1], edges[1:])))
    shares = []
    for a, b in zip(edges[:-1], edges[1:]):
        group = [r for r in results if a <= r.true_deg < b or (b == edges[-1] and r.true_deg == b)]
        ok = sum(1 for r in group if r.error is not None and abs(r.error) <= tolerance)
        shares.append(f"{ok / len(group):>9.0%}" if group else f"{'-':>9}")
    lines.append("  within tolerance:" + " ".join(shares))
    return lines


def plot_errors(results: list[Result], angle_range: tuple[float, float], tolerance: float, path: Path) -> None:
    """Save a chart of the errors against the true seam angle, with the tolerance band shaded."""
    plt.switch_backend("Agg")  # draw to a file, no window
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor=SURFACE)
    limit = 3 * tolerance
    ax.set_facecolor(SURFACE)
    ax.axhspan(-tolerance, tolerance, color=BAND, zorder=0)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    found = [r for r in results if r.error is not None]
    x = np.array([r.true_deg for r in found])
    y = np.clip(np.array([r.error for r in found], dtype=float), -limit, limit)
    ax.scatter(x, y, s=10, color=SERIES, alpha=0.55, linewidths=0, zorder=2)
    within = sum(1 for r in found if abs(r.error or 0) <= tolerance)
    ax.set_title(f"{results[0].test_set.capitalize()}: {within / len(results):.1%} within ±{tolerance:g}°",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(angle_range[0] - 1, angle_range[1] + 1)
    ax.set_ylim(-limit - 0.3, limit + 0.3)
    ax.set_xlabel("True seam angle (°)", color=INK_2)
    ax.set_ylabel("Found minus true (°)", color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.text(angle_range[1], tolerance + 0.15, f"±{tolerance:g}° tolerance", color=MUTED, fontsize=8,
            ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def draw_worst(results: list[Result], path: Path, count: int = 6) -> None:
    """Save the worst cases in a row, with the true seam (white dashes) and the found seam (yellow)."""
    ranked = sorted(results, key=lambda r: -1 if r.error is None else -abs(r.error))[:count]
    row = []
    for r in ranked:
        tile = r.image.copy()
        reach = BALL_DIAMETER * 0.62
        _dashed(tile, r.centre, math.radians(r.reference_deg + r.true_deg), reach, (255, 255, 255))
        if r.found_deg is not None:
            a = math.radians(r.reference_deg + r.found_deg)
            p = (round(r.centre[0] - reach * math.cos(a)), round(r.centre[1] - reach * math.sin(a)))
            q = (round(r.centre[0] + reach * math.cos(a)), round(r.centre[1] + reach * math.sin(a)))
            cv2.line(tile, p, q, (0, 230, 255), 1, cv2.LINE_AA)
        label = "no seam" if r.error is None else f"true {r.true_deg:+.1f}  error {r.error:+.1f}"
        canvas = np.full((IMAGE_SIZE + 26, IMAGE_SIZE, 3), 255, np.uint8)
        canvas[:IMAGE_SIZE] = tile
        cv2.putText(canvas, label, (6, IMAGE_SIZE + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1,
                    cv2.LINE_AA)
        row.append(canvas)
    while len(row) < count:
        row.append(np.full((IMAGE_SIZE + 26, IMAGE_SIZE, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.hstack([np.pad(t, ((6, 6), (6, 6), (0, 0)), constant_values=255) for t in row]))


def _dashed(image: np.ndarray, centre: tuple[float, float], angle: float, reach: float,
            colour: tuple[int, int, int]) -> None:
    """Draw a dashed line through centre at angle (in radians), reach px either side."""
    for start in np.arange(-reach, reach, 9.0):
        end = min(start + 5.0, reach)
        p = (round(centre[0] + start * math.cos(angle)), round(centre[1] + start * math.sin(angle)))
        q = (round(centre[0] + end * math.cos(angle)), round(centre[1] + end * math.sin(angle)))
        cv2.line(image, p, q, colour, 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> None:
    """Run the test, save results.csv, errors.png and worst_cases.png, and print a summary."""
    parser = argparse.ArgumentParser(description="Test the automatic seam angle detector on photographed balls.")
    parser.add_argument("count", nargs="?", type=int, default=500, help="test images (default 500)")
    parser.add_argument("--range", nargs=2, type=float, default=(-30.0, 30.0), metavar=("MIN", "MAX"),
                        help="seam angles to test, in degrees (default -30 30)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEG,
                        help=f"error counted as right, in degrees (default {TOLERANCE_DEG:g})")
    parser.add_argument("--folders", nargs="+", type=Path, default=DEFAULT_FOLDERS,
                        help="ball image folders with a seam_labels.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for the results")
    parser.add_argument("--click-error", type=float, default=CLICK_ERROR_PX,
                        help=f"most the given ball centre is off each way, in pixels (default {CLICK_ERROR_PX:g})")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    args = parser.parse_args(argv)
    angle_range = (float(args.range[0]), float(args.range[1]))
    if not -90 <= angle_range[0] < angle_range[1] <= 90:
        parser.error("--range needs -90 <= MIN < MAX <= 90")
    if args.count < 1:
        parser.error("test at least one image")

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    results = photographed_ball_tests(args.count, angle_range, rng, args.folders, args.click_error)
    seconds = time.perf_counter() - started

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_set", "index", "true_seam_deg", "found_seam_deg", "error_deg", "contrast"])
        for r in results:
            writer.writerow([r.test_set, r.index, f"{r.true_deg:.3f}",
                             "" if r.found_deg is None else f"{r.found_deg:.3f}",
                             "" if r.error is None else f"{r.error:.3f}", f"{r.contrast:.2f}"])
    plot_errors(results, angle_range, args.tolerance, output / "errors.png")
    draw_worst(results, output / "worst_cases.png")

    print(f"\nSeam angles from {angle_range[0]:+g} to {angle_range[1]:+g} deg, {args.count} images of photographed "
          f"balls, centre off by up to {args.click_error:g} px, seed {args.seed} "
          f"({1000 * seconds / args.count:.0f} ms per image)\n")
    print("\n".join(summarise(results, angle_range, args.tolerance)))
    print(f"\nSaved results.csv, errors.png and worst_cases.png to {output}")


if __name__ == "__main__":
    main()
