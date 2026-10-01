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
TOLERANCE_DEG = 2.5
CLICK_ERROR_PX = 2.0
BLUR_RANGE = (0.10, 0.30)  # motion blur length of the drawn balls, as a fraction of the ball's width

# Chart colours (light theme)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = "#2a78d6"
BAND = "#f0efec"


@dataclass
class TestImage(BallImage):
    """A ball image held in memory instead of in a file."""

    pixels: np.ndarray | None = None

    def load(self) -> np.ndarray:
        assert self.pixels is not None
        return self.pixels


@dataclass
class Result:
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
        return None if self.found_deg is None else fold_line_angle(self.found_deg - self.true_deg)


def ball_image_tests(count: int, angle_range: tuple[float, float], rng: np.random.Generator,
                     folders: list[Path], click_error: float = CLICK_ERROR_PX) -> list[Result]:
    """Run the detector on the labelled ball images turned to seam angles within angle_range."""
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
        results.append(Result("ball images", index, example.seam_angle_deg, found, _contrast(seam),
                              example.image, centre, 0.0))
    return results


def drawn_ball_tests(count: int, angle_range: tuple[float, float], rng: np.random.Generator,
                     click_error: float = CLICK_ERROR_PX,
                     blur_range: tuple[float, float] = BLUR_RANGE) -> list[Result]:
    """Run the detector on drawn balls with the seam at a known angle to the direction of travel."""
    detector = AutomaticSeamAngleDetector()
    results = []
    for index in range(count):
        seam_angle = float(rng.uniform(*angle_range))
        travel = float(rng.uniform(-180.0, 180.0))
        pixels, centre = draw_ball(rng, seam_angle, travel, blur_range)
        given = (centre[0] + rng.uniform(-click_error, click_error),
                 centre[1] + rng.uniform(-click_error, click_error))
        image = TestImage(index, Path(f"drawn_{index}.png"), given, BALL_DIAMETER, (0, 0), IMAGE_SIZE, travel,
                          pixels=pixels)
        seam = detector.find_seam(image)
        found = None if seam is None else seam.seam_angle_deg
        results.append(Result("drawn balls", index, seam_angle, found, _contrast(seam), pixels, centre, travel))
    return results


def draw_ball(rng: np.random.Generator, seam_angle: float, travel: float,
              blur_range: tuple[float, float] = BLUR_RANGE) -> tuple[np.ndarray, tuple[float, float]]:
    """
    A 200 x 200 ball like the training images: red leather, a band of white
    stitching at seam_angle to the direction of travel, motion blur along the
    direction of travel, and black outside the ball. Returns it and its centre.
    """
    size, radius = IMAGE_SIZE, BALL_DIAMETER / 2
    middle = (size - 1) / 2
    centre = (middle + rng.uniform(-MAX_SHIFT, MAX_SHIFT), middle + rng.uniform(-MAX_SHIFT, MAX_SHIFT))
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = cols - centre[0], rows - centre[1]
    distance = np.hypot(dx, dy)

    # Leather: dark red, darker towards the edge, lit a little more from one side, with some mottling.
    leather = np.array([rng.uniform(25, 45), rng.uniform(18, 35), rng.uniform(95, 140)])
    light = math.radians(rng.uniform(0, 360))
    shade = (1 - 0.45 * (distance / radius) ** 2) * (1 + 0.15 * (dx * math.cos(light) + dy * math.sin(light)) / radius)
    mottle = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), radius * 0.15)
    shade = shade * (1 + 0.08 * mottle / max(mottle.std(), 1e-6))
    colour = shade[..., None] * leather

    # Stitching: 4-6 rows of stitches along a slightly curved band, at the seam's angle in the image.
    line = math.radians(travel + seam_angle)
    along = dx * math.cos(line) + dy * math.sin(line)
    across = -dx * math.sin(line) + dy * math.cos(line)
    band = rng.uniform(-0.3, 0.3) * radius + rng.uniform(-0.12, 0.12) * radius * (along / radius) ** 2
    rows_count = int(rng.integers(4, 7))
    spacing = rng.uniform(0.085, 0.12) * radius
    width = max(0.8, rng.uniform(0.012, 0.02) * radius)
    period = rng.uniform(0.04, 0.08) * radius
    stitching = np.zeros((size, size))
    for k in range(rows_count):
        offset = (k - (rows_count - 1) / 2) * spacing
        stitches = 0.75 + 0.25 * np.sin(2 * np.pi * along / period + rng.uniform(0, 2 * np.pi))
        row = rng.uniform(0.6, 1.0) * stitches * np.exp(-0.5 * ((across - band - offset) / width) ** 2)
        stitching = np.maximum(stitching, row)
    thread = np.array([200.0, 205.0, 232.0]) * rng.uniform(0.8, 1.0)
    colour = colour * (1 - stitching[..., None]) + thread * np.clip(shade, 0.4, 1.2)[..., None] * stitching[..., None]

    # Motion blur along the direction of travel, over a green or grey background.
    alpha = np.clip(radius + 0.5 - distance, 0, 1)
    background = np.array([70.0, 110.0, 75.0]) if rng.random() < 0.5 else np.array([60.0, 62.0, 64.0])
    length = rng.uniform(*blur_range) * 2 * radius
    kernel = _line_kernel(length, travel)
    smeared = cv2.filter2D(colour * alpha[..., None], -1, kernel)
    coverage = cv2.filter2D(alpha, -1, kernel)
    image = smeared + (1 - coverage[..., None]) * background

    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.6, 1.2)) + rng.normal(0, rng.uniform(1.5, 3.5), image.shape)
    image[distance > radius] = 0
    return np.clip(image + 0.5, 0, 255).astype(np.uint8), centre


def _line_kernel(length: float, angle_deg: float) -> np.ndarray:
    """A motion blur kernel: a line of the given length at angle_deg, adding up to 1."""
    half = int(math.ceil(length / 2)) + 2
    kernel = np.zeros((2 * half + 1, 2 * half + 1), np.float64)
    a = math.radians(angle_deg)
    for t in np.linspace(-length / 2, length / 2, max(2, int(length * 4))):
        x, y = half + t * math.cos(a), half + t * math.sin(a)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        kernel[y0, x0] += (1 - fx) * (1 - fy)
        kernel[y0, x0 + 1] += fx * (1 - fy)
        kernel[y0 + 1, x0] += (1 - fx) * fy
        kernel[y0 + 1, x0 + 1] += fx * fy
    return kernel / kernel.sum()


def _contrast(seam: object) -> float:
    return float(getattr(seam, "contrast", 0.0)) if seam is not None else 0.0


def summarise(results: list[Result], angle_range: tuple[float, float], tolerance: float) -> list[str]:
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


def plot_errors(sets: list[list[Result]], angle_range: tuple[float, float], tolerance: float, path: Path) -> None:
    plt.switch_backend("Agg")  # draw to a file, no window
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    limit = 3 * tolerance
    for ax, results in zip(axes, sets):
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
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.text(angle_range[1], tolerance + 0.15, f"±{tolerance:g}° tolerance", color=MUTED, fontsize=8,
                ha="right", va="bottom")
    axes[0].set_ylabel("Found minus true (°)", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)


def draw_worst(sets: list[list[Result]], path: Path, per_set: int = 6) -> None:
    """The worst cases of each set, with the true seam (white dashes) and the found seam (yellow)."""
    tiles = []
    for results in sets:
        ranked = sorted(results, key=lambda r: -1 if r.error is None else -abs(r.error))[:per_set]
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
        while len(row) < per_set:
            row.append(np.full((IMAGE_SIZE + 26, IMAGE_SIZE, 3), 255, np.uint8))
        tiles.append(np.hstack([np.pad(t, ((6, 6), (6, 6), (0, 0)), constant_values=255) for t in row]))
    cv2.imwrite(str(path), np.vstack(tiles))


def _dashed(image: np.ndarray, centre: tuple[float, float], angle: float, reach: float,
            colour: tuple[int, int, int]) -> None:
    for start in np.arange(-reach, reach, 9.0):
        end = min(start + 5.0, reach)
        p = (round(centre[0] + start * math.cos(angle)), round(centre[1] + start * math.sin(angle)))
        q = (round(centre[0] + end * math.cos(angle)), round(centre[1] + end * math.sin(angle)))
        cv2.line(image, p, q, colour, 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test the automatic seam angle detector.")
    parser.add_argument("count", nargs="?", type=int, default=500, help="test images in each set (default 500)")
    parser.add_argument("--range", nargs=2, type=float, default=(-30.0, 30.0), metavar=("MIN", "MAX"),
                        help="seam angles to test, in degrees (default -30 30)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEG,
                        help=f"error counted as right, in degrees (default {TOLERANCE_DEG:g})")
    parser.add_argument("--folders", nargs="+", type=Path, default=DEFAULT_FOLDERS,
                        help="ball image folders with a seam_labels.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for the results")
    parser.add_argument("--click-error", type=float, default=CLICK_ERROR_PX,
                        help=f"most the given ball centre is off each way, in pixels (default {CLICK_ERROR_PX:g})")
    parser.add_argument("--blur", nargs=2, type=float, default=BLUR_RANGE, metavar=("MIN", "MAX"),
                        help="motion blur of the drawn balls, as a fraction of the ball's width (default 0.1 0.3)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    args = parser.parse_args(argv)
    angle_range = (float(args.range[0]), float(args.range[1]))
    if not -90 <= angle_range[0] < angle_range[1] <= 90:
        parser.error("--range needs -90 <= MIN < MAX <= 90")

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    blur_range = (float(args.blur[0]), float(args.blur[1]))
    sets = [ball_image_tests(args.count, angle_range, rng, args.folders, args.click_error),
            drawn_ball_tests(args.count, angle_range, rng, args.click_error, blur_range)]
    seconds = time.perf_counter() - started

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_set", "index", "true_seam_deg", "found_seam_deg", "error_deg", "contrast"])
        for results in sets:
            for r in results:
                writer.writerow([r.test_set, r.index, f"{r.true_deg:.3f}",
                                 "" if r.found_deg is None else f"{r.found_deg:.3f}",
                                 "" if r.error is None else f"{r.error:.3f}", f"{r.contrast:.2f}"])
    plot_errors(sets, angle_range, args.tolerance, output / "errors.png")
    draw_worst(sets, output / "worst_cases.png")

    print(f"\nSeam angles from {angle_range[0]:+g} to {angle_range[1]:+g} deg, {args.count} images per set, "
          f"centre off by up to {args.click_error:g} px, drawn balls blurred by {blur_range[0]:.0%}-"
          f"{blur_range[1]:.0%} of their width, seed {args.seed} ({1000 * seconds / (2 * args.count):.0f} ms per image)\n")
    for results in sets:
        print("\n".join(summarise(results, angle_range, args.tolerance)))
        print()
    print(f"Saved results.csv, errors.png and worst_cases.png to {output}")


if __name__ == "__main__":
    main()
