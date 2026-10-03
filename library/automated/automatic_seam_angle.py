"""Automatic seam angle detector that finds the seam on every ball image by lining up
its stitching stripes. The seam angles found are averaged, with a warning when they
spread more than MAX_SPREAD_DEG.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from library.detect_seam_angle_file import (BALL_IMAGE_FOLDER, DEFAULT_SAVE_DIR, BallImage,
                                            SeamAngleDetector, SeamMeasurement, fold_line_angle,
                                            load_ball_images, save_seam_measurement)

SEARCH_RANGE_DEG = 45.0
"""Range of angles searched either side of the direction of travel, in degrees."""
COARSE_STEP_DEG = 1.0
"""Step between the angles tried in the first pass over the whole range, in degrees."""
FINE_STEP_DEG = 0.05
"""Step of the second pass, around the best angle of the first, in degrees. Refining
also stops once a pass moves the angle less than this.
"""
COLUMN_FRACTION = 0.6
"""Columns used, as a fraction of the radius either side of the ball's centre."""
EDGE_MARGIN = 0.08
"""Width of the ring left out inside the ball's edge, as a fraction of the radius."""
SHADING_SCALE = 0.075
"""Scale above which shading is ignored in the search, as a fraction of the radius."""
REFINE_RANGE_DEG = 5.0
"""Range of angles tried either side of the levelled seam when refining, in degrees."""
REFINE_PASSES = 2
"""Largest number of refine passes after the search."""
REFINE_SHADING_SCALE = 0.2
"""Scale above which shading is ignored when refining, as a fraction of the radius."""
REFINE_SMOOTHING_SCALE = 0.04
"""Scale below which detail is smoothed when refining, as a fraction of the radius."""
MIN_CONTRAST = 3.0
"""Lowest contrast (sharpest sum over the typical one) for a seam to count as found."""
MAX_SPREAD_DEG = 5.0
"""Largest spread of the per-frame seam angles before a warning is given, in degrees."""
SEAM_LINE_LENGTH = 1.6
"""Length of the seam line returned for each ball image, in ball radii."""


@dataclass
class AutomaticSeamMeasurement(SeamMeasurement):
    """A seam found automatically on one ball image, with how clearly the stitching lined up."""

    contrast: float = 0.0


@dataclass
class AveragedSeamMeasurement(SeamMeasurement):
    """
    The average of the seams found on the ball images. Its line is drawn on
    the image where the stitching was clearest, turned to the average seam
    angle, so seam_angle_deg is the average.
    """

    frame_angles_deg: dict[int, float] = field(default_factory=dict)  # seam angle found on each frame

    @property
    def spread_deg(self) -> float:
        """Largest minus smallest of the seam angles found."""
        middle = average_seam_angle(list(self.frame_angles_deg.values()))
        offsets = [fold_line_angle(angle - middle) for angle in self.frame_angles_deg.values()]
        return max(offsets) - min(offsets) if offsets else 0.0

    @property
    def warning(self) -> str | None:
        """A warning if the seam angles found differ by more than MAX_SPREAD_DEG, otherwise None."""
        if self.spread_deg <= MAX_SPREAD_DEG:
            return None
        return (f"The seam angles found on the {len(self.frame_angles_deg)} ball images differ by "
                f"{self.spread_deg:.1f} deg, more than {MAX_SPREAD_DEG:g} deg, so their average may not "
                f"be reliable. Check the ball images, or pick the seam by hand.")

    def details(self) -> dict[str, object]:
        """Return each frame's angle, the spread and any warning, for the seam file."""
        details: dict[str, object] = {
            "frames_averaged": len(self.frame_angles_deg),
            "seam_angle_spread_deg": round(self.spread_deg, 3),
            "frame_seam_angles_deg": {frame: round(angle, 3) for frame, angle in self.frame_angles_deg.items()},
        }
        if self.warning is not None:
            details["warning"] = self.warning
        return details


@dataclass
class StripeFit:
    """The stitching stripes' angle in a crop and where they cross the ball's middle."""

    angle_deg: float       # atan2(dy, dx) convention, positive clockwise
    contrast: float        # sharpest sum over the median sum of all the angles tried
    band_offset_px: float  # where the stitching band crosses the centre column, below the centre


class AutomaticSeamAngleDetector(SeamAngleDetector):
    """Finds the seam on the cropped ball images without any clicking."""

    method = "automatic"
    """Method name for seams found automatically."""

    def detect(self, ball_images: list[BallImage]) -> SeamMeasurement | None:
        """
        Look for the seam on every ball image and return the average of the seam
        angles found, or None if no seam was found on any of them. Prints a
        warning if the angles differ by more than MAX_SPREAD_DEG.
        """
        print("\nAutomatic seam detection")
        seams = []
        for image in ball_images:
            seam = self.find_seam(image)
            if seam is None:
                print(f"Frame {image.frame_number}: no seam found")
                continue
            angle = "-" if seam.seam_angle_deg is None else f"{seam.seam_angle_deg:+.2f} deg"
            print(f"Frame {image.frame_number}: seam line {seam.raw_angle_deg:+.2f} deg in the image, "
                  f"seam angle {angle}, clarity {_contrast(seam):.1f}")
            seams.append(seam)

        average = self.average_seams(seams)
        if average is None:
            print("No seam found on any ball image.")
            return None
        print(f"Average of {len(seams)} of {len(ball_images)} frames: {_angle(average):+.2f} deg, "
              f"spread {average.spread_deg:.2f} deg")
        if average.warning is not None:
            print(f"WARNING: {average.warning}")
        return average

    def find_seam(self, image: BallImage) -> SeamMeasurement | None:
        """
        Find the seam on one ball image, or return None if it cannot be found.
        The angles tried are within SEARCH_RANGE_DEG of the direction of travel,
        or of the image's x axis when the direction of travel is not known.
        """
        crop = image.load()
        radius = image.diameter_px / 2
        centre = find_ball_centre(crop, image.to_crop(*image.centre), radius)
        travel = 0.0 if image.travel_direction_deg is None else image.travel_direction_deg

        # Search with the direction of travel turned along the x axis.
        fit = fit_stripes(_turned(crop, centre, travel), centre, radius)
        if fit is None or fit.contrast < MIN_CONTRAST:
            return None
        angle_deg = travel + fit.angle_deg
        offset = fit.band_offset_px

        # Measure again with the seam itself turned level (see the notes at the top).
        for _ in range(REFINE_PASSES):
            again = fit_stripes(_turned(crop, centre, angle_deg), centre, radius, REFINE_RANGE_DEG,
                                REFINE_SHADING_SCALE, REFINE_SMOOTHING_SCALE)
            if again is None:
                break
            angle_deg += again.angle_deg
            offset = again.band_offset_px
            if abs(again.angle_deg) < FINE_STEP_DEG:
                break

        # Back in the crop: the seam line runs at the angle through the middle of the stitching band.
        angle = math.radians(angle_deg)
        middle = (centre[0] - offset * math.sin(angle), centre[1] + offset * math.cos(angle))
        half = SEAM_LINE_LENGTH * radius / 2
        start = (middle[0] - half * math.cos(angle), middle[1] - half * math.sin(angle))
        end = (middle[0] + half * math.cos(angle), middle[1] + half * math.sin(angle))
        return AutomaticSeamMeasurement(image, image.to_frame(*start), image.to_frame(*end), self.method,
                                        contrast=fit.contrast)

    def average_seams(self, seams: list[SeamMeasurement]) -> AveragedSeamMeasurement | None:
        """
        The average of the seam angles of seams, or None if there are none. Its
        line is the clearest seam's line turned to the average seam angle.
        """
        if not seams:
            return None
        frame_angles = {seam.frame_number: _angle(seam) for seam in seams}
        average = average_seam_angle(list(frame_angles.values()))
        clearest = max(seams, key=_contrast)
        travel = clearest.image.travel_direction_deg
        line = math.radians(average + (0.0 if travel is None else travel))
        middle = ((clearest.start[0] + clearest.end[0]) / 2, (clearest.start[1] + clearest.end[1]) / 2)
        half = clearest.length_px / 2
        start = (middle[0] - half * math.cos(line), middle[1] - half * math.sin(line))
        end = (middle[0] + half * math.cos(line), middle[1] + half * math.sin(line))
        return AveragedSeamMeasurement(clearest.image, start, end, self.method, frame_angles_deg=frame_angles)


def average_seam_angle(angles: list[float]) -> float:
    """
    The average of seam angles in degrees. A seam line has no direction, so
    +89 and -89 degrees are nearly the same seam. The angles are measured from
    their middle direction (found by doubling them) before averaging, so seams
    either side of 90 degrees average correctly. Otherwise it is the ordinary
    average.
    """
    if not angles:
        return 0.0
    sine = sum(math.sin(math.radians(2 * angle)) for angle in angles)
    cosine = sum(math.cos(math.radians(2 * angle)) for angle in angles)
    middle = math.degrees(math.atan2(sine, cosine)) / 2
    offsets = [fold_line_angle(angle - middle) for angle in angles]
    return fold_line_angle(middle + sum(offsets) / len(offsets))


def _turned(crop: np.ndarray, centre: tuple[float, float], angle_deg: float) -> np.ndarray:
    """crop turned about centre so that a line at angle_deg becomes level."""
    if angle_deg == 0.0:
        return crop
    matrix = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)  # anticlockwise by angle_deg
    return cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def find_ball_centre(crop: np.ndarray, centre: tuple[float, float], radius: float) -> tuple[float, float]:
    """
    The ball's centre in the crop, found from its red colour around the
    clicked centre. If the ball cannot be picked out clearly, the clicked
    centre is kept.
    """
    height, width = crop.shape[:2]
    bgr = crop.astype(np.float32)
    redness = bgr[..., 2] - np.maximum(bgr[..., 0], bgr[..., 1])

    reach = max(2, round(radius / 2))
    x, y = round(centre[0]), round(centre[1])
    core = redness[max(y - reach, 0):y + reach, max(x - reach, 0):x + reach]
    if core.size == 0:
        return centre
    level = float(np.percentile(core, 60))
    if level < 5:
        return centre

    mask: np.ndarray = (redness > level / 2).astype(np.uint8)
    small = 2 * max(1, round(radius * 0.05)) + 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small, small)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * small, 3 * small)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return centre
    inside = 0 <= x < width and 0 <= y < height
    label = labels[y, x] if inside and labels[y, x] else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    moments = cv2.moments(filled, binaryImage=True)
    if moments["m00"] == 0:
        return centre

    found = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    found_radius = math.sqrt(moments["m00"] / math.pi)
    if math.dist(found, centre) > 0.35 * radius or not 0.7 < found_radius / radius < 1.3:
        return centre
    return found


def fit_stripes(crop: np.ndarray, centre: tuple[float, float], radius: float,
                search_range_deg: float = SEARCH_RANGE_DEG, shading_scale: float = SHADING_SCALE,
                smoothing_scale: float = 0.0) -> StripeFit | None:
    """
    Angle of the stitching stripes in crop, within search_range_deg of the
    x axis. centre and radius are the ball's, in crop pixels. Shading smoother
    than shading_scale, and detail finer than smoothing_scale (both fractions
    of the radius), are left out.
    """
    height, width = crop.shape[:2]
    cx, cy = centre
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    distance = np.hypot(cols - cx, rows - cy)
    inner = radius * (1 - EDGE_MARGIN)
    inside = distance <= inner

    # White stitching is bright in green and blue, the red leather is dark in both.
    image = crop[..., :2].astype(np.float64).mean(axis=2)
    stripes = _high_pass_down_columns(image, inside, shading_scale * radius)
    if smoothing_scale > 0:
        stripes = _smooth_down_columns(stripes, smoothing_scale * radius)
    # Fade to zero at the ball's edge, so the shifts below see no edge.
    fade = np.clip((radius - distance) / max(radius - inner, 1e-6), 0, 1)
    stripes *= fade * fade * (3 - 2 * fade)

    used = np.array([c for c in range(width) if abs(c - cx) <= COLUMN_FRACTION * radius], dtype=np.int64)
    if len(used) < 5:
        return None
    across = used - cx
    chord = np.sqrt(np.maximum(inner ** 2 - across ** 2, 0.0))
    top, bottom = cy - chord, cy + chord
    spectrum = np.fft.rfft(stripes[:, used], axis=0)
    frequency = np.fft.rfftfreq(height)[:, None]
    row = np.arange(height, dtype=np.float64)[:, None]

    def score(angle_deg: float) -> tuple[float, float]:
        """
        How well the columns line up at angle_deg: the energy of their sum
        (for the contrast), and the same without each column's own energy, so
        only agreement between columns counts (for the angle). Each column's
        own energy, noise included, does not depend on the angle, but how much
        of it is counted does, and that would pull faint seams towards 0.
        """
        shift = math.tan(math.radians(angle_deg)) * across
        # Exact sub-pixel shifts. Linear interpolation would blur every column
        # except at 0 degrees and so also pull nearly level seams to 0.
        shifted = np.fft.irfft(spectrum * np.exp(2j * np.pi * frequency * shift), n=height, axis=0)
        valid = (row + shift >= top) & (row + shift <= bottom)
        inside_values = shifted * valid
        total = inside_values.sum(axis=1)
        own = (inside_values ** 2).sum(axis=1)
        count = valid.sum(axis=1)
        keep = count >= 2
        energy = float(np.sum(total[keep] ** 2 / count[keep]))
        agreement = float(np.sum((total[keep] ** 2 - own[keep]) / count[keep]))
        return energy, agreement

    coarse = np.arange(-search_range_deg, search_range_deg + 1e-9, COARSE_STEP_DEG)
    coarse_energy, coarse_scores = np.array([score(a) for a in coarse]).T
    best = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.arange(best - COARSE_STEP_DEG, best + COARSE_STEP_DEG + 1e-9, FINE_STEP_DEG)
    fine_energy, fine_scores = np.array([score(a) for a in fine]).T
    k = int(np.argmax(fine_scores))
    angle = float(fine[k])
    if 0 < k < len(fine) - 1:  # parabola through the top three points
        before, peak, after = fine_scores[k - 1:k + 2]
        curve = before - 2 * peak + after
        if curve < 0:
            angle += FINE_STEP_DEG * 0.5 * (before - after) / curve

    typical = float(np.median(coarse_energy))
    contrast = float(fine_energy[k] / typical) if typical > 0 else 0.0
    return StripeFit(angle, contrast, _band_offset(image, used, across, top, bottom, angle, cy))


def _smooth_down_columns(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing down each column only."""
    reach = max(1, math.ceil(3 * sigma))
    kernel = cv2.getGaussianKernel(2 * reach + 1, sigma)
    return cv2.sepFilter2D(image, -1, np.array([[1.0]]), kernel, borderType=cv2.BORDER_REFLECT)


def _high_pass_down_columns(image: np.ndarray, inside: np.ndarray, sigma: float) -> np.ndarray:
    """image minus its smooth shading down each column, using only pixels inside the ball."""
    weight = inside.astype(np.float64)
    smooth = _smooth_down_columns(image * weight, sigma) / np.maximum(_smooth_down_columns(weight, sigma), 1e-6)
    return cast(np.ndarray, image - smooth)


def _band_offset(image: np.ndarray, used: np.ndarray, across: np.ndarray, top: np.ndarray,
                 bottom: np.ndarray, angle_deg: float, cy: float) -> float:
    """How far below the centre the stitching band crosses the centre column, in pixels."""
    height = image.shape[0]
    shift = np.round(math.tan(math.radians(angle_deg)) * across).astype(np.int64)
    total = np.zeros(height)
    count = np.zeros(height)
    for column, step, low, high in zip(used, shift, top, bottom):
        rows = np.arange(height) + step
        ok = (rows >= low) & (rows <= high) & (rows >= 0) & (rows < height)
        total[ok] += image[rows[ok], column]
        count[ok] += 1
    valid = count >= len(used) / 2
    if not valid.any():
        return 0.0
    profile = np.where(valid, total / np.maximum(count, 1), 0.0)
    base = np.percentile(profile[valid], 25)
    peak = profile[valid].max()
    weight = np.where(valid, np.clip(profile - (base + 0.35 * (peak - base)), 0, None), 0.0)
    if weight.sum() == 0:
        return 0.0
    return float(np.sum(weight * np.arange(height)) / weight.sum() - cy)


def _contrast(seam: SeamMeasurement) -> float:
    """Return how clearly a seam's stitching lined up, or 1 for other kinds of seam."""
    return seam.contrast if isinstance(seam, AutomaticSeamMeasurement) else 1.0


def _angle(seam: SeamMeasurement) -> float:
    """Return the seam angle, or the seam line's image angle when travel is unknown."""
    return seam.raw_angle_deg if seam.seam_angle_deg is None else seam.seam_angle_deg


def main(argv: list[str] | None = None) -> None:
    """Run the automatic detector on a folder of ball images from an earlier run."""
    arguments = sys.argv[1:] if argv is None else argv
    folder = Path(arguments[0]) if arguments else Path(DEFAULT_SAVE_DIR) / BALL_IMAGE_FOLDER
    seam = AutomaticSeamAngleDetector().detect(load_ball_images(folder))
    if seam is None:
        print("No seam found.")
        return
    angle = "-" if seam.seam_angle_deg is None else f"{seam.seam_angle_deg:+.2f} deg"
    frames = len(seam.frame_angles_deg) if isinstance(seam, AveragedSeamMeasurement) else 1
    print(f"Calculated seam angle: {angle} (average of {frames} frames)")
    print(f"Seam saved to {save_seam_measurement(seam)}")


if __name__ == "__main__":
    main()
