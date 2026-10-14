"""Makes sample ball images at known seam angles from photographed balls, for comparing the
automatic seam angle detector with seams clicked by hand.

Each image is a labelled ball image turned, shifted and re-shaded. They are saved like the
GUI's ball images, and the seam angle each was made with is kept in a separate file so the
clicking stays blind.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Started as a plain script (python report_data/seam_detector_comparison/<name>.py,
    # or the Run button in VS Code), so put the project folder on the path for the
    # imports. python -m report_data.seam_detector_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_seam_detector import CLICK_ERROR_PX
from generate_seam_training_data import (BALL_DIAMETER, DEFAULT_FOLDERS, IMAGE_SIZE, MAX_SHIFT,
                                         LabelledBall, _project_path, load_labelled_balls,
                                         make_example, write_image)
from library import paths
from library.automated.automatic_seam_angle import SEARCH_RANGE_DEG
from library.detect_seam_angle_file import CROP_SCALE, BallImage, _write_index, fold_line_angle

FOLDER = paths.REPORT_DATA_DIR / "seam_detector_comparison"
"""Folder of the comparison's scripts, sample images and results."""
IMAGE_FOLDER = FOLDER / "ball_images"
"""Folder of the sample images and the ball_images.yaml index the seam detectors read."""
DETAILS_PATH = FOLDER / "image_details.csv"
"""CSV of how each sample image was made, including the seam angle it was made with."""
MANUAL_PATH = FOLDER / "manual_seam_angles.csv"
"""CSV of the seams clicked by hand, written by click_seams.py."""
IMAGE_NAME = "ball_{number:03d}.png"
"""File name pattern for a sample image, from its number."""
INDEX_SOURCE = "sample images made by report_data/seam_detector_comparison/make_seam_images.py"
"""Text saved as the video in ball_images.yaml, saying where the images came from."""

PHOTO_COUNT = 100
"""Default number of sample images."""
SEAM_RANGE = (-30.0, 30.0)
"""Default range of seam angles to the direction of travel, in degrees."""

DETAIL_COLUMNS = ["image", "seam_angle_deg", "seam_line_deg", "travel_direction_deg", "ball_centre_x_px",
                  "ball_centre_y_px", "given_centre_x_px", "given_centre_y_px", "photo", "rotation_deg"]
"""Column headings of image_details.csv."""


@dataclass
class SampleImage:
    """One sample image and how it was made."""

    number: int
    pixels: np.ndarray
    seam_angle_deg: float               # to the direction of travel, positive clockwise
    travel_direction_deg: float
    centre: tuple[float, float]         # the ball's real centre, in px
    given_centre: tuple[float, float]   # centre saved in the index, off by a click error
    photo: str                          # labelled ball image the image was made from
    rotation_deg: float                 # how far that ball image was turned, clockwise

    @property
    def name(self) -> str:
        """Return the image's file name."""
        return IMAGE_NAME.format(number=self.number)

    @property
    def seam_line_deg(self) -> float:
        """Return the angle of the seam line in the image, folded into (-90, 90]."""
        return fold_line_angle(self.travel_direction_deg + self.seam_angle_deg)


@dataclass
class ImageDetails:
    """How a sample image was made, as read back from image_details.csv."""

    name: str
    photographed: bool   # False only for a drawn ball made by an earlier version
    seam_angle_deg: float
    travel_direction_deg: float


def spread_angles(count: int, seam_range: tuple[float, float], rng: np.random.Generator) -> list[float]:
    """Return count seam angles covering seam_range evenly, one at random in each of count
    equal slices of it.
    """
    low, high = seam_range
    width = (high - low) / count if count else 0.0
    return [low + (slot + float(rng.random())) * width for slot in range(count)]


def make_photo(number: int, ball: LabelledBall, seam_angle: float, rng: np.random.Generator,
               click_error: float) -> SampleImage:
    """Turn a photographed ball so its seam is at seam_angle to a random direction of travel."""
    travel = float(rng.uniform(-180.0, 180.0))
    turned_over = 180.0 if rng.random() < 0.5 else 0.0  # same seam line, ball the other way round
    rotation = (travel + seam_angle - ball.seam_angle_deg + turned_over) % 360.0
    # The photograph's motion blur turns with the ball, so unlike in a real frame it no
    # longer runs along the direction of travel.
    example = make_example(ball, rng, MAX_SHIFT, rotation)
    middle = (IMAGE_SIZE - 1) / 2
    centre = (middle + example.shift_px[0], middle + example.shift_px[1])
    return SampleImage(number, example.image, seam_angle, travel, centre, _clicked(centre, rng, click_error),
                       _project_path(ball.image.path), rotation)


def make_samples(balls: list[LabelledBall], count: int, seam_range: tuple[float, float],
                 rng: np.random.Generator, click_error: float) -> list[SampleImage]:
    """Make count sample images with seam angles spread evenly over seam_range, in a random
    order, using every photographed ball equally often.
    """
    angles = spread_angles(count, seam_range, rng)
    return [make_photo(number, balls[(number - 1) % len(balls)], angles[int(position)], rng, click_error)
            for number, position in enumerate(rng.permutation(count), start=1)]


def _clicked(centre: tuple[float, float], rng: np.random.Generator,
             click_error: float) -> tuple[float, float]:
    """Return centre moved up to click_error px each way, like a ball centre clicked by hand."""
    return (centre[0] + float(rng.uniform(-click_error, click_error)),
            centre[1] + float(rng.uniform(-click_error, click_error)))


def save_samples(samples: list[SampleImage], folder: Path) -> list[BallImage]:
    """Write the images and their ball_images.yaml index, replacing any from an earlier run."""
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.glob("ball_*.png"):
        path.unlink()
    images = []
    for sample in samples:
        path = folder / sample.name
        write_image(path, sample.pixels)
        images.append(BallImage(frame_number=sample.number, path=path, centre=sample.given_centre,
                                diameter_px=BALL_DIAMETER, origin=(0, 0), size_px=IMAGE_SIZE,
                                travel_direction_deg=sample.travel_direction_deg))
    _write_index(folder, INDEX_SOURCE, 0, CROP_SCALE, images)
    return images


def write_details(samples: list[SampleImage], path: Path) -> None:
    """Write image_details.csv: how each sample image was made."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DETAIL_COLUMNS)
        for sample in samples:
            writer.writerow([sample.name, f"{sample.seam_angle_deg:.3f}", f"{sample.seam_line_deg:.3f}",
                             f"{sample.travel_direction_deg:.3f}", f"{sample.centre[0]:.2f}",
                             f"{sample.centre[1]:.2f}", f"{sample.given_centre[0]:.2f}",
                             f"{sample.given_centre[1]:.2f}", sample.photo, f"{sample.rotation_deg:.3f}"])


def load_image_details(path: Path) -> dict[str, ImageDetails]:
    """Read image_details.csv back, by image file name, or return nothing if it is missing."""
    if not path.exists():
        return {}
    details = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            # Files from before the drawn balls were taken out have a kind column, which is
            # "photo" or "drawn".
            details[row["image"]] = ImageDetails(row["image"], row.get("kind", "photo") == "photo",
                                                 float(row["seam_angle_deg"]), float(row["travel_direction_deg"]))
    return details


def main(argv: list[str] | None = None) -> None:
    """Make the sample images, their index and image_details.csv, replacing any from an
    earlier run.
    """
    parser = argparse.ArgumentParser(description="Make sample ball images from photographed balls for "
                                                 "comparing the automatic seam angle detector with seams "
                                                 "clicked by hand.")
    parser.add_argument("--photos", type=int, default=PHOTO_COUNT,
                        help=f"number of images to make (default {PHOTO_COUNT})")
    parser.add_argument("--range", nargs=2, type=float, default=SEAM_RANGE, metavar=("MIN", "MAX"),
                        help="seam angles to the direction of travel, in degrees (default -30 30)")
    parser.add_argument("--folders", nargs="+", type=Path, default=DEFAULT_FOLDERS,
                        help="ball image folders with a seam_labels.yaml, for the photographed balls "
                             "(default: output/frames/1t/ball_images)")
    parser.add_argument("--click-error", type=float, default=CLICK_ERROR_PX,
                        help=f"most the saved ball centre is off each way, in pixels (default {CLICK_ERROR_PX:g})")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    parser.add_argument("--force", action="store_true",
                        help="make new images even though seams were clicked on the old ones; "
                             "those clicks are kept under a new name")
    args = parser.parse_args(argv)

    if args.photos < 1:
        parser.error("make at least one image")
    seam_range = (float(args.range[0]), float(args.range[1]))
    if not -SEARCH_RANGE_DEG <= seam_range[0] < seam_range[1] <= SEARCH_RANGE_DEG:
        parser.error(f"--range needs {-SEARCH_RANGE_DEG:g} <= MIN < MAX <= {SEARCH_RANGE_DEG:g}, "
                     "the angles the automatic detector looks at")

    try:
        balls = [ball for folder in args.folders for ball in load_labelled_balls(folder)]
    except FileNotFoundError as error:
        print(f"{error}\nThe sample images are made from labelled ball images (with a seam_labels.yaml), "
              "which are not in the repository.")
        raise SystemExit(1)
    if not balls:
        print("There are no labelled ball images to make the sample images from.")
        raise SystemExit(1)

    if MANUAL_PATH.exists():
        if not args.force:
            print(f"Seams have already been clicked on the current images ({MANUAL_PATH.name}), and new "
                  "images would leave those clicks without their images. Run with --force to make new "
                  "images anyway; the old clicks are kept under a new name.")
            raise SystemExit(1)
        kept = MANUAL_PATH.with_name(f"{MANUAL_PATH.stem}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        MANUAL_PATH.rename(kept)
        print(f"Kept the earlier clicks as {kept.name}")

    rng = np.random.default_rng(args.seed)
    samples = make_samples(balls, args.photos, seam_range, rng, args.click_error)
    save_samples(samples, IMAGE_FOLDER)
    write_details(samples, DETAILS_PATH)

    print(f"Saved {len(samples)} sample images to {IMAGE_FOLDER}, made from {len(balls)} photographed balls, "
          f"seam angles {seam_range[0]:+g} to {seam_range[1]:+g} deg, seed {args.seed}.")
    print(f"The seam angle each was made with is in {DETAILS_PATH.name}.")
    print("Next, click the seams: python -m report_data.seam_detector_comparison.click_seams")


if __name__ == "__main__":
    main()
