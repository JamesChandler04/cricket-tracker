"""Makes training images for the seam angle network from labelled ball images.

Each is a ball turned, moved and re-shaded on a 200 x 200 px image, black outside the
ball, and labels.csv records the angle of its seam line and how it was made.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from library import paths
from library.detect_seam_angle_file import CROP_SCALE, BallImage, fold_line_angle, load_ball_images

IMAGE_SIZE = 200
"""Width and height of each training image, in px."""
BALL_DIAMETER = IMAGE_SIZE / CROP_SCALE
"""Ball diameter in each training image, in px, matching the ball images' framing."""
MAX_SHIFT = 10.0
"""Default largest shift of the ball centre from the image's middle, each way, in px."""

LABEL_FILE_NAME = "seam_labels.yaml"
"""File in a ball image folder giving each image's seam line angle, in degrees."""
DEFAULT_FOLDERS = [paths.OUTPUT_DIR / "frames" / "1t" / "ball_images"]
"""Ball image folders the labelled balls are read from by default."""
DEFAULT_OUTPUT_DIR = paths.OUTPUT_DIR / "seam_training"
"""Folder the training images are written to by default."""
IMAGE_PREFIX = "seam_"
"""Start of each training image's file name."""
LABELS_FILE_NAME = "labels.csv"
"""CSV file listing each training image's seam line angle and how it was made."""

# Shade changes, picked at random for each image. Hue shifts are in degrees
# either way; the others multiply the saturation and brightness.
LEATHER_HUE_SHIFT = 6.0
"""Largest random shift of the leather's hue, either way, in degrees."""
LEATHER_SATURATION = (0.85, 1.15)
"""Range of the random factor the leather's saturation is multiplied by."""
LEATHER_BRIGHTNESS = (0.85, 1.15)
"""Range of the random factor the leather's brightness is multiplied by."""
SEAM_HUE_SHIFT = 10.0
"""Largest random shift of the stitching's hue, either way, in degrees."""
SEAM_SATURATION = (0.7, 1.3)
"""Range of the random factor the stitching's saturation is multiplied by."""
SEAM_BRIGHTNESS = (0.85, 1.15)
"""Range of the random factor the stitching's brightness is multiplied by."""

_ROWS, _COLUMNS = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]


@dataclass
class LabelledBall:
    """A ball image and the angle of its seam line in the image."""

    image: BallImage
    seam_angle_deg: float
    pixels: np.ndarray


@dataclass
class Example:
    """One training image and how it was made."""

    image: np.ndarray
    seam_angle_deg: float
    rotation_deg: float           # clockwise
    shift_px: tuple[float, float]  # ball centre from the middle of the image


def load_labelled_balls(folder: Path) -> list[LabelledBall]:
    """The ball images in folder that have a seam angle in its seam_labels.yaml."""
    label_path = folder / LABEL_FILE_NAME
    if not label_path.exists():
        raise FileNotFoundError(f"{label_path} not found. It should give each ball image's seam angle, "
                                f"one per line, like  frame_000184.png: 0.7")
    with open(label_path) as handle:
        labels = yaml.safe_load(handle) or {}
    images = {image.path.name: image for image in load_ball_images(folder)}

    balls = []
    for name, angle in labels.items():
        if name not in images:
            raise ValueError(f"{label_path} has a seam angle for {name}, "
                             f"but {name} is not in that folder's ball_images.yaml")
        image = images[name]
        balls.append(LabelledBall(image, float(angle), image.load()))
    unlabelled = sorted(set(images) - set(labels))
    if unlabelled:
        print(f"No seam angle for {', '.join(unlabelled)} in {label_path}, so they are not used.")
    return balls


def pick_rotation(ball: LabelledBall, rng: np.random.Generator, seam_range: tuple[float, float]) -> float:
    """A clockwise rotation that puts the ball's seam at a random angle within seam_range."""
    target = rng.uniform(*seam_range)
    upside_down = 180.0 if rng.random() < 0.5 else 0.0  # same seam angle, ball turned over
    return float((target - ball.seam_angle_deg + upside_down) % 360.0)


def make_example(ball: LabelledBall, rng: np.random.Generator, max_shift: float = MAX_SHIFT,
                 rotation_deg: float | None = None) -> Example:
    """
    One training image: the ball turned, moved and re-shaded, black outside
    the ball. It is turned by rotation_deg clockwise, or a random amount if
    rotation_deg is None.
    """
    rotation = float(rng.uniform(0.0, 360.0)) if rotation_deg is None else rotation_deg
    shift = (float(rng.uniform(-max_shift, max_shift)), float(rng.uniform(-max_shift, max_shift)))
    middle = (IMAGE_SIZE - 1) / 2
    centre = (middle + shift[0], middle + shift[1])

    # Scale the ball to BALL_DIAMETER and turn it about its centre, then move
    # its centre into place. getRotationMatrix2D turns anticlockwise for
    # positive angles, so the clockwise rotation goes in negative.
    source = ball.image.to_crop(*ball.image.centre)
    matrix = cv2.getRotationMatrix2D(source, -rotation, BALL_DIAMETER / ball.image.diameter_px)
    matrix[0, 2] += centre[0] - source[0]
    matrix[1, 2] += centre[1] - source[1]
    turned = cv2.warpAffine(ball.pixels, matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    image = shade(turned, rng)
    outside = (_COLUMNS - centre[0]) ** 2 + (_ROWS - centre[1]) ** 2 > (BALL_DIAMETER / 2) ** 2
    image[outside] = 0

    # Turning the image clockwise turns the seam line clockwise by the same amount.
    return Example(image, fold_line_angle(ball.seam_angle_deg + rotation), rotation, shift)


def seam_weight(bgr: np.ndarray) -> np.ndarray:
    """
    How much each pixel looks like stitching rather than leather, from 0 to 1.
    Stitching is bright and nearly white, so its smallest colour channel is
    close to its largest. Leather is darker and strongly red.
    """
    brightest = bgr.max(axis=2)
    whiteness = bgr.min(axis=2) / np.maximum(brightest, 1e-3)
    return _smoothstep(0.35, 0.6, whiteness) * _smoothstep(0.28, 0.42, brightest)


def shade(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Give the leather and the stitching slightly different shades, keeping them red and white."""
    bgr = image.astype(np.float32) / 255
    weight = seam_weight(bgr)[..., None]
    leather = _change_colour(bgr, rng.uniform(-LEATHER_HUE_SHIFT, LEATHER_HUE_SHIFT),
                             rng.uniform(*LEATHER_SATURATION), rng.uniform(*LEATHER_BRIGHTNESS))
    seam = _change_colour(bgr, rng.uniform(-SEAM_HUE_SHIFT, SEAM_HUE_SHIFT),
                          rng.uniform(*SEAM_SATURATION), rng.uniform(*SEAM_BRIGHTNESS))
    mixed = leather * (1 - weight) + seam * weight
    return np.clip(mixed * 255 + 0.5, 0, 255).astype(np.uint8)


def _change_colour(bgr: np.ndarray, hue_shift: float, saturation: float, brightness: float) -> np.ndarray:
    """Return the image with hue shifted (deg) and saturation and brightness scaled."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # from floats: hue 0-360, saturation and value 0-1
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    """Return each value mapped smoothly from 0 at low to 1 at high, clamped outside."""
    t = np.clip((values - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def clear_earlier_run(folder: Path) -> None:
    """Delete the images and labels an earlier run left in folder, and nothing else."""
    for path in folder.glob(f"{IMAGE_PREFIX}*.png"):
        path.unlink()
    (folder / LABELS_FILE_NAME).unlink(missing_ok=True)


def write_image(path: Path, image: np.ndarray) -> None:
    """Save an image to path as a PNG."""
    # imencode and write_bytes instead of cv2.imwrite, which fails on Windows
    # when the folder name has characters outside the system code page.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def ask_count() -> int:
    """Ask the user how many images to make until they give a whole number above 0."""
    while True:
        answer = input("How many images should be made? ").strip()
        try:
            count = int(answer)
        except ValueError:
            print(f"'{answer}' is not a whole number.")
            continue
        if count > 0:
            return count
        print("Make at least one image.")


def _project_path(path: Path) -> str:
    """Return path relative to the project folder, or as given if it is outside it."""
    try:
        return path.resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> None:
    """Make the training images and labels.csv, replacing any from an earlier run."""
    parser = argparse.ArgumentParser(description="Make training images for the seam angle network.")
    parser.add_argument("count", nargs="?", type=int, help="how many images to make")
    parser.add_argument("--folders", nargs="+", type=Path, default=DEFAULT_FOLDERS,
                        help="ball image folders with a seam_labels.yaml (default: output/frames/1t/ball_images)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="folder to write the images to (default: output/seam_training)")
    parser.add_argument("--max-shift", type=float, default=MAX_SHIFT,
                        help=f"most the ball moves from the middle each way, in pixels (default {MAX_SHIFT:g})")
    parser.add_argument("--seam-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="only make seam angles between MIN and MAX degrees (default: any angle)")
    parser.add_argument("--seed", type=int, help="random seed, to make the same images again")
    args = parser.parse_args(argv)

    count = ask_count() if args.count is None else args.count
    if count < 1:
        parser.error("make at least one image")
    shift_limit = (IMAGE_SIZE - BALL_DIAMETER) / 2 - 1
    if not 0 <= args.max_shift <= shift_limit:
        parser.error(f"--max-shift has to be between 0 and {shift_limit:.0f} so the ball stays in the image")
    seam_range = None if args.seam_range is None else (args.seam_range[0], args.seam_range[1])
    if seam_range is not None and not -90 <= seam_range[0] < seam_range[1] <= 90:
        parser.error("--seam-range needs -90 <= MIN < MAX <= 90")

    balls = [ball for folder in args.folders for ball in load_labelled_balls(folder)]
    if not balls:
        print("No labelled ball images, so no training images were made.")
        return

    seed = int(np.random.default_rng().integers(2**31)) if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    clear_earlier_run(output)

    print(f"\nMaking {count} images from {len(balls)} labelled ball images (seed {seed})")
    digits = max(5, len(str(count - 1)))
    with open(output / LABELS_FILE_NAME, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "seam_angle_deg", "source_image", "source_seam_angle_deg",
                         "rotation_deg", "shift_x_px", "shift_y_px"])
        for index in range(count):
            ball = balls[index % len(balls)]  # every base image used equally often
            rotation = None if seam_range is None else pick_rotation(ball, rng, seam_range)
            example = make_example(ball, rng, args.max_shift, rotation)
            name = f"{IMAGE_PREFIX}{index:0{digits}d}.png"
            write_image(output / name, example.image)
            writer.writerow([name, f"{example.seam_angle_deg:.3f}", _project_path(ball.image.path),
                             f"{ball.seam_angle_deg:g}", f"{example.rotation_deg:.3f}",
                             f"{example.shift_px[0]:.2f}", f"{example.shift_px[1]:.2f}"])
            if (index + 1) % 500 == 0 and index + 1 < count:
                print(f"{index + 1} of {count} made")

    print(f"Saved {count} images and {LABELS_FILE_NAME} to {output}")


if __name__ == "__main__":
    main()
