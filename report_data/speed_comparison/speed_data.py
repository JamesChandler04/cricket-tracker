"""Reads and writes the speed comparison's data: the radar speeds, and each delivery's
per-frame top-down speeds, saved by the GUI under the ball and delivery numbers in the
video's file name.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

from library import paths

FOLDER = paths.REPORT_DATA_DIR / "speed_comparison"
"""Folder of the speed comparison's data, program and chart."""
SPEEDS_PATH = FOLDER / "actual_top_down_data.csv"
"""CSV of each delivery's per-frame top-down speeds, by ball and delivery number."""
RADAR_PATH = FOLDER / "radar_speeds.yaml"
"""YAML file of the radar speeds in km/h, under keys like ball_1 and delivery_3."""
COLUMNS = ["ball", "delivery", "frame_speeds_km_h"]
"""Column headings of actual_top_down_data.csv."""

BALL_PATTERN = re.compile(r"(?:^|[^a-z0-9])b(\d+)(?![a-z0-9])", re.IGNORECASE)
"""Ball number in a video's file name: a B and a number on their own."""
DELIVERY_PATTERN = re.compile(r"(?:^|[^a-z0-9])(\d+)$", re.IGNORECASE)
"""Delivery number in a video's file name: the number at its end."""


def ball_and_delivery(video_path: str | Path) -> tuple[int, int] | None:
    """Return the ball and delivery numbers in a video's file name, or None if it does not
    have both.
    """
    stem = Path(video_path).stem
    ball = BALL_PATTERN.search(stem)
    delivery = DELIVERY_PATTERN.search(stem)
    if ball is None or delivery is None:
        return None
    return int(ball.group(1)), int(delivery.group(1))


def load_calculated_speeds(path: Path = SPEEDS_PATH) -> dict[tuple[int, int], list[float]]:
    """Return the per-frame top-down speeds in km/h by (ball, delivery), or none if the file
    does not exist yet.
    """
    if not path.exists():
        return {}
    speeds = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            values = json.loads(row["frame_speeds_km_h"])
            speeds[(int(row["ball"]), int(row["delivery"]))] = [float(value) for value in values]
    return speeds


def save_calculated_speeds(speeds: dict[tuple[int, int], list[float]], path: Path = SPEEDS_PATH) -> None:
    """Write the per-frame top-down speeds to path, sorted by ball and then delivery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for (ball, delivery), values in sorted(speeds.items()):
            writer.writerow([ball, delivery, "[" + ", ".join(f"{value:.1f}" for value in values) + "]"])


def record_speeds(video_path: str | Path, frame_speeds: list[float], path: Path = SPEEDS_PATH,
                  radar_path: Path = RADAR_PATH) -> tuple[int, int] | None:
    """Save one delivery's per-frame speeds in km/h under the ball and delivery numbers in
    the video's file name, replacing any saved for that delivery before. Returns the
    numbers, or None if the file name has none or the file could not be written.
    """
    name = Path(video_path).name
    numbers = ball_and_delivery(video_path)
    if numbers is None:
        print(f"Speeds not saved for the radar comparison: {name} has no ball and delivery "
              "number in its name, like Jordan_B1_Birdseye_003.mp4.")
        return None
    ball, delivery = numbers
    try:
        speeds = load_calculated_speeds(path)
        replaced = numbers in speeds
        speeds[numbers] = list(frame_speeds)
        save_calculated_speeds(speeds, path)
    except (OSError, ValueError, KeyError) as error:
        print(f"Could not save the speeds for the radar comparison to {path}: {error}")
        return None
    print(f"Per-frame speeds for ball {ball}, delivery {delivery} saved to {path.name}"
          + (", replacing the earlier ones." if replaced else "."))

    try:
        radar = load_radar_speeds(radar_path).get(numbers)
    except (OSError, ValueError, yaml.YAMLError):
        radar = None
    mean = sum(frame_speeds) / len(frame_speeds) if frame_speeds else 0.0
    if radar is None:
        print(f"No radar speed for ball {ball}, delivery {delivery} in {radar_path.name}.")
    else:
        print(f"Radar speed {radar:g} km/h, top-down mean {mean:.1f} km/h.")
    return numbers


def load_radar_speeds(path: Path = RADAR_PATH) -> dict[tuple[int, int], float]:
    """Return the radar speeds in km/h by (ball, delivery), from keys like ball_1 and delivery_3."""
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    speeds = {}
    for ball_key, deliveries in data.items():
        ball = _key_number(ball_key, "ball")
        for delivery_key, speed in (deliveries or {}).items():
            if speed is not None:
                speeds[(ball, _key_number(delivery_key, "delivery"))] = float(speed)
    return speeds


def _key_number(key: object, prefix: str) -> int:
    """Return the number in a radar_speeds.yaml key like ball_1 or delivery_3."""
    match = re.fullmatch(rf"{prefix}_(\d+)", str(key).strip())
    if match is None:
        raise ValueError(f"{key!r} in radar_speeds.yaml should look like {prefix}_1")
    return int(match.group(1))
