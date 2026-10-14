"""Reads and writes the swing and seam comparison's data: the recorded deliveries, each with a
folder holding its seam angles, measured on the top-down ball images, and the ball centres
clicked in its side-on video, and the table of swing values calculate_swing makes from them
at a set speed, at the last tracked point and at a common distance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from library import paths
from library.physics_engines.side_on_physics_engine import TrackedPoint

FOLDER = paths.REPORT_DATA_DIR / "swing_seam_comparison"
"""Folder of the swing and seam comparison's data, programs and charts."""
RECORDED_FOLDER = FOLDER / "recorded_deliveries"
"""Folder with one subfolder per recorded delivery."""
DELIVERY_FOLDER_NAME = "ball_{ball}_delivery_{delivery:03d}"
"""Name of a recorded delivery's folder, from its ball and delivery numbers."""
DELIVERY_FILE_NAME = "delivery.yaml"
"""File in a recorded delivery's folder with its seam angles and clicked side-on points."""
SEAM_METHODS = ("manual", "automatic")
"""Ways a seam angle can be measured, named as the seam detectors name them."""
POINT_COLUMNS = ["frame", "u_px", "v_px"]
"""What each clicked side-on point in delivery.yaml lists, in order."""
TABLE_PATH = FOLDER / "swing_seam_comparison.csv"
"""CSV of the swing calculated for each recorded delivery, with its seam angles."""
TABLE_COLUMNS = ["ball", "delivery", "manual_seam_angle_deg", "automatic_seam_angle_deg", "automatic_seam_spread_deg",
                 "speed_km_h", "speed_source", "speed_error_percent", "side_on_points", "first_frame", "last_frame",
                 "last_point_distance_m", "swing_at_last_point_cm", "swing_at_last_point_slower_cm",
                 "swing_at_last_point_faster_cm", "baseline_residual_cm", "projection_residual_cm",
                 "implied_side_force_coefficient", "common_distance_m", "swing_at_common_distance_cm",
                 "swing_at_common_distance_slower_cm", "swing_at_common_distance_faster_cm", "extrapolated_m",
                 "points_past_common_distance"]
"""Column headings of swing_seam_comparison.csv. Files from before the swing at a common
distance was added stop at implied_side_force_coefficient.
"""


@dataclass
class SeamAngle:
    """One measurement of a delivery's seam angle."""

    method: str             # "manual" or "automatic"
    angle_deg: float        # from the direction of travel, folded into (-90, 90]
    frame: int              # top-down frame of the ball image the seam line is on
    frame_angles_deg: dict[int, float] = field(default_factory=dict)  # automatic: the angle on each ball image
    spread_deg: float | None = None  # automatic: largest minus smallest of frame_angles_deg
    warning: str | None = None       # automatic: set when the spread is over the detector's limit


@dataclass
class RecordedDelivery:
    """One delivery recorded for the comparison: its seam angles and the ball centres
    clicked in its side-on video.
    """

    ball: int
    delivery: int
    recorded: str             # local date and time it was recorded
    top_down_video: str
    ball_frames: list[int]    # top-down frames the seam angle was measured on
    ball_found_by: str        # how the ball was found in those frames
    seams: dict[str, SeamAngle]  # by method
    side_on_video: str
    side_on_fps: float
    side_on_points: list[TrackedPoint]

    @property
    def label(self) -> str:
        """Return the delivery's name for messages, like "ball 1 delivery 3"."""
        return f"ball {self.ball} delivery {self.delivery}"

    def seam_angle(self, method: str) -> float | None:
        """Return the seam angle measured by method, in degrees, or None if it was not measured that way."""
        seam = self.seams.get(method)
        return None if seam is None else seam.angle_deg


@dataclass
class SwingValues:
    """One delivery's seam angles and the swing calculated for it: a row of swing_seam_comparison.csv."""

    ball: int
    delivery: int
    manual_seam_deg: float | None
    automatic_seam_deg: float | None
    automatic_spread_deg: float | None
    speed_km_h: float
    speed_source: str              # "set", "radar", "top-down" or "top-down (no radar)"
    speed_error_percent: float     # the slower and faster speeds are this far below and above speed_km_h
    side_on_points: int
    first_frame: int
    last_frame: int
    last_point_distance_m: float   # forward distance from the first tracked point to the last
    swing_cm: float                # at the last tracked point
    swing_slower_cm: float         # at the last tracked point, at the slower speed
    swing_faster_cm: float         # at the last tracked point, at the faster speed
    baseline_residual_cm: float
    projection_residual_cm: float
    side_force_coefficient: float | None
    # The swing at the common distance, from the first tracked point: tracks that go further
    # are cut there, and shorter ones carried on to it. None where there were too few points
    # before it, and in files from before it was added.
    common_distance_m: float | None = None
    common_swing_cm: float | None = None
    common_swing_slower_cm: float | None = None
    common_swing_faster_cm: float | None = None
    extrapolated_m: float | None = None              # how far the swing was carried on past the last point used
    points_past_common_distance: int | None = None   # tracked points past the common distance, left out

    def seam_angle(self, method: str) -> float | None:
        """Return the seam angle measured by method, in degrees, or None if it was not measured that way."""
        return self.manual_seam_deg if method == "manual" else self.automatic_seam_deg

    @property
    def swing_range_cm(self) -> tuple[float, float]:
        """Return the lowest and highest swing over the speeds from the slower to the faster one, in cm."""
        swings = (self.swing_cm, self.swing_slower_cm, self.swing_faster_cm)
        return min(swings), max(swings)

    @property
    def common_swing_range_cm(self) -> tuple[float, float] | None:
        """Return the lowest and highest swing at the common distance over the speeds from the
        slower to the faster one, in cm, or None if there is no swing there.
        """
        swings = [swing for swing in (self.common_swing_cm, self.common_swing_slower_cm, self.common_swing_faster_cm)
                  if swing is not None]
        return (min(swings), max(swings)) if swings else None


def delivery_folder(ball: int, delivery: int, root: Path = RECORDED_FOLDER) -> Path:
    """Return the folder a delivery is recorded in."""
    return root / DELIVERY_FOLDER_NAME.format(ball=ball, delivery=delivery)


def save_delivery(recorded: RecordedDelivery, root: Path = RECORDED_FOLDER) -> Path:
    """Write a recorded delivery to delivery.yaml in its folder, replacing any recorded
    before. Returns the file's path.
    """
    folder = delivery_folder(recorded.ball, recorded.delivery, root)
    folder.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "ball": recorded.ball,
        "delivery": recorded.delivery,
        "recorded": recorded.recorded,
        "top_down": {
            "video": recorded.top_down_video,
            "ball_found_by": recorded.ball_found_by,
            "ball_frames": list(recorded.ball_frames),
        },
        "seam_angles": {method: _seam_mapping(seam) for method, seam in recorded.seams.items()},
        "side_on": {
            "video": recorded.side_on_video,
            "fps": recorded.side_on_fps,
            "point_columns": POINT_COLUMNS,
            "points": [[point.frame, point.u, point.v] for point in recorded.side_on_points],
        },
    }
    path = folder / DELIVERY_FILE_NAME
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=None, allow_unicode=True, width=4096)
    return path


def load_delivery(path: Path) -> RecordedDelivery:
    """Read back a delivery.yaml written by save_delivery."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    top_down, side_on = data["top_down"], data["side_on"]
    seams = {str(method): _seam_angle(str(method), values)
             for method, values in (data.get("seam_angles") or {}).items()}
    return RecordedDelivery(
        ball=int(data["ball"]),
        delivery=int(data["delivery"]),
        recorded=str(data.get("recorded", "")),
        top_down_video=str(top_down["video"]),
        ball_frames=[int(frame) for frame in top_down.get("ball_frames") or []],
        ball_found_by=str(top_down.get("ball_found_by", "")),
        seams=seams,
        side_on_video=str(side_on["video"]),
        side_on_fps=float(side_on["fps"]),
        side_on_points=[TrackedPoint(int(frame), float(u), float(v)) for frame, u, v in side_on["points"] or []],
    )


def load_deliveries(root: Path = RECORDED_FOLDER) -> list[RecordedDelivery]:
    """Return every delivery recorded in root, sorted by ball and then delivery. A file
    that cannot be read is reported and left out.
    """
    deliveries = []
    for path in sorted(root.glob(f"*/{DELIVERY_FILE_NAME}")):
        try:
            deliveries.append(load_delivery(path))
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
            print(f"Could not read {path}: {error}")
    return sorted(deliveries, key=lambda recorded: (recorded.ball, recorded.delivery))


def save_swing_values(values: list[SwingValues], path: Path = TABLE_PATH) -> None:
    """Write the swing values to path, one row per delivery."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TABLE_COLUMNS)
        for row in values:
            writer.writerow([row.ball, row.delivery, _format(row.manual_seam_deg, 2),
                             _format(row.automatic_seam_deg, 2), _format(row.automatic_spread_deg, 2),
                             f"{row.speed_km_h:.1f}", row.speed_source, f"{row.speed_error_percent:g}",
                             row.side_on_points, row.first_frame, row.last_frame, f"{row.last_point_distance_m:.2f}",
                             f"{row.swing_cm:.2f}", f"{row.swing_slower_cm:.2f}", f"{row.swing_faster_cm:.2f}",
                             f"{row.baseline_residual_cm:.2f}", f"{row.projection_residual_cm:.2f}",
                             _format(row.side_force_coefficient, 3), _format(row.common_distance_m, 2),
                             _format(row.common_swing_cm, 2), _format(row.common_swing_slower_cm, 2),
                             _format(row.common_swing_faster_cm, 2), _format(row.extrapolated_m, 2),
                             "" if row.points_past_common_distance is None else row.points_past_common_distance])


def load_swing_values(path: Path = TABLE_PATH) -> list[SwingValues]:
    """Read back the swing values written by save_swing_values. Files from before the swing
    at a common distance was added are read with none.
    """
    with open(path, newline="") as handle:
        return [SwingValues(
            ball=int(row["ball"]),
            delivery=int(row["delivery"]),
            manual_seam_deg=_number(row["manual_seam_angle_deg"]) + 10.0, # Adjusting for known top-down camera rotation
            automatic_seam_deg=_number(row["automatic_seam_angle_deg"]),
            automatic_spread_deg=_number(row["automatic_seam_spread_deg"]),
            speed_km_h=float(row["speed_km_h"]),
            speed_source=row["speed_source"],
            speed_error_percent=float(row["speed_error_percent"]),
            side_on_points=int(row["side_on_points"]),
            first_frame=int(row["first_frame"]),
            last_frame=int(row["last_frame"]),
            last_point_distance_m=float(row["last_point_distance_m"]),
            swing_cm=float(row["swing_at_last_point_cm"]),
            swing_slower_cm=float(row["swing_at_last_point_slower_cm"]),
            swing_faster_cm=float(row["swing_at_last_point_faster_cm"]),
            baseline_residual_cm=float(row["baseline_residual_cm"]),
            projection_residual_cm=float(row["projection_residual_cm"]),
            side_force_coefficient=_number(row["implied_side_force_coefficient"]),
            common_distance_m=_number(row.get("common_distance_m") or ""),
            common_swing_cm=_number(row.get("swing_at_common_distance_cm") or ""),
            common_swing_slower_cm=_number(row.get("swing_at_common_distance_slower_cm") or ""),
            common_swing_faster_cm=_number(row.get("swing_at_common_distance_faster_cm") or ""),
            extrapolated_m=_number(row.get("extrapolated_m") or ""),
            points_past_common_distance=_whole_number(row.get("points_past_common_distance") or ""),
        ) for row in csv.DictReader(handle)]


def _seam_mapping(seam: SeamAngle) -> dict[str, Any]:
    """Return a seam angle as the mapping saved in delivery.yaml, leaving out what its method does not give."""
    mapping: dict[str, Any] = {"angle_deg": round(seam.angle_deg, 3), "frame": seam.frame}
    if seam.frame_angles_deg:
        mapping["frame_angles_deg"] = {frame: round(angle, 3) for frame, angle in seam.frame_angles_deg.items()}
    if seam.spread_deg is not None:
        mapping["spread_deg"] = round(seam.spread_deg, 3)
    if seam.warning is not None:
        mapping["warning"] = seam.warning
    return mapping


def _seam_angle(method: str, values: dict[str, Any]) -> SeamAngle:
    """Return the seam angle saved in delivery.yaml under method."""
    spread = values.get("spread_deg")
    return SeamAngle(
        method=method,
        angle_deg=float(values["angle_deg"]),
        frame=int(values["frame"]),
        frame_angles_deg={int(frame): float(angle) for frame, angle in (values.get("frame_angles_deg") or {}).items()},
        spread_deg=None if spread is None else float(spread),
        warning=values.get("warning"),
    )


def _format(value: float | None, decimals: int) -> str:
    """Format a number to decimals places, or leave it blank when there is none."""
    return "" if value is None else f"{value:.{decimals}f}"


def _number(text: str) -> float | None:
    """Return the number in a CSV cell, or None if the cell is blank."""
    return float(text) if text.strip() else None


def _whole_number(text: str) -> int | None:
    """Return the whole number in a CSV cell, or None if the cell is blank."""
    return int(text) if text.strip() else None
