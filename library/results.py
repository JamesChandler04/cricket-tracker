from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from library.detect_seam_angle_file import BALL_IMAGE_FOLDER, SEAM_FILE_NAME
from library.physics_engines.side_on_physics_engine import ANALYSIS_YAML_NAME, POINTS_CSV_NAME

# Reads back what a delivery left in its save folder, for the GUI's Results and
# Coordinates tabs. Everything comes from the saved files, so the tabs show
# exactly what was recorded, and an older delivery can be looked at again by
# pointing the save folder at it.

TOP_DOWN_FILE_NAME = "top_down_analysis.yaml"
SEAM_METHODS = ("automatic", "manual")
NOT_SAVED = "Not in this folder yet"

# tracked_points.csv column, heading in the table, decimal places shown
COORDINATE_COLUMNS = (
    ("frame", "Frame", 0),
    ("time_s", "Time (s)", 4),
    ("x_m", "X (m)", 4),
    ("y_m", "Y (m)", 4),
    ("z_m", "Z (m)", 4),
    ("dx_m", "dX (m)", 4),
    ("dy_m", "dY (m)", 4),
    ("dz_m", "dZ (m)", 4),
)


@dataclass
class ResultRow:
    """One line of the Results tab, with any lines that sit under it."""

    name: str
    value: str = ""
    children: list[ResultRow] = field(default_factory=list)


@dataclass
class ResultSection:
    """The results read from one saved file."""

    title: str
    path: Path
    rows: list[ResultRow]


@dataclass
class CoordinateTable:
    """The ball's position on each tracked frame, from tracked_points.csv."""

    path: Path
    headers: list[str]
    rows: list[list[str]]
    message: str = ""  # why there are no rows, when there are none


def load_results(folder: str | Path, top_down_file: str = TOP_DOWN_FILE_NAME) -> list[ResultSection]:
    """The results saved in folder, one section per file, in the order the steps run."""
    folder = Path(folder)
    sections = [_top_down_section(folder / top_down_file)]
    seam_files = [(method, folder / BALL_IMAGE_FOLDER / SEAM_FILE_NAME.format(method=method))
                  for method in SEAM_METHODS]
    found = [(method, path) for method, path in seam_files if path.exists()]
    if found:
        sections += [_seam_section(path, method) for method, path in found]
    else:
        sections.append(ResultSection("Seam angle", seam_files[0][1].parent, [ResultRow(NOT_SAVED)]))
    sections.append(_side_on_section(folder / ANALYSIS_YAML_NAME))
    return sections


def load_coordinates(folder: str | Path) -> CoordinateTable:
    """X, Y, Z and dX, dY, dZ on each tracked frame, as text for the Coordinates tab."""
    path = Path(folder) / POINTS_CSV_NAME
    headers = [heading for _, heading, _ in COORDINATE_COLUMNS]
    if not path.exists():
        return CoordinateTable(path, headers, [], f"No {POINTS_CSV_NAME} in this folder yet. "
                                                   "It is written by the side-on stage.")
    try:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column, _, _ in COORDINATE_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                return CoordinateTable(path, headers, [], f"{path.name} has no {', '.join(missing)} column.")
            rows = [[_cell(record[column], places) for column, _, places in COORDINATE_COLUMNS]
                    for record in reader]
    except (OSError, csv.Error) as error:
        return CoordinateTable(path, headers, [], f"Could not read {path.name}: {error}")
    return CoordinateTable(path, headers, rows)


def _top_down_section(path: Path) -> ResultSection:
    data, problem = _read_yaml(path)
    if problem:
        return ResultSection("Top-down", path, [ResultRow(problem)])
    return ResultSection("Top-down", path, [
        ResultRow("Velocity", _number(data.get("velocity_km_h"), "{:.2f} km/h")),
        ResultRow("Seam angle", _number(data.get("seam_angle_deg"), "{:+.2f} deg", "not set")),
        ResultRow("Frame rate", _number(data.get("fps"), "{:.2f} fps")),
        ResultRow("Tracked points", _number(data.get("point_count"), "{:.0f}")),
    ])


def _seam_section(path: Path, method: str) -> ResultSection:
    title = f"Seam angle ({method})"
    data, problem = _read_yaml(path)
    if problem:
        return ResultSection(title, path, [ResultRow(problem)])
    rows = [
        ResultRow("Seam angle", _number(data.get("seam_angle_deg"), "{:+.2f} deg", "not set")),
        ResultRow("Seam line in the image", _number(data.get("raw_angle_deg"), "{:+.2f} deg")),
        ResultRow("Direction of travel", _number(data.get("travel_direction_deg"), "{:+.2f} deg")),
    ]
    per_frame = data.get("frame_seam_angles_deg")
    if isinstance(per_frame, dict):
        rows.append(ResultRow("Frames averaged", _number(data.get("frames_averaged"), "{:.0f}")))
        rows.append(ResultRow("Spread (largest - smallest)", _number(data.get("seam_angle_spread_deg"), "{:.2f} deg")))
        rows.append(ResultRow("Seam angle on each frame", "", [
            ResultRow(f"Frame {frame}", _number(angle, "{:+.2f} deg")) for frame, angle in sorted(per_frame.items())
        ]))
    else:
        rows.append(ResultRow("Picked on frame", _number(data.get("frame"), "{:.0f}")))
    if data.get("warning"):
        rows.append(ResultRow("Warning", str(data["warning"])))
    return ResultSection(title, path, rows)


def _side_on_section(path: Path) -> ResultSection:
    data, problem = _read_yaml(path)
    if problem:
        return ResultSection("Side-on", path, [ResultRow(problem)])
    delivery = _mapping(data.get("delivery"))
    swing = _mapping(data.get("swing"))
    baseline = _mapping(data.get("baseline_fit"))
    projection = _mapping(data.get("projection_fit"))
    calibration = _mapping(data.get("calibration"))
    target = swing.get("target_distance_m")
    rows = [
        ResultRow("Delivery speed used", _speed(delivery.get("speed_km_h"), delivery.get("speed_m_s"))),
        ResultRow("Swing at last tracked point",
                  _join(_number(swing.get("at_last_tracked_point_cm"), "{:+.2f} cm"),
                        _number(swing.get("last_point_distance_m"), "at {:.2f} m"))),
        ResultRow(f"Swing at {_number(target, '{:.0f} m', 'target')} (projected)",
                  _number(swing.get("at_target_distance_cm"), "{:+.2f} cm")),
        ResultRow("Last point off the projection",
                  _number(swing.get("last_point_offset_from_projection_cm"), "{:+.2f} cm")),
        ResultRow("Baseline fit", _fit(baseline, "first")),
        ResultRow("Projection fit", _fit(projection, "last")),
        ResultRow("Tracked points",
                  _join(_number(delivery.get("point_count"), "{:.0f}"),
                        _frames(delivery.get("first_frame"), delivery.get("last_frame")))),
        ResultRow("Duration", _number(delivery.get("duration_s"), "{:.3f} s")),
        ResultRow("Frame rate", _number(delivery.get("fps"), "{:.2f} fps")),
        ResultRow("Drag coefficient", _number(delivery.get("drag_coefficient"), "{:g}")),
        ResultRow("Calibration", "", [
            ResultRow("File", str(calibration.get("path", "-"))),
            ResultRow("Focal length", _vector(calibration.get("focal_px"), "{:.1f}", "px")),
            ResultRow("Principal point", _vector(calibration.get("principal_point_px"), "{:.1f}", "px")),
            ResultRow("Camera centre (X, Y, Z)", _vector(calibration.get("camera_centre_m"), "{:.3f}", "m")),
        ]),
        ResultRow("Saved", str(data.get("written_utc", "-")).replace("T", " ").replace("Z", " UTC")),
    ]
    return ResultSection("Side-on", path, rows)


def _read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    """The file's contents, or an empty mapping and the reason it could not be read."""
    if not path.exists():
        return {}, NOT_SAVED
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        return {}, f"Could not read {path.name}: {error}"
    if not isinstance(data, dict):
        return {}, f"{path.name} is not laid out as expected"
    return data, ""


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any, layout: str, missing: str = "-") -> str:
    if value is None or isinstance(value, bool):
        return missing
    try:
        return layout.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _join(*parts: str) -> str:
    return " ".join(part for part in parts if part and part != "-") or "-"


def _speed(km_h: object, m_s: object) -> str:
    speed = _number(km_h, "{:.2f} km/h")
    in_m_s = _number(m_s, "({:.2f} m/s)")
    return speed if in_m_s == "-" else f"{speed} {in_m_s}"


def _fit(fit: dict[str, Any], which: str) -> str:
    points = _number(fit.get("source_point_count"), f"{which} {{:.0f}} points")
    residual = _number(fit.get("residual_rms_cm"), "residual {:.2f} cm RMS")
    return ", ".join(part for part in (points, residual) if part != "-") or "-"


def _frames(first: object, last: object) -> str:
    if first is None or last is None:
        return ""
    return f"(frames {first}-{last})"


def _vector(values: object, layout: str, unit: str) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return "-"
    return ", ".join(_number(value, layout) for value in values) + f" {unit}"


def _cell(text: str | None, places: int) -> str:
    if text is None:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:.0f}" if places == 0 else f"{number:.{places}f}"
