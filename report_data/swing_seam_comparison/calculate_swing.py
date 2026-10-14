"""Calculates the swing of every delivery recorded for the swing and seam comparison at a set
speed, and again at that speed minus and plus a speed error, and saves the swing values with
the seam angles to swing_seam_comparison.csv for plot_swing_seam. Each delivery's
reconstruction at the set speed is saved in its folder as well.

Every swing is given twice: at the delivery's last tracked point, and at a common distance
from the first tracked point (8 m by default), so that every delivery is compared at the
same distance. A track that goes past the common distance is cut there. A shorter one is
carried on to it as a constant side force would carry it.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

if __package__ in (None, ""):
    # Started as a plain script (python report_data/swing_seam_comparison/<name>.py, or
    # the Run button in VS Code), so put the project folder on the path for the imports.
    # python -m report_data.swing_seam_comparison.<name> does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library import paths
from library.physics_engines.side_on_physics_engine import (AIR_DENSITY_KG_M3, BALL_DIAMETER_M, BALL_MASS_KG,
                                                            SideOnPhysicsEngine, SwingResult, TrackedPoint)
from report_data.speed_comparison.speed_data import RADAR_PATH, SPEEDS_PATH, load_calculated_speeds, load_radar_speeds
from report_data.swing_seam_comparison.swing_seam_data import (FOLDER, RECORDED_FOLDER, TABLE_PATH, RecordedDelivery,
                                                               SwingValues, delivery_folder, load_deliveries,
                                                               save_swing_values)

FloatArray = npt.NDArray[np.float64]
"""Type of the float64 numpy arrays used in this module's type annotations."""

SWING_PLOT_NAME = "swing.png"
"""File name of the swing plot saved in a delivery's folder with --plots."""
SPEED_ERROR_PERCENT = 5.0
"""Default speed error, in percent. The swing is also calculated at speeds this far below
and above the speed used, for the error bars.
"""
SPEED_SOURCES = ("radar", "top-down")
"""Words --speed takes instead of a speed in km/h."""
MIN_POINTS = 3
"""Fewest clicked side-on points the swing can be calculated from."""
COMMON_DISTANCE_M = 8.0
"""Default common distance every delivery's swing is also given at, in metres from the
first tracked point. Tracks that go further are cut there, and shorter ones are carried on
to it.
"""
MIN_UNIT_GAP_M2 = 1e-6
"""Smallest swing per unit of curvature (see unit_gap) the two line fits can give, in m².
Below it they are taken to use the same points, so no swing can be scaled from them.
"""


@dataclass
class CommonDistanceSwing:
    """A delivery's swing at the common distance, and how its track was made to reach it."""

    swing_cm: float
    last_point_distance_m: float  # forward distance of the last tracked point used
    extrapolated_m: float         # how far the swing was carried on past that point
    points_left_out: int          # tracked points past the common distance, left out


def speed_choice(text: str) -> str | float:
    """Read the --speed option: radar, top-down, or one speed in km/h for every delivery."""
    lowered = text.strip().lower()
    if lowered in SPEED_SOURCES:
        return lowered
    try:
        speed = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected radar, top-down or a speed in km/h, not {text!r}") from None
    if not speed > 0:
        raise argparse.ArgumentTypeError("the speed has to be above 0 km/h")
    return speed


def delivery_speed(recorded: RecordedDelivery, choice: str | float, radar: dict[tuple[int, int], float],
                   top_down: dict[tuple[int, int], list[float]]) -> tuple[float, str] | None:
    """Return the speed to reconstruct a delivery at, in km/h, and where it came from: the
    set speed, the delivery's radar speed, or its mean top-down speed. A delivery without a
    radar speed falls back to its top-down speed. Returns None if it has neither.
    """
    if isinstance(choice, float):
        return choice, "set"
    numbers = (recorded.ball, recorded.delivery)
    if choice == "radar" and numbers in radar:
        return radar[numbers], "radar"
    speeds = top_down.get(numbers)
    if speeds:
        return sum(speeds) / len(speeds), "top-down" if choice == "top-down" else "top-down (no radar)"
    return None


def analyse_delivery(recorded: RecordedDelivery, speed_km_h: float,
                     calibration: Path) -> tuple[SideOnPhysicsEngine, SwingResult]:
    """Reconstruct a delivery from its clicked side-on points at speed_km_h with the side-on
    physics engine, and return the engine and its swing analysis.
    """
    if len(recorded.side_on_points) < MIN_POINTS:
        raise ValueError(f"only {len(recorded.side_on_points)} side-on points were clicked, "
                         f"and at least {MIN_POINTS} are needed")
    engine = SideOnPhysicsEngine.from_calibration_file(calibration, fps=recorded.side_on_fps)
    return engine, engine.analyse(recorded.side_on_points, speed_km_h)


def unit_gap(baseline_y: FloatArray, projection_y: FloatArray, at_m: float) -> float:
    """Return the gap between the baseline and projection line fits at forward distance at_m
    on the curve x = y², with the baseline fitted at the forward distances baseline_y and the
    projection at projection_y, in m².

    A constant side force bends the path into x = q·y², so this is the swing per unit of q:
    the swing those fits give on that path is q times it.
    """
    baseline = np.polyfit(baseline_y, baseline_y ** 2, 1)
    projection = np.polyfit(projection_y, projection_y ** 2, 1)
    return float(np.polyval(projection, at_m) - np.polyval(baseline, at_m))


def implied_side_force_coefficient(result: SwingResult) -> float | None:
    """Return the side-force coefficient that would give the swing at the last tracked point
    if the side force were constant from the first point, or None if the baseline and
    projection fits use the same points.

    A constant side force bends the path into x = q·y², with q = C_S·p·A / (4m). The same
    two line fits on x = y^2 give the gap per unit of q, so q is the swing divided by it.
    """
    y = result.trajectory.y
    gap = unit_gap(y[:result.baseline_points], y[-result.extrapolation_points:], result.last_point_distance_m)
    if gap < MIN_UNIT_GAP_M2:
        return None
    q = result.swing_at_last_point_cm / 100 / gap
    area = math.pi * (BALL_DIAMETER_M / 2) ** 2
    return q * 4 * BALL_MASS_KG / (AIR_DENSITY_KG_M3 * area)


def swing_at_common_distance(engine: SideOnPhysicsEngine, points: Sequence[TrackedPoint], speed_km_h: float,
                             distance_m: float) -> CommonDistanceSwing | None:
    """Return a delivery's swing at distance_m from its first tracked point, reconstructed at
    speed_km_h.

    Points past distance_m are left out, and the swing is measured at the last point left,
    the same way as at the last tracked point. It is then carried on to distance_m as a
    constant side force would carry it: multiplied by unit_gap at distance_m over unit_gap
    at that point. For unit_gap at distance_m, the frames after that point are filled in at
    the delivery's frame rate and speed, as if the ball had been tracked all the way. A
    track cut at distance_m is only carried on over the last part of a frame.

    Returns None if fewer than MIN_POINTS points come before distance_m, or if the baseline
    and projection fits would use the same points.
    """
    if not speed_km_h > 0:
        return None
    trajectory = engine.reconstruct_trajectory(points, speed_km_h)
    beyond = np.flatnonzero(trajectory.y > distance_m)
    used = list(points) if beyond.size == 0 else list(points[:int(beyond[0])])
    if len(used) < MIN_POINTS:
        return None
    result = engine.analyse(used, speed_km_h)
    y = result.trajectory.y
    baseline_y = y[:result.baseline_points]
    gap_there = unit_gap(baseline_y, y[-result.extrapolation_points:], result.last_point_distance_m)
    if gap_there < MIN_UNIT_GAP_M2:
        return None

    # Where the ball would have been in each frame after the last one used, up to the
    # common distance, had it been tracked that far.
    first_frame, speed_m_s, drag = used[0].frame, speed_km_h / 3.6, trajectory.drag_constant_per_m
    filled = [float(distance) for distance in y]
    frame = used[-1].frame + 1
    while (ahead := engine.forward_distance((frame - first_frame) / trajectory.fps, speed_m_s, drag)) <= distance_m:
        filled.append(ahead)
        frame += 1
    gap_at_distance = unit_gap(baseline_y, np.array(filled[-result.extrapolation_points:], dtype=np.float64),
                               distance_m)

    return CommonDistanceSwing(
        swing_cm=result.swing_at_last_point_cm * gap_at_distance / gap_there,
        last_point_distance_m=result.last_point_distance_m,
        extrapolated_m=distance_m - result.last_point_distance_m,
        points_left_out=len(points) - len(used),
    )


def swing_values(recorded: RecordedDelivery, speed_km_h: float, speed_source: str, error_percent: float,
                 result: SwingResult, slower: SwingResult, faster: SwingResult, distance_m: float,
                 at_distance: tuple[CommonDistanceSwing | None, CommonDistanceSwing | None,
                                    CommonDistanceSwing | None]) -> SwingValues:
    """Return a delivery's row of swing_seam_comparison.csv, from its analyses at the speed
    used and at the slower and faster speeds, and its swings at the common distance
    distance_m at those three speeds, in that order.
    """
    automatic = recorded.seams.get("automatic")
    points = recorded.side_on_points
    common, common_slower, common_faster = at_distance
    return SwingValues(
        ball=recorded.ball,
        delivery=recorded.delivery,
        manual_seam_deg=recorded.seam_angle("manual"),
        automatic_seam_deg=recorded.seam_angle("automatic"),
        automatic_spread_deg=None if automatic is None else automatic.spread_deg,
        speed_km_h=speed_km_h,
        speed_source=speed_source,
        speed_error_percent=error_percent,
        side_on_points=len(points),
        first_frame=points[0].frame,
        last_frame=points[-1].frame,
        last_point_distance_m=result.last_point_distance_m,
        swing_cm=result.swing_at_last_point_cm,
        swing_slower_cm=slower.swing_at_last_point_cm,
        swing_faster_cm=faster.swing_at_last_point_cm,
        baseline_residual_cm=result.baseline_residual_cm,
        projection_residual_cm=result.projection_residual_cm,
        side_force_coefficient=implied_side_force_coefficient(result),
        common_distance_m=distance_m,
        common_swing_cm=None if common is None else common.swing_cm,
        common_swing_slower_cm=None if common_slower is None else common_slower.swing_cm,
        common_swing_faster_cm=None if common_faster is None else common_faster.swing_cm,
        extrapolated_m=None if common is None else common.extrapolated_m,
        points_past_common_distance=None if common is None else common.points_left_out,
    )


def main(argv: list[str] | None = None) -> None:
    """Calculate the swing of the recorded deliveries at the last tracked point and at the
    common distance, at the speed chosen and at the speed error either side of it, save each
    delivery's reconstruction, and save the swing values.
    """
    parser = argparse.ArgumentParser(description="Calculate the swing of the recorded deliveries at the last "
                                                 "tracked point and at a common distance, at a set speed.")
    parser.add_argument("--speed", type=speed_choice, default="radar",
                        help="speed to reconstruct the deliveries at: radar (each delivery's radar speed, the "
                             "default), top-down (its mean top-down speed), or a speed in km/h for all of them")
    parser.add_argument("--speed-error", type=float, default=SPEED_ERROR_PERCENT,
                        help=f"speed error for the error bars, in percent: the swing is also calculated this far "
                             f"below and above the speed (default {SPEED_ERROR_PERCENT:g})")
    parser.add_argument("--distance", type=float, default=COMMON_DISTANCE_M,
                        help=f"common distance from the first tracked point to give every delivery's swing at, in "
                             f"metres: longer tracks are cut there and shorter ones carried on to it "
                             f"(default {COMMON_DISTANCE_M:g})")
    parser.add_argument("--calibration", type=Path, default=paths.CALIBRATION_PATH,
                        help=f"side-on camera calibration (default {paths.CALIBRATION_PATH.name})")
    parser.add_argument("--balls", nargs="+", type=int, metavar="BALL", help="only these balls (default: all)")
    parser.add_argument("--plots", action="store_true",
                        help=f"also save each delivery's swing plot, {SWING_PLOT_NAME}, in its folder")
    args = parser.parse_args(argv)
    if not 0 <= args.speed_error < 100:
        parser.error("--speed-error has to be from 0 to under 100 percent")
    if not args.distance > 0:
        parser.error("--distance has to be above 0 m")
    if not args.calibration.exists():
        print(f"There is no calibration at {args.calibration}.")
        raise SystemExit(1)

    recorded = [delivery for delivery in load_deliveries(RECORDED_FOLDER)
                if args.balls is None or delivery.ball in args.balls]
    if not recorded:
        print(f"No deliveries are recorded in {RECORDED_FOLDER}"
              + ("" if args.balls is None else " for those balls")
              + " yet. Record them with record_delivery.py first.")
        raise SystemExit(1)
    radar = load_radar_speeds(RADAR_PATH) if RADAR_PATH.exists() else {}
    top_down = load_calculated_speeds(SPEEDS_PATH)

    values = []
    for delivery in recorded:
        speed = delivery_speed(delivery, args.speed, radar, top_down)
        if speed is None:
            print(f"{delivery.label}: left out, it has no radar or top-down speed")
            continue
        speed_km_h, source = speed
        try:
            engine, result = analyse_delivery(delivery, speed_km_h, args.calibration)
        except ValueError as error:
            print(f"{delivery.label}: left out, {error}")
            continue
        speeds = (speed_km_h, speed_km_h * (1 - args.speed_error / 100), speed_km_h * (1 + args.speed_error / 100))
        slower = engine.analyse(delivery.side_on_points, speeds[1])
        faster = engine.analyse(delivery.side_on_points, speeds[2])
        common, common_slower, common_faster = (swing_at_common_distance(engine, delivery.side_on_points, at_speed,
                                                                          args.distance) for at_speed in speeds)
        folder = delivery_folder(delivery.ball, delivery.delivery)
        engine.save_data_to_files(result, folder, points=delivery.side_on_points)
        if args.plots:
            engine.plot_swing(result, save_path=str(folder / SWING_PLOT_NAME))
        row = swing_values(delivery, speed_km_h, source, args.speed_error, result, slower, faster, args.distance,
                           (common, common_slower, common_faster))
        values.append(row)
        seams = ", ".join(f"{seam.angle_deg:+.1f} deg {seam.method}" for seam in delivery.seams.values())
        low, high = row.swing_range_cm
        print(f"{delivery.label}: seam {seams}; {speed_km_h:.1f} km/h ({source}); {row.side_on_points} points to "
              f"{row.last_point_distance_m:.1f} m; swing there {row.swing_cm:+.1f} cm ({low:+.1f} to {high:+.1f} cm "
              f"for ±{args.speed_error:g}% speed)")
        print("    " + _common_note(row, common, args.distance))
    if not values:
        print("No delivery could be calculated.")
        raise SystemExit(1)

    save_swing_values(values, TABLE_PATH)
    print(f"\nSaved {TABLE_PATH.name} to {FOLDER}. Plot it with plot_swing_seam.py.")


def _common_note(row: SwingValues, common: CommonDistanceSwing | None, distance_m: float) -> str:
    """Return the line printed under a delivery about its swing at the common distance, distance_m."""
    distance = f"{distance_m:g} m"
    if common is None or row.common_swing_cm is None:
        return f"at {distance}: none, too few points before it"
    if common.points_left_out:
        reached = f"{common.points_left_out} points past {distance} left out"
    else:
        reached = f"carried on {common.extrapolated_m:.1f} m from the last tracked point"
    spread = row.common_swing_range_cm
    swings = "" if spread is None else f" ({spread[0]:+.1f} to {spread[1]:+.1f} cm)"
    return f"at {distance}: swing {row.common_swing_cm:+.1f} cm{swings}, {reached}"


if __name__ == "__main__":
    main()
