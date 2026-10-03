"""Old file: text-based version of the cricket ball tracker.

Kept for reference only; the current program is main/GUI_new_cricket_ball_tracker.py.
"""

import sys
import yaml
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import paths
from library.manual_tracker import TopDownTracker, SideOnTracker
from library.physics_engines.top_down_physics_engine import SelectionType, TopDownPhysicsEngine
from library.physics_engines.side_on_physics_engine import SideOnPhysicsEngine

SAVE_DIR = str(paths.DELIVERY_DIR)
"""Folder the delivery's results and plots are saved in."""
TOP_DOWN_VALUE_FILE = "top_down_analysis"
"""File name of the top-down results in the save folder, without the .yaml extension."""
SIDE_ON_VALUE_FILE = "side_on_analysis"
"""Unused name for the side-on results file; the side-on engine names its own files."""
SWING_PLOT_FILE = "swing"
"""File name of the swing plot in the save folder, without the .png matplotlib adds."""
TRAJECTORY_PLOT_FILE = "trajectory_3d"
"""File name of the 3D trajectory plot in the save folder, without the .png matplotlib
adds.
"""

CALIBRATION_PATH = str(paths.CALIBRATION_PATH)
"""Side-on camera calibration file."""
DISPLAY_3D_PLOT = False
"""Whether the 3D trajectory plot is shown in a window as well as saved."""

def top_down():
    """Let the user mark the ball and seam in the top-down video, then print and save
    the speed and seam angle. Return the mean speed and the speeds between consecutive
    marked points in km/h, and the seam angle in degrees (or None).
    """
    # Click points on top-down video
    tracker = TopDownTracker()
    clicked_points = tracker.get_top_down_points()

    engine = TopDownPhysicsEngine()
    fps = tracker.top_down_video.fps

    # Calculate velocity using the clicked points and fps
    velocity, velocity_list = engine.calculate_velocity(clicked_points, fps=fps, type=SelectionType.MEAN)
    print(f"Calculated velocity: {velocity:.2f} km/h")

    # Calculate seam angle
    seam_angle = engine.calculate_seam_angle(clicked_points)
    print(f"Calculated seam angle: {seam_angle}")

    # Save values to file
    path = engine.save_top_down_analysis(SAVE_DIR, TOP_DOWN_VALUE_FILE, velocity, seam_angle, fps, len(clicked_points))
    print(f"Top down values saved to {path}")

    return velocity, velocity_list, seam_angle

def side_on(velocity):
    """Let the user mark the ball in the side-on video, then print and save its 3D path,
    swing and plots, at the given speed in km/h or one the user enters.
    """
    print(f"Top down velocity was calculated to be {velocity:.2f} km/h")
    auto_vel = input("Do you want to use this calculated velocity? (y/n)\n")
    if auto_vel == "n":
        velocity = float(input("New velocity: "))

    tracker = SideOnTracker()
    clicked_points = tracker.get_side_on_points()

    fps = tracker.side_on_video.fps

    engine = SideOnPhysicsEngine.from_calibration_file(CALIBRATION_PATH, fps=fps)
    calibration = engine.calibration

    print(f"calibration: focal {calibration.focal_px[0]:.1f} px, "
            f"principal point {calibration.principal_point[0]:.1f},"
            f"{calibration.principal_point[1]:.1f}")
    print(f"camera centre (X, Y, Z) = {np.round(calibration.camera_centre, 3)}")
    print(f"{len(clicked_points)} tracked points, frames "
            f"{clicked_points[0].frame}-{clicked_points[-1].frame}\n")

    trajectory = engine.reconstruct_trajectory(clicked_points, velocity)

    print(f"swing_at_last_tracked_point : "
            f"{engine.swing_at_last_tracked_point(trajectory):+.2f} cm")
    print(f"swing_at_17m                : "
            f"{engine.swing_at_17m(trajectory):+.2f} cm")

    cubes = engine.ball_coordinates(trajectory, origin="cubes")
    release = engine.ball_coordinates(trajectory, origin="release")
    print(f"\n{'frame':>6} {'X':>8} {'Y':>8} {'Z':>8}   "
            f"{'dX':>8} {'dY':>8} {'dZ':>8}")
    for i in (0, len(cubes) // 2, len(cubes) - 1):
        f = trajectory.frames[i]
        print(f"{f:>6} {cubes[i,0]:>8.3f} {cubes[i,1]:>8.3f} {cubes[i,2]:>8.3f}   "
                f"{release[i,0]:>8.3f} {release[i,1]:>8.3f} {release[i,2]:>8.3f}")

    result = engine.analyse(clicked_points, velocity)
    print("\n" + result.summary())

    written = engine.save_data_to_files(result, SAVE_DIR, points=clicked_points)
    print("\nwrote " + "\nwrote ".join(written.values()))

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    swing_path = engine.plot_swing(result, save_path=str(Path(SAVE_DIR) / SWING_PLOT_FILE))
    print(f"wrote {swing_path}")

    engine.plot_trajectory_3d(result, show=DISPLAY_3D_PLOT, save_path=str(Path(SAVE_DIR) / TRAJECTORY_PLOT_FILE))


vel, vel_list, angle = top_down()

ball = input("Enter ball number: ")
delivery_num = input("Enter delivery number: ")

with open(paths.REPORT_DATA_DIR / "radar_speeds.yaml", "r") as f:
    radar_data = yaml.safe_load(f)
    try:
        speed = radar_data[f"ball_{ball}"][f"delivery_{delivery_num}"]
        print(f"Radar speed for ball is {speed}")
    except KeyError:
        print(f"Radar speed for ball {ball} delivery {delivery_num} not found in radar_speeds.yaml")
        sys.exit()

with open(paths.REPORT_DATA_DIR / "actual_top_down_data.csv", "a") as f:
    f.write(f"{ball},{delivery_num},{[int(v) for v in vel_list]},{int(angle)}\n")
    print(f"Data saved as {ball},{delivery_num},{[int(v) for v in vel_list]},{int(angle)}")



sys.exit()

print(f"Using calculated velocity {vel}.")

side_on(vel)

print("\nTracking fininshed.")
print(f"All data saved to {SAVE_DIR}")
