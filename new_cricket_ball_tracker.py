from manual_tracker import TopDownTracker, SideOnTracker
from top_down_physics_engine import SelectionType, TopDownPhysicsEngine
from side_on_physics_engine import SideOnPhysicsEngine

import numpy as np

SAVE_DIR = "output_folder/demo_delivery"
TOP_DOWN_VALUE_FILE = "top_down_analysis"

CALIBRATION_PATH = "camera_calibration.npz"
DISPLAY_3D_PLOT = False

def top_down():
    # Click points on top-down video
    tracker = TopDownTracker()
    clicked_points = tracker.get_top_down_points()

    engine = TopDownPhysicsEngine()
    fps = tracker.top_down_video.fps

    # Calculate velocity using the clicked points and fps
    velocity = engine.calculate_velocity(clicked_points, fps=fps, type=SelectionType.MEAN)
    print(f"Calculated velocity: {velocity:.2f} km/h")

    # Calculate seam angle
    seam_angle = engine.calculate_seam_angle(clicked_points)
    print(f"Calculated seam angle: {seam_angle}")

    # Save values to file
    path = engine.save_top_down_analysis(SAVE_DIR, TOP_DOWN_VALUE_FILE, velocity, seam_angle, fps, len(clicked_points))
    print(f"Top down values saved to {path}")

    return velocity, seam_angle

def side_on(velocity):
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

    engine.plot_swing(result, save_path="swing.png")
    print("wrote swing.png")

    engine.plot_trajectory_3d(result, show=DISPLAY_3D_PLOT)

vel, angle = top_down()

print(f"Using calculated velocity {vel}.")

side_on(vel)

print("\nTracking fininshed.")
print(f"All data saved to {SAVE_DIR}")
