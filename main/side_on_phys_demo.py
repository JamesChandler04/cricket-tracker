"""Worked example of the side-on physics engine on one delivery's clicked points."""

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Started as a plain script (python main/<name>.py, or the Run button in
    # VS Code), so put the project folder on the path for the library imports.
    # python -m main.<name> from the project folder does not need this.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library import paths
from library.physics_engines.side_on_physics_engine import SideOnPhysicsEngine, TrackedPoint


CALIBRATION_PATH = str(paths.CALIBRATION_PATH)
"""Side-on camera calibration file the demo uses."""
FPS = 240.0
"""Frame rate of the side-on video the points were clicked on, in fps."""
SPEED_KMH = 120.0
"""Delivery speed used to reconstruct the delivery, in km/h."""
SAVE_DIRECTORY = str(paths.DELIVERY_DIR)
"""Folder the demo writes tracked_points.csv and side_on_analysis.yaml to."""
DISPLAY_3D_PLOT = False
"""Whether to open the interactive 3D trajectory plot at the end."""

POINTS = [
    (724, 895, 386), (725, 908, 389), (726, 919, 392), (727, 930, 394),
    (728, 939, 396), (729, 949, 399), (730, 958, 401), (731, 966, 404),
    (732, 974, 405), (733, 982, 408), (734, 988, 410), (735, 995, 411),
    (736, 1003, 413), (737, 1008, 415), (738, 1013, 417), (739, 1019, 419),
    (740, 1023, 420), (741, 1029, 422), (742, 1033, 423), (743, 1038, 424),
    (744, 1042, 426), (745, 1046, 428), (746, 1050, 429), (747, 1054, 430),
    (748, 1058, 432), (749, 1061, 433), (750, 1065, 434), (751, 1069, 434),
    (752, 1071, 436), (753, 1074, 438), (754, 1078, 439), (755, 1081, 441),
    (756, 1083, 442), (757, 1086, 443), (758, 1088, 444), (759, 1091, 446),
    (760, 1093, 447), (761, 1095, 448), (762, 1098, 449), (763, 1100, 450),
    (764, 1102, 451), (765, 1104, 452), (766, 1106, 453), (767, 1108, 453),
    (768, 1109, 454), (769, 1111, 455), (770, 1113, 457), (771, 1115, 458),
    (772, 1117, 459), (773, 1119, 460), (774, 1120, 461), (775, 1122, 462),
    (776, 1123, 463), (777, 1125, 464),
]
"""Clicked ball centres of the delivery, as (frame, u, v) with u and v in px."""


def main():
    """Reconstruct the demo delivery, print the numbers, save them, draw both plots."""
    engine = SideOnPhysicsEngine.from_calibration_file(CALIBRATION_PATH, fps=FPS)
    calibration = engine.calibration
    points = [TrackedPoint(f, u, v) for f, u, v in POINTS]

    print(f"calibration: focal {calibration.focal_px[0]:.1f} px, "
          f"principal point {calibration.principal_point[0]:.1f},"
          f"{calibration.principal_point[1]:.1f}")
    print(f"camera centre (X, Y, Z) = {np.round(calibration.camera_centre, 3)}")
    print(f"{len(points)} tracked points, frames "
          f"{points[0].frame}-{points[-1].frame}\n")

    trajectory = engine.reconstruct_trajectory(points, SPEED_KMH)

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

    result = engine.analyse(points, SPEED_KMH)
    print("\n" + result.summary())

    written = engine.save_data_to_files(result, SAVE_DIRECTORY, points=points)
    print("\nwrote " + "\nwrote ".join(written.values()))

    swing_path = paths.OUTPUT_DIR / "swing.png"
    engine.plot_swing(result, save_path=str(swing_path))
    print(f"wrote {swing_path}")

    engine.plot_trajectory_3d(result, show=DISPLAY_3D_PLOT)


if __name__ == "__main__":
    main()