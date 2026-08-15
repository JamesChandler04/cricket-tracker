from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Config

# Points from release used to fit the swingless baseline. Too few and the fit
# is noise-dominated; too many and the baseline bends to follow real swing and
# biases every result low. Keep it under about a third of the track length.
BASELINE_SOURCE_POINT_COUNT = 10

# Trailing points used to fit the projection line. Fix this once and keep it
# constant across deliveries, or comparisons between them are meaningless.
EXTRAPOLATION_SOURCE_POINT_COUNT = 10

# Forward distance at which the projected swing is reported. Well past the
# tracked range, so it is a lower bound rather than a measurement.
TARGET_DISTANCE_M = 17.0

# Quadratic drag on forward motion. Sets how tracked time maps to forward
# distance.
DRAG_COEFFICIENT = 0.0092

GRAVITY = 9.81

# --- calibration cube geometry, for the 3D plot -----------------------------
# Rings are squares of side CUBE_SIDE_M at these forward distances, centred
# laterally on CUBE_CENTRE_X_M and vertically on CUBE_CENTRE_Z_M. Defaults
# assume the calibration origin is the centre of the first ring. Adjust to
# match how the cubes were actually clicked.
RING_FORWARD_DISTANCES_M = (0.0, 3.0, 6.0, 9.0)
CUBE_SIDE_M = 3.0
CUBE_CENTRE_X_M = 0.0
CUBE_CENTRE_Z_M = 0.0

_BLUE, _ORANGE, _GREEN, _GREY, _INK = '#2a78d6', '#eb6834', '#1baf7a', '#888780', '#111111'

# Data types

@dataclass(frozen=True)
class TrackedPoint:
    """One clicked ball centre in the side-on view."""
    frame: int
    u: float
    v: float


@dataclass(frozen=True)
class Calibration:
    K_inv: np.ndarray
    R_T: np.ndarray
    t_std: np.ndarray

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["K_inv"], d["R_T"], d["t_std"])

    @property
    def focal_px(self):
        return 1.0 / self.K_inv[0, 0], 1.0 / self.K_inv[1, 1]

    @property
    def principal_point(self):
        fx, fy = self.focal_px
        return -self.K_inv[0, 2] * fx, -self.K_inv[1, 2] * fy

    @property
    def camera_centre(self):
        return -self.R_T @ self.t_std

    @property
    def implied_frame_size(self):
        """Approximate frame size this calibration was built at."""
        cx, cy = self.principal_point
        return 2.0 * cx, 2.0 * cy

    def project(self, xyz):
        """World point -> pixel. Exact inverse of the reconstruction."""
        K = np.linalg.inv(self.K_inv)
        cam = self.R_T.T @ np.asarray(xyz, float) + self.t_std
        p = K @ cam
        return p[0] / p[2], p[1] / p[2]


@dataclass
class Trajectory:
    """A reconstructed delivery in world (cube-relative) coordinates."""
    frames: np.ndarray
    times_s: np.ndarray
    world: np.ndarray            # (X, Y, Z) X right=positive, Y forward, Z down=positive
    fps: float
    speed_km_h: float
    drag: float

    @property
    def x(self): return self.world[:, 0]

    @property
    def y(self): return self.world[:, 1]

    @property
    def z(self): return self.world[:, 2]

    @property
    def relative_to_release(self):
        return self.world - self.world[0]

    @property
    def last_distance_m(self):
        return float(self.world[-1, 1])


@dataclass
class SwingResult:
    swing_at_last_point_cm: float
    swing_at_17m_cm: float

    baseline_slope: float
    baseline_intercept: float
    baseline_residual_cm: float

    projection_slope: float
    projection_intercept: float
    projection_residual_cm: float

    last_point_offset_cm: float
    last_point_distance_m: float
    target_distance_m: float

    baseline_points: int
    extrapolation_points: int

    trajectory: Trajectory = field(repr=False)

    def summary(self) -> str:
        return (
            f"swing at last tracked point ({self.last_point_distance_m:.2f} m): "
            f"{self.swing_at_last_point_cm:+.2f} cm\n"
            f"swing at {self.target_distance_m:.0f} m: "
            f"{self.swing_at_17m_cm:+.2f} cm   [lower bound, projected]\n"
            f"\n"
            f"baseline:   first {self.baseline_points} points, "
            f"residual {self.baseline_residual_cm:.2f} cm RMS\n"
            f"projection: last  {self.extrapolation_points} points, "
            f"residual {self.projection_residual_cm:.2f} cm RMS\n"
            f"final point sits {self.last_point_offset_cm:+.2f} cm off the projection"
        )


# Reconstruction

def forward_distance(t, speed_m_s, drag=DRAG_COEFFICIENT):
    """Distance travelled at time t under quadratic drag."""
    if drag <= 0:
        return speed_m_s * t
    return math.log(1.0 + drag * speed_m_s * t) / drag


def reconstruct_trajectory(points, fps, speed_km_h, calibration,
                           drag=DRAG_COEFFICIENT) -> Trajectory:
    """
    Pixel track -> 3D world track.

    Each pixel gives only a ray from the camera. The speed-and-drag model
    supplies the forward distance that fixes where along that ray the ball sits.
    """
    speed_m_s = speed_km_h / 3.6
    frame0 = points[0].frame
    row_y = calibration.R_T[1, :]

    frames, times, world = [], [], []
    for p in points:
        t = (p.frame - frame0) / fps
        y = forward_distance(t, speed_m_s, drag)
        ray = calibration.K_inv @ np.array([p.u, p.v, 1.0])
        scale = (y + row_y @ calibration.t_std) / (row_y @ ray)
        world.append(calibration.R_T @ (scale * ray - calibration.t_std))
        frames.append(p.frame)
        times.append(t)

    return Trajectory(np.asarray(frames), np.asarray(times), np.asarray(world),
                      float(fps), float(speed_km_h), float(drag))


def ball_coordinates(trajectory: Trajectory, origin="cubes") -> np.ndarray:
    """
    Ball position in metres, as an (N, 3) array of X, Y, Z.

    origin="cubes"    world frame set by the clicked cube corners: Y = 0 at the
                      release ring, X positive to the bowler's right, Z positive
                      downward.
    origin="release"  same axes, shifted so the first tracked point is (0, 0, 0).
    """
    if origin == "cubes":
        return trajectory.world.copy()
    if origin == "release":
        return trajectory.relative_to_release
    raise ValueError(f"origin must be 'cubes' or 'release', got {origin!r}")


# ---------------------------------------------------------------------------
# Linear fits
# ---------------------------------------------------------------------------

def _fit_line(y, x):
    slope, intercept = np.polyfit(y, x, 1)
    rms_cm = float(np.std(x - (intercept + slope * y)) * 100.0)
    return float(slope), float(intercept), rms_cm


def fit_swingless_baseline(trajectory, n_points=BASELINE_SOURCE_POINT_COUNT):
    """Straight line through the first n_points - the no-swing path."""
    n = min(n_points, len(trajectory.world))
    if n < 3:
        raise ValueError(f"need at least 3 points for a baseline, got {n}")
    return _fit_line(trajectory.y[:n], trajectory.x[:n])


def fit_projection(trajectory, n_points=EXTRAPOLATION_SOURCE_POINT_COUNT):
    """Straight line through the final n_points."""
    n = min(n_points, len(trajectory.world))
    if n < 3:
        raise ValueError(f"need at least 3 points for a projection, got {n}")
    return _fit_line(trajectory.y[-n:], trajectory.x[-n:])


def line_x(y, slope, intercept):
    return intercept + slope * np.asarray(y, dtype=float)


def _gap_cm(y, b, p):
    return float((line_x(y, p[0], p[1]) - line_x(y, b[0], b[1])) * 100.0)


# ---------------------------------------------------------------------------
# The two swing values
# ---------------------------------------------------------------------------

def swing_at_last_tracked_point(trajectory: Trajectory,
                                baseline_points=BASELINE_SOURCE_POINT_COUNT,
                                projection_points=EXTRAPOLATION_SOURCE_POINT_COUNT) -> float:
    """
    Swing in cm at the last tracked point: the gap between the swingless
    baseline and the projection line, both evaluated there.

    Negative is to the bowler's left. This is the measured value - it sits
    inside the tracked data and is stable to a few millimetres across sensible
    window sizes. It is the number to report.
    """
    b = fit_swingless_baseline(trajectory, baseline_points)
    p = fit_projection(trajectory, projection_points)
    return _gap_cm(trajectory.last_distance_m, b, p)


def swing_at_17m(trajectory: Trajectory,
                 baseline_points=BASELINE_SOURCE_POINT_COUNT,
                 projection_points=EXTRAPOLATION_SOURCE_POINT_COUNT,
                 target_distance_m=TARGET_DISTANCE_M) -> float:
    """
    Swing in cm at TARGET_DISTANCE_M: the same gap, extended to 17 m.

    A straight projection assumes the lateral velocity at the end of tracking
    persists, but swing accelerates - so this is a LOWER BOUND. It also depends
    heavily on projection_points, since it extrapolates most of the way again
    past the tracked range. Quote it as a range, not a figure.
    """
    b = fit_swingless_baseline(trajectory, baseline_points)
    p = fit_projection(trajectory, projection_points)
    return _gap_cm(target_distance_m, b, p)


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------

def analyse(points, fps, speed_km_h, calibration,
            drag=DRAG_COEFFICIENT,
            baseline_points=BASELINE_SOURCE_POINT_COUNT,
            projection_points=EXTRAPOLATION_SOURCE_POINT_COUNT,
            target_distance_m=TARGET_DISTANCE_M) -> SwingResult:
    traj = reconstruct_trajectory(points, fps, speed_km_h, calibration, drag)

    b_slope, b_int, b_rms = fit_swingless_baseline(traj, baseline_points)
    p_slope, p_int, p_rms = fit_projection(traj, projection_points)
    b, p = (b_slope, b_int), (p_slope, p_int)

    y_last = traj.last_distance_m
    offset_cm = float((traj.x[-1] - line_x(y_last, p_slope, p_int)) * 100.0)

    return SwingResult(
        swing_at_last_point_cm=_gap_cm(y_last, b, p),
        swing_at_17m_cm=_gap_cm(target_distance_m, b, p),
        baseline_slope=b_slope, baseline_intercept=b_int, baseline_residual_cm=b_rms,
        projection_slope=p_slope, projection_intercept=p_int, projection_residual_cm=p_rms,
        last_point_offset_cm=offset_cm,
        last_point_distance_m=y_last,
        target_distance_m=float(target_distance_m),
        baseline_points=min(baseline_points, len(traj.world)),
        extrapolation_points=min(projection_points, len(traj.world)),
        trajectory=traj,
    )


# ---------------------------------------------------------------------------
# Cube geometry
# ---------------------------------------------------------------------------

def ring_corners(forward_distance_m, side=CUBE_SIDE_M,
                 centre_x=CUBE_CENTRE_X_M, centre_z=CUBE_CENTRE_Z_M):
    """Four corners of one calibration ring, in world coordinates."""
    h = side / 2.0
    return np.array([
        [centre_x - h, forward_distance_m, centre_z - h],
        [centre_x + h, forward_distance_m, centre_z - h],
        [centre_x + h, forward_distance_m, centre_z + h],
        [centre_x - h, forward_distance_m, centre_z + h],
    ])


def cube_edges(distances=RING_FORWARD_DISTANCES_M, **kw):
    """Line segments for the cube wireframe: each ring, plus the rails between."""
    rings = [ring_corners(d, **kw) for d in distances]
    segments = []
    for r in rings:
        for i in range(4):
            segments.append((r[i], r[(i + 1) % 4]))
    for a, b in zip(rings, rings[1:]):
        for i in range(4):
            segments.append((a[i], b[i]))
    return segments


# ---------------------------------------------------------------------------
# 2D swing plot
# ---------------------------------------------------------------------------

def plot_swing(result: SwingResult, save_path="swing.png", show=False):
    """Trajectory, baseline, projection, and both swing values as measured gaps."""
    import matplotlib.pyplot as plt

    traj = result.trajectory
    target = result.target_distance_m
    n_fit, n_proj = result.baseline_points, result.extrapolation_points
    b = (result.baseline_slope, result.baseline_intercept)
    p = (result.projection_slope, result.projection_intercept)

    y_grid = np.linspace(0.0, target * 1.03, 300)
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(y_grid, line_x(y_grid, *b) * 100, '-', color=_INK, lw=2.2, zorder=4,
            label=f"swingless baseline (first {n_fit} points)")
    ax.plot(y_grid, line_x(y_grid, *p) * 100, '--', color=_ORANGE, lw=2.6, zorder=3,
            label=f"projection (last {n_proj} points)")
    ax.plot(traj.y, traj.x * 100, 'o', color=_BLUE, ms=5, zorder=5, label="tracked points")
    ax.plot(traj.y[:n_fit], traj.x[:n_fit] * 100, 'o', color=_GREEN, ms=6, zorder=6,
            label="baseline fit points")
    ax.plot(traj.y[-n_proj:], traj.x[-n_proj:] * 100, 'o', color=_ORANGE, ms=6, zorder=6,
            label="projection fit points")

    y_last = result.last_point_distance_m
    ax.axvline(y_last, color='#c3c2b7', lw=1.0, ls=':', zorder=1)

    for yv, val, col, lbl, off in (
        (y_last, result.swing_at_last_point_cm, _BLUE,
         f"swing at last tracked point\n{result.swing_at_last_point_cm:+.2f} cm\nat Y = {y_last:.2f} m",
         (-152, -20)),
        (target, result.swing_at_17m_cm, '#993C1D',
         f"swing at {target:.0f} m\n{result.swing_at_17m_cm:+.2f} cm\n[lower bound]", (-118, 10)),
    ):
        xb, xp = float(line_x(yv, *b)) * 100, float(line_x(yv, *p)) * 100
        ax.annotate("", xy=(yv, xp), xytext=(yv, xb),
                    arrowprops=dict(arrowstyle='<->', color=col, lw=2.4,
                                    shrinkA=0, shrinkB=0), zorder=7)
        ax.annotate(lbl, (yv, 0.5 * (xp + xb)), textcoords="offset points",
                    xytext=off, fontsize=10, color=col, ha='left',
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=col, lw=0.8))

    ax.set_xlabel("forward distance from release, Y (m)")
    ax.set_ylabel("lateral position, X (cm)   -  negative is the bowler's left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc='best')
    ax.set_title(f"Swing analysis at {traj.speed_km_h:.1f} km/h ", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 3D trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory_3d(result: SwingResult, show=True, save_path=None,
                       distances=RING_FORWARD_DISTANCES_M,
                       show_baseline=True, elev=18, azim=-70):
    """
    Full 3D trajectory with the calibration cubes drawn as reference.

    With show=True the figure opens in an interactive window: drag with the left
    mouse button to rotate, scroll or right-drag to zoom. That needs a GUI
    backend - you have PyQt5, so matplotlib will pick Qt5Agg automatically.
    Headless, set MPLBACKEND=Agg and pass show=False with a save_path.

    Z is negated for display so that up is up on screen.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

    traj = result.trajectory
    n_fit, n_proj = result.baseline_points, result.extrapolation_points

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for a, b in cube_edges(distances):
        ax.plot([a[1], b[1]], [a[0], b[0]], [-a[2], -b[2]],
                color=_GREY, lw=1.0, alpha=0.55, zorder=1)
    for d in distances:
        r = ring_corners(d)
        ax.text(d, r[1, 0], -r[1, 2] + 0.15, f"{d:.0f} m",
                color=_GREY, fontsize=8)

    ax.plot(traj.y, traj.x, -traj.z, '-', color=_BLUE, lw=1.6, alpha=0.8)
    ax.scatter(traj.y, traj.x, -traj.z, c=_BLUE, s=16, depthshade=False,
               label="tracked ball")
    ax.scatter(traj.y[:n_fit], traj.x[:n_fit], -traj.z[:n_fit], c=_GREEN, s=28,
               depthshade=False, label=f"baseline points (first {n_fit})")
    ax.scatter(traj.y[-n_proj:], traj.x[-n_proj:], -traj.z[-n_proj:], c=_ORANGE,
               s=28, depthshade=False, label=f"projection points (last {n_proj})")

    if show_baseline:
        yg = np.linspace(0.0, max(traj.y[-1], max(distances)), 60)
        xb = line_x(yg, result.baseline_slope, result.baseline_intercept)
        # carry the vertical motion along so the comparison line is physical
        zc = np.polyfit(traj.y, traj.z, 2)
        ax.plot(yg, xb, -np.polyval(zc, yg), '--', color=_INK, lw=1.4,
                alpha=0.85, label="swingless baseline")

    cam = None
    ax.set_xlabel("Y  forward down the pitch (m)")
    ax.set_ylabel("X  lateral, + is bowler's right (m)")
    ax.set_zlabel("height (m)")

    ymax = max(float(traj.y.max()), max(distances))
    half = CUBE_SIDE_M / 2.0 + 0.2
    ax.set_xlim(0, ymax)
    ax.set_ylim(CUBE_CENTRE_X_M - half, CUBE_CENTRE_X_M + half)
    ax.set_zlim(-CUBE_CENTRE_Z_M - half, -CUBE_CENTRE_Z_M + half)
    ax.set_box_aspect((ymax, 2 * half, 2 * half))
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title(f"Delivery reconstruction at {traj.speed_km_h:.1f} km/h   -   "
                 f"swing {result.swing_at_last_point_cm:+.1f} cm at "
                 f"{result.last_point_distance_m:.1f} m", fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Demo: delivery 724-777, side-on, 2704x1520 at 239.76 fps, 120 km/h
#
# Wanger release in the vertical plane, seam rotated 30 degrees counter-
# clockwise viewed from above, so conventional swing is expected to the
# bowler's left. The reconstruction returns -8.38 cm, i.e. left.
# ---------------------------------------------------------------------------

DEMO_CALIBRATION = "camera_calibration.npz"
DEMO_FPS = 240.0
DEMO_SPEED_KMH = 120.0

DEMO_POINTS = [
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

SWINGLESS_CALIBRATION = "reference_swingless_calibration.npz"
SWINGLESS_FPS = 120.0
SWINGLESS_SPEED_KMH = 120.0

DEMO_POINTS_SINGLESS = [
    (428, 1710, 721), (429, 1741, 728), (430, 1765, 734), (431, 1789, 740),
    (432, 1810, 746), (433, 1829, 750), (434, 1847, 755), (435, 1862, 759),
    (436, 1876, 763), (437, 1890, 767), (438, 1902, 771), (439, 1914, 775),
    (440, 1924, 779), (441, 1934, 782), (442, 1942, 785), (443, 1950, 789),
    (444, 1958, 793), (445, 1965, 797), (446, 1972, 800), (447, 1979, 803),
    (448, 1985, 807), (449, 1992, 810), (450, 1997, 813), (451, 2002, 816),
    (452, 2007, 819), (453, 2011, 822), (454, 2016, 825), (455, 2019, 829),
    (456, 2024, 832), (457, 2028, 836), (458, 2032, 839), (459, 2034, 841),
    (460, 2039, 844), (461, 2041, 847), (462, 2045, 850), (463, 2048, 852),
    (464, 2052, 855),
]


def _demo():
    calibration = Calibration.load(DEMO_CALIBRATION)
    points = [TrackedPoint(f, u, v) for f, u, v in DEMO_POINTS]

    print(f"calibration: focal {calibration.focal_px[0]:.1f} px, "
          f"principal point {calibration.principal_point[0]:.1f},"
          f"{calibration.principal_point[1]:.1f}")
    print(f"camera centre (X, Y, Z) = {np.round(calibration.camera_centre, 3)}")
    print(f"{len(points)} tracked points, frames "
          f"{points[0].frame}-{points[-1].frame}\n")

    trajectory = reconstruct_trajectory(points, DEMO_FPS, DEMO_SPEED_KMH, calibration)

    print(f"swing_at_last_tracked_point : "
          f"{swing_at_last_tracked_point(trajectory):+.2f} cm")
    print(f"swing_at_17m                : {swing_at_17m(trajectory):+.2f} cm")

    cubes = ball_coordinates(trajectory, origin="cubes")
    release = ball_coordinates(trajectory, origin="release")
    print(f"{'frame':>6} {'X':>8} {'Y':>8} {'Z':>8}   {'dX':>8} {'dY':>8} {'dZ':>8}")
    for i in (0, len(cubes) // 2, len(cubes) - 1):
        f = trajectory.frames[i]
        print(f"{f:>6} {cubes[i,0]:>8.3f} {cubes[i,1]:>8.3f} {cubes[i,2]:>8.3f}   "
              f"{release[i,0]:>8.3f} {release[i,1]:>8.3f} {release[i,2]:>8.3f}")

    result = analyse(points, DEMO_FPS, DEMO_SPEED_KMH, calibration)
    print("\n" + result.summary())

    plot_swing(result, save_path="swing.png")
    print("\nwrote swing.png")
    plot_trajectory_3d(result, show=True)      # drag to rotate, scroll to zoom


if __name__ == "__main__":
    _demo()