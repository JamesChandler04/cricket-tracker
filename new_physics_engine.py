"""
Side-on swing analysis for cricket deliveries.

Reconstructs a 3D ball trajectory from a clicked side-on pixel track and
measures lateral swing as the gap between two straight lines: a swingless
baseline fitted to the first points of the delivery, and a projection fitted to
the last points. Negative lateral values are to the bowler's left.

The engine is bound to one camera and one frame rate. Delivery speed is passed
in per call, since it changes from delivery to delivery.

Requires numpy and PyYAML. matplotlib is imported only by the two plot methods.

Public API
----------
Calibration.load(path)
    Load the side-on camera calibration written by the calibration step.

TrackedPoint(frame, u, v)
    One clicked ball centre. Build a list of these for the delivery.

SideOnPhysicsEngine(calibration, fps, ...)
SideOnPhysicsEngine.from_calibration_file(path, fps, ...)
    An engine bound to one camera and one frame rate.

engine.reconstruct_trajectory(points, speed_km_h) -> Trajectory
    Pixel track -> 3D world track.

engine.ball_coordinates(trajectory, origin="cubes" | "release") -> (N, 3) array
    Ball position in metres as X, Y, Z.

engine.swing_at_last_tracked_point(trajectory) -> float
    Measured swing in cm at the last tracked point. This is the number to
    report: it sits inside the tracked data.

engine.swing_at_17m(trajectory) -> float
    Projected swing in cm at TARGET_DISTANCE_M. A lower bound.

engine.analyse(points, speed_km_h) -> SwingResult
    Everything at once: the trajectory, both swing values and the fit
    diagnostics. SwingResult.summary() renders it as text.

engine.save_data_to_files(result, save_directory, points=None) -> dict
    Write the whole analysis to disk: tracked_points.csv holds the per-point
    coordinates, analysis.yaml holds everything else.

engine.plot_swing(result, save_path=..., show=...)
    2D lateral-position plot with both swing gaps annotated.

engine.plot_trajectory_3d(result, show=..., save_path=...)
    Interactive 3D trajectory with the calibration cubes drawn as reference.

Typical use
-----------
    from physics_engine import SideOnPhysicsEngine, TrackedPoint

    engine = SideOnPhysicsEngine.from_calibration_file(
        "camera_calibration.npz", fps=239.76)

    points = [TrackedPoint(f, u, v) for f, u, v in rows]
    result = engine.analyse(points, speed_km_h=120.0)
    print(result.summary())

    engine.save_data_to_files(result, "output/delivery_724", points=points)
    engine.plot_swing(result, save_path="swing.png")
    engine.plot_trajectory_3d(result, show=True)

Configuration
-------------
BASELINE_SOURCE_POINT_COUNT
    Points from release used to fit the swingless baseline. Too few and the fit
    is noise-dominated; too many and the baseline bends to follow real swing and
    biases every result low. Keep it under about a third of the track length.

EXTRAPOLATION_SOURCE_POINT_COUNT
    Trailing points used to fit the projection line. Fix this once and keep it
    constant across deliveries, or comparisons between them are meaningless.

TARGET_DISTANCE_M
    Forward distance at which the projected swing is reported. Well past the
    tracked range, so it is a lower bound rather than a measurement.

DRAG_COEFFICIENT
    Quadratic drag on forward motion. Sets how tracked time maps to forward
    distance.

RING_FORWARD_DISTANCES_M, CUBE_SIDE_M, CUBE_CENTRE_X_M, CUBE_CENTRE_Z_M
    Calibration cube geometry for the 3D plot. Rings are squares of side
    CUBE_SIDE_M at those forward distances, centred laterally on CUBE_CENTRE_X_M
    and vertically on CUBE_CENTRE_Z_M. Defaults assume the calibration origin is
    the centre of the first ring. Adjust to match how the cubes were actually
    clicked.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


BASELINE_SOURCE_POINT_COUNT = 10
EXTRAPOLATION_SOURCE_POINT_COUNT = 10
TARGET_DISTANCE_M = 17.0
DRAG_COEFFICIENT = 0.0092
GRAVITY = 9.81

RING_FORWARD_DISTANCES_M = (0.0, 3.0, 6.0, 9.0)
CUBE_SIDE_M = 3.0
CUBE_CENTRE_X_M = 0.0
CUBE_CENTRE_Z_M = 0.0

POINTS_CSV_NAME = "tracked_points.csv"
ANALYSIS_YAML_NAME = "analysis.yaml"

_BLUE, _ORANGE, _GREEN, _GREY, _INK = '#2a78d6', '#eb6834', '#1baf7a', '#888780', '#111111'


class _YamlDumper(yaml.SafeDumper):
    """SafeDumper tuned for readable output. See the representer below."""


def _represent_list(dumper, data):
    """Write sequences of plain scalars inline, so matrix rows stay on one line."""
    inline = all(v is None or isinstance(v, (bool, int, float)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_YamlDumper.add_representer(list, _represent_list)


@dataclass(frozen=True)
class TrackedPoint:
    """One clicked ball centre in the side-on view."""
    frame: int
    u: float
    v: float


@dataclass(frozen=True)
class Calibration:
    """Side-on camera calibration: inverse intrinsics, rotation and translation."""
    K_inv: np.ndarray
    R_T: np.ndarray
    t_std: np.ndarray

    @classmethod
    def load(cls, path):
        """Load a calibration from the .npz written by the calibration step."""
        d = np.load(path)
        return cls(d["K_inv"], d["R_T"], d["t_std"])

    @property
    def focal_px(self):
        """Focal length in pixels, as (fx, fy)."""
        return 1.0 / self.K_inv[0, 0], 1.0 / self.K_inv[1, 1]

    @property
    def principal_point(self):
        """Principal point in pixels, as (cx, cy)."""
        fx, fy = self.focal_px
        return -self.K_inv[0, 2] * fx, -self.K_inv[1, 2] * fy

    @property
    def camera_centre(self):
        """Camera centre in world coordinates."""
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
    world: np.ndarray
    fps: float
    speed_km_h: float
    drag: float

    @property
    def x(self):
        """Lateral position in metres, positive to the bowler's right."""
        return self.world[:, 0]

    @property
    def y(self):
        """Forward distance in metres, down the pitch."""
        return self.world[:, 1]

    @property
    def z(self):
        """Vertical position in metres, positive downward."""
        return self.world[:, 2]

    @property
    def relative_to_release(self):
        """The world track shifted so the first tracked point is the origin."""
        return self.world - self.world[0]

    @property
    def last_distance_m(self):
        """Forward distance of the last tracked point, in metres."""
        return float(self.world[-1, 1])


@dataclass
class SwingResult:
    """Both swing values for one delivery, with the fits they came from."""
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
        """Render the result as a short text block."""
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


class SideOnPhysicsEngine:
    """
    Swing analysis bound to one side-on calibration and one frame rate.

    The engine holds the calibration, frame rate, drag, fit windows and cube
    geometry. Delivery speed is passed to reconstruct_trajectory and analyse,
    since it varies delivery to delivery. Anything held on the engine can still
    be overridden per call where the signature allows it.
    """

    def __init__(self, calibration, fps,
                 drag=DRAG_COEFFICIENT,
                 baseline_points=BASELINE_SOURCE_POINT_COUNT,
                 projection_points=EXTRAPOLATION_SOURCE_POINT_COUNT,
                 target_distance_m=TARGET_DISTANCE_M,
                 ring_distances=RING_FORWARD_DISTANCES_M,
                 cube_side_m=CUBE_SIDE_M,
                 cube_centre_x_m=CUBE_CENTRE_X_M,
                 cube_centre_z_m=CUBE_CENTRE_Z_M,
                 calibration_path=None):
        """Bind a calibration and a frame rate to the engine."""
        self.calibration = calibration
        self.fps = float(fps)
        self.drag = float(drag)
        self.baseline_points = int(baseline_points)
        self.projection_points = int(projection_points)
        self.target_distance_m = float(target_distance_m)
        self.ring_distances = tuple(ring_distances)
        self.cube_side_m = float(cube_side_m)
        self.cube_centre_x_m = float(cube_centre_x_m)
        self.cube_centre_z_m = float(cube_centre_z_m)
        self.calibration_path = None if calibration_path is None else str(calibration_path)

    @classmethod
    def from_calibration_file(cls, path, fps, **kwargs):
        """Build an engine directly from a calibration .npz path."""
        kwargs.setdefault("calibration_path", path)
        return cls(Calibration.load(path), fps, **kwargs)

    @staticmethod
    def forward_distance(t, speed_m_s, drag=DRAG_COEFFICIENT):
        """Distance travelled at time t under quadratic drag."""
        if drag <= 0:
            return speed_m_s * t
        return math.log(1.0 + drag * speed_m_s * t) / drag

    def reconstruct_trajectory(self, points, speed_km_h, fps=None,
                               drag=None) -> Trajectory:
        """
        Pixel track -> 3D world track, at the given delivery speed in km/h.

        Each pixel gives only a ray from the camera. The speed-and-drag model
        supplies the forward distance that fixes where along that ray the ball
        sits. fps and drag fall back to the engine's values.
        """
        fps = self.fps if fps is None else float(fps)
        drag = self.drag if drag is None else float(drag)
        speed_km_h = float(speed_km_h)

        speed_m_s = speed_km_h / 3.6
        frame0 = points[0].frame
        row_y = self.calibration.R_T[1, :]

        frames, times, world = [], [], []
        for p in points:
            t = (p.frame - frame0) / fps
            y = self.forward_distance(t, speed_m_s, drag)
            ray = self.calibration.K_inv @ np.array([p.u, p.v, 1.0])
            scale = (y + row_y @ self.calibration.t_std) / (row_y @ ray)
            world.append(self.calibration.R_T @ (scale * ray - self.calibration.t_std))
            frames.append(p.frame)
            times.append(t)

        return Trajectory(np.asarray(frames), np.asarray(times), np.asarray(world),
                          float(fps), float(speed_km_h), float(drag))

    @staticmethod
    def ball_coordinates(trajectory: Trajectory, origin="cubes") -> np.ndarray:
        """
        Ball position in metres, as an (N, 3) array of X, Y, Z.

        origin="cubes"    world frame set by the clicked cube corners: Y = 0 at
                          the release ring, X positive to the bowler's right,
                          Z positive downward.
        origin="release"  same axes, shifted so the first tracked point is
                          (0, 0, 0).
        """
        if origin == "cubes":
            return trajectory.world.copy()
        if origin == "release":
            return trajectory.relative_to_release
        raise ValueError(f"origin must be 'cubes' or 'release', got {origin!r}")

    @staticmethod
    def _fit_line(y, x):
        """Least-squares line x(y), as (slope, intercept, residual RMS in cm)."""
        slope, intercept = np.polyfit(y, x, 1)
        rms_cm = float(np.std(x - (intercept + slope * y)) * 100.0)
        return float(slope), float(intercept), rms_cm

    def fit_swingless_baseline(self, trajectory: Trajectory, n_points=None):
        """Straight line through the first n_points - the no-swing path."""
        n_points = self.baseline_points if n_points is None else n_points
        n = min(n_points, len(trajectory.world))
        if n < 3:
            raise ValueError(f"need at least 3 points for a baseline, got {n}")
        return self._fit_line(trajectory.y[:n], trajectory.x[:n])

    def fit_projection(self, trajectory: Trajectory, n_points=None):
        """Straight line through the final n_points."""
        n_points = self.projection_points if n_points is None else n_points
        n = min(n_points, len(trajectory.world))
        if n < 3:
            raise ValueError(f"need at least 3 points for a projection, got {n}")
        return self._fit_line(trajectory.y[-n:], trajectory.x[-n:])

    @staticmethod
    def line_x(y, slope, intercept):
        """Lateral position of a fitted line at forward distance y."""
        return intercept + slope * np.asarray(y, dtype=float)

    @classmethod
    def _gap_cm(cls, y, b, p):
        """Signed gap in cm between the projection and the baseline at y."""
        return float((cls.line_x(y, p[0], p[1]) - cls.line_x(y, b[0], b[1])) * 100.0)

    def swing_at_last_tracked_point(self, trajectory: Trajectory,
                                    baseline_points=None,
                                    projection_points=None) -> float:
        """
        Swing in cm at the last tracked point: the gap between the swingless
        baseline and the projection line, both evaluated there.

        Negative is to the bowler's left. This is the measured value - it sits
        inside the tracked data and is stable to a few millimetres across
        sensible window sizes. It is the number to report.
        """
        b = self.fit_swingless_baseline(trajectory, baseline_points)
        p = self.fit_projection(trajectory, projection_points)
        return self._gap_cm(trajectory.last_distance_m, b, p)

    def swing_at_17m(self, trajectory: Trajectory,
                     baseline_points=None,
                     projection_points=None,
                     target_distance_m=None) -> float:
        """
        Swing in cm at TARGET_DISTANCE_M: the same gap, extended to 17 m.

        A straight projection assumes the lateral velocity at the end of
        tracking persists, but swing accelerates - so this is a LOWER BOUND. It
        also depends heavily on projection_points, since it extrapolates most of
        the way again past the tracked range. Quote it as a range, not a figure.
        """
        target_distance_m = (self.target_distance_m if target_distance_m is None
                             else target_distance_m)
        b = self.fit_swingless_baseline(trajectory, baseline_points)
        p = self.fit_projection(trajectory, projection_points)
        return self._gap_cm(target_distance_m, b, p)

    def analyse(self, points, speed_km_h, fps=None, drag=None,
                baseline_points=None, projection_points=None,
                target_distance_m=None) -> SwingResult:
        """
        Reconstruct the delivery and return both swing values plus diagnostics.

        This is the one call that produces everything: the trajectory, the two
        fitted lines and their residuals, the swing at the last tracked point
        and the projected swing at the target distance.
        """
        baseline_points = self.baseline_points if baseline_points is None else baseline_points
        projection_points = (self.projection_points if projection_points is None
                             else projection_points)
        target_distance_m = (self.target_distance_m if target_distance_m is None
                             else target_distance_m)

        traj = self.reconstruct_trajectory(points, speed_km_h, fps, drag)

        b_slope, b_int, b_rms = self.fit_swingless_baseline(traj, baseline_points)
        p_slope, p_int, p_rms = self.fit_projection(traj, projection_points)
        b, p = (b_slope, b_int), (p_slope, p_int)

        y_last = traj.last_distance_m
        offset_cm = float((traj.x[-1] - self.line_x(y_last, p_slope, p_int)) * 100.0)

        return SwingResult(
            swing_at_last_point_cm=self._gap_cm(y_last, b, p),
            swing_at_17m_cm=self._gap_cm(target_distance_m, b, p),
            baseline_slope=b_slope, baseline_intercept=b_int, baseline_residual_cm=b_rms,
            projection_slope=p_slope, projection_intercept=p_int,
            projection_residual_cm=p_rms,
            last_point_offset_cm=offset_cm,
            last_point_distance_m=y_last,
            target_distance_m=float(target_distance_m),
            baseline_points=min(baseline_points, len(traj.world)),
            extrapolation_points=min(projection_points, len(traj.world)),
            trajectory=traj,
        )

    def ring_corners(self, forward_distance_m, side=None,
                     centre_x=None, centre_z=None):
        """Four corners of one calibration ring, in world coordinates."""
        side = self.cube_side_m if side is None else side
        centre_x = self.cube_centre_x_m if centre_x is None else centre_x
        centre_z = self.cube_centre_z_m if centre_z is None else centre_z
        h = side / 2.0
        return np.array([
            [centre_x - h, forward_distance_m, centre_z - h],
            [centre_x + h, forward_distance_m, centre_z - h],
            [centre_x + h, forward_distance_m, centre_z + h],
            [centre_x - h, forward_distance_m, centre_z + h],
        ])

    def cube_edges(self, distances=None, **kw):
        """Line segments for the cube wireframe: each ring, plus the rails between."""
        distances = self.ring_distances if distances is None else distances
        rings = [self.ring_corners(d, **kw) for d in distances]
        segments = []
        for r in rings:
            for i in range(4):
                segments.append((r[i], r[(i + 1) % 4]))
        for a, b in zip(rings, rings[1:]):
            for i in range(4):
                segments.append((a[i], b[i]))
        return segments

    @classmethod
    def _plain(cls, value):
        """Convert numpy scalars and arrays to plain Python types, recursively."""
        if isinstance(value, dict):
            return {k: cls._plain(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [cls._plain(v) for v in value]
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    def _analysis_mapping(self, result: SwingResult, points_csv_name) -> dict:
        """Assemble everything that is not per-point data into one plain mapping."""
        traj = result.trajectory
        cal = self.calibration
        return self._plain({
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "points_csv": points_csv_name,
            "delivery": {
                "point_count": len(traj.frames),
                "first_frame": traj.frames[0],
                "last_frame": traj.frames[-1],
                "fps": traj.fps,
                "speed_km_h": traj.speed_km_h,
                "speed_m_s": traj.speed_km_h / 3.6,
                "duration_s": traj.times_s[-1],
                "drag_coefficient": traj.drag,
            },
            "swing": {
                "at_last_tracked_point_cm": result.swing_at_last_point_cm,
                "at_target_distance_cm": result.swing_at_17m_cm,
                "last_point_distance_m": result.last_point_distance_m,
                "target_distance_m": result.target_distance_m,
                "last_point_offset_from_projection_cm": result.last_point_offset_cm,
            },
            "baseline_fit": {
                "source_point_count": result.baseline_points,
                "slope": result.baseline_slope,
                "intercept": result.baseline_intercept,
                "residual_rms_cm": result.baseline_residual_cm,
            },
            "projection_fit": {
                "source_point_count": result.extrapolation_points,
                "slope": result.projection_slope,
                "intercept": result.projection_intercept,
                "residual_rms_cm": result.projection_residual_cm,
            },
            "calibration": {
                "path": self.calibration_path,
                "focal_px": list(cal.focal_px),
                "principal_point_px": list(cal.principal_point),
                "camera_centre_m": cal.camera_centre,
                "implied_frame_size_px": list(cal.implied_frame_size),
                "K_inv": cal.K_inv,
                "R_T": cal.R_T,
                "t_std": cal.t_std,
            },
            "engine_config": {
                "fps": self.fps,
                "drag_coefficient": self.drag,
                "baseline_source_point_count": self.baseline_points,
                "extrapolation_source_point_count": self.projection_points,
                "target_distance_m": self.target_distance_m,
                "ring_forward_distances_m": self.ring_distances,
                "cube_side_m": self.cube_side_m,
                "cube_centre_x_m": self.cube_centre_x_m,
                "cube_centre_z_m": self.cube_centre_z_m,
            },
        })

    def save_data_to_files(self, result: SwingResult, save_directory,
                           points=None, prefix="") -> dict:
        """
        Write the whole analysis to save_directory, creating it if needed.

        Two files are written. The CSV carries one row per tracked point: frame,
        time, the source pixel coordinates if points is supplied, the world
        coordinates X, Y, Z and the release-relative dX, dY, dZ. The YAML
        carries everything else - delivery, swing values, both line fits, the
        calibration and the engine settings.

        prefix, if given, is prepended to both file names so several deliveries
        can share one directory. Returns a dict of the paths written.
        """
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_" if prefix else ""
        csv_path = directory / f"{stem}{POINTS_CSV_NAME}"
        yaml_path = directory / f"{stem}{ANALYSIS_YAML_NAME}"

        traj = result.trajectory
        cubes = self.ball_coordinates(traj, origin="cubes")
        release = self.ball_coordinates(traj, origin="release")

        header = ["frame", "time_s"]
        if points is not None:
            header += ["u_px", "v_px"]
        header += ["x_m", "y_m", "z_m", "dx_m", "dy_m", "dz_m"]

        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for i in range(len(cubes)):
                row = [int(traj.frames[i]), f"{traj.times_s[i]:.6f}"]
                if points is not None:
                    row += [f"{points[i].u:g}", f"{points[i].v:g}"]
                row += [f"{v:.6f}" for v in cubes[i]]
                row += [f"{v:.6f}" for v in release[i]]
                writer.writerow(row)

        with open(yaml_path, "w") as handle:
            yaml.dump(self._analysis_mapping(result, csv_path.name), handle,
                      Dumper=_YamlDumper, sort_keys=False, width=4096,
                      allow_unicode=True)

        return {"points_csv": str(csv_path), "analysis_yaml": str(yaml_path)}

    def plot_swing(self, result: SwingResult, save_path="swing.png", show=False):
        """Trajectory, baseline, projection, and both swing values as measured gaps."""
        import matplotlib.pyplot as plt

        traj = result.trajectory
        target = result.target_distance_m
        n_fit, n_proj = result.baseline_points, result.extrapolation_points
        b = (result.baseline_slope, result.baseline_intercept)
        p = (result.projection_slope, result.projection_intercept)

        y_grid = np.linspace(0.0, target * 1.03, 300)
        fig, ax = plt.subplots(figsize=(13, 7))

        ax.plot(y_grid, self.line_x(y_grid, *b) * 100, '-', color=_INK, lw=2.2, zorder=4,
                label=f"swingless baseline (first {n_fit} points)")
        ax.plot(y_grid, self.line_x(y_grid, *p) * 100, '--', color=_ORANGE, lw=2.6, zorder=3,
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
            xb, xp = float(self.line_x(yv, *b)) * 100, float(self.line_x(yv, *p)) * 100
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

    def plot_trajectory_3d(self, result: SwingResult, show=True, save_path=None,
                           distances=None, show_baseline=True, elev=18, azim=-70):
        """
        Full 3D trajectory with the calibration cubes drawn as reference.

        With show=True the figure opens in an interactive window: drag with the
        left mouse button to rotate, scroll or right-drag to zoom. That needs a
        GUI backend - you have PyQt5, so matplotlib will pick Qt5Agg
        automatically. Headless, set MPLBACKEND=Agg and pass show=False with a
        save_path.

        Z is negated for display so that up is up on screen.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        distances = self.ring_distances if distances is None else distances
        traj = result.trajectory
        n_fit, n_proj = result.baseline_points, result.extrapolation_points

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        for a, b in self.cube_edges(distances):
            ax.plot([a[1], b[1]], [a[0], b[0]], [-a[2], -b[2]],
                    color=_GREY, lw=1.0, alpha=0.55, zorder=1)
        for d in distances:
            r = self.ring_corners(d)
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
            xb = self.line_x(yg, result.baseline_slope, result.baseline_intercept)
            zc = np.polyfit(traj.y, traj.z, 2)
            ax.plot(yg, xb, -np.polyval(zc, yg), '--', color=_INK, lw=1.4,
                    alpha=0.85, label="swingless baseline")

        ax.set_xlabel("Y  forward down the pitch (m)")
        ax.set_ylabel("X  lateral, + is bowler's right (m)")
        ax.set_zlabel("height (m)")

        ymax = max(float(traj.y.max()), max(distances))
        half = self.cube_side_m / 2.0 + 0.2
        ax.set_xlim(0, ymax)
        ax.set_ylim(self.cube_centre_x_m - half, self.cube_centre_x_m + half)
        ax.set_zlim(-self.cube_centre_z_m - half, -self.cube_centre_z_m + half)
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