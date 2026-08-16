import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from enum import Enum
import yaml
from pathlib import Path

from helpers import TopDownBallData, SideOnBallData, SideOnBallData, Coord
from automations import TopDownBallDataPoint, SideOnBallDataPoint, SideOnBallData, Coord

SAVE_DIRECTORY = "output_folder/demo_delivery"
BALL_METRE_DIAMETER = 0.072 # Ball diameter in metres
GRAVITY = 9.81
FRAMES_FIT_SEAMLESS = 15 # Number of early frames used to fit the swingless trajectory's initial velocity

# Lateral drag coefficient, per metre - fitted from a known-swingless reference
# delivery. Same physical idea as the forward drag_coefficient, applied to the
# lateral component instead. Note: validated to fit well within the reference
# ball's FIRST FRAMES_FIT_SEAMLESS frames, but the reference ball's own later
# (held-out) frames show non-monotonic behaviour this model can't represent -
# see calculate_relative_swing_bestfit for an alternative that doesn't assume
# this functional form at all.
LATERAL_DRAG_COEFFICIENT = 24.377

# ---------------------------------------------------------------------------
# Known swingless reference delivery - a real, confirmed-straight ball, used
# by calculate_relative_swing / calculate_relative_swing_bestfit to measure
# relative swing on future deliveries.
# ---------------------------------------------------------------------------
REFERENCE_FPS = 119.88
REFERENCE_INITIAL_SPEED_KMH = 107
REFERENCE_DRAG_COEFFICIENT = 0.0092
REFERENCE_CALIBRATION_PATH = "reference_swingless_calibration.npz"

REFERENCE_SIDE_ON_DATA = [
    (428,1710,721),(429,1741,728),(430,1765,734),(431,1789,740),(432,1810,746),
    (433,1829,750),(434,1847,755),(435,1862,759),(436,1876,763),(437,1890,767),
    (438,1902,771),(439,1914,775),(440,1924,779),(441,1934,782),(442,1942,785),
    (443,1950,789),(444,1958,793),(445,1965,797),(446,1972,800),(447,1979,803),
    (448,1985,807),(449,1992,810),(450,1997,813),(451,2002,816),(452,2007,819),
    (453,2011,822),(454,2016,825),(455,2019,829),(456,2024,832),(457,2028,836),
    (458,2032,839),(459,2034,841),(460,2039,844),(461,2041,847),(462,2045,850),
    (463,2048,852),(464,2052,855),
]

class SelectionType(Enum):
    MIN = 1
    MEAN = 2
    MAX = 3

class TopDownPhysicsEngine:
    def calculate_velocity(self, top_down_points: list[TopDownBallDataPoint], fps: float, type: SelectionType) -> float:
        '''
        Turns list of top down points to velocities (km/h).
        Uses average velocity of all points.
        When using manual tracking:
        - Some points will have top-left and bottom-right but most wont.
        - Only one point will have seam start and seam end.
        '''
        velocities = []
        ball_pixel_diameter = None
        for i in range(1, len(top_down_points)):
            prev_point = top_down_points[i - 1]
            curr_point = top_down_points[i]

            if prev_point.data.top_left is not None and prev_point.data.bottom_right is not None:
                ball_pixel_diameter = prev_point.data.top_left.distance_to(prev_point.data.bottom_right)

            if not ball_pixel_diameter:
                raise ValueError("First chosen top down frame does not have corresponding diameter data.")

            if not curr_point.data.centre or not prev_point.data.centre:
                continue # Skip if the either frame doesn't have any location data

            pixel_distance = prev_point.data.centre.distance_to(curr_point.data.centre)

            time_diff = (curr_point.frame_number - prev_point.frame_number) / fps
            
            metre_distance = (pixel_distance / ball_pixel_diameter) * BALL_METRE_DIAMETER

            metre_per_second = metre_distance / time_diff if time_diff > 0 else 0
            kmh = metre_per_second * 3.6
            velocities.append(kmh)

        if len(velocities) == 0:
            raise ValueError("No valid velocity data could be calculated from the provided top down points.")
        
        if type == SelectionType.MIN:
            return min(velocities)
        if type == SelectionType.MAX:
            return max(velocities)
        # SelectionType.MEAN
        return sum(velocities) / len(velocities)

    def calculate_seam_angle(self, top_down_points: list[TopDownBallDataPoint]) -> float | None:
        '''
        Turns list of top down points to seam angles (degrees).
        Calculates seam angle relative to the ball's direction.
        Uses first seam angle found.
        Returns none if no seam angles are recorded.
        '''
        for point_idx in range(len(top_down_points) - 1):
            curr_point = top_down_points[point_idx]
            next_point = top_down_points[point_idx + 1]

            if curr_point.data.centre and next_point.data.centre:
                ball_direction = math.degrees(math.atan2(next_point.data.centre.y - curr_point.data.centre.y, next_point.data.centre.x - curr_point.data.centre.x))

            if curr_point.data.seam_angle:
                print(f"Ball direction is {ball_direction}")
                print(f"Raw seam angle is {curr_point.data.seam_angle}")

                return curr_point.data.seam_angle - ball_direction
        return None # If no seam angle on any points

    def save_top_down_analysis(save_directory, file_name, velocity, seam_angle, fps, point_count):
        """Write the top-down velocity and seam angle to a YAML file in save_directory."""
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        full_file_name = file_name + ".yaml"
        path = directory / full_file_name

        with open(path, "w") as handle:
            yaml.safe_dump({
                "velocity_km_h": float(velocity),
                "seam_angle_deg": None if seam_angle is None else float(seam_angle),
                "fps": float(fps),
                "point_count": point_count,
            }, handle, sort_keys=False)

        return str(path)


class SideOnPhysicsEngine:

    def build_cube_ring_distances(
        self,
        num_cubes: int,
        cube_edge_m: float = 3.0,
        release_to_first_ring_m: float = 0.0,
    ) -> list[float]:
        '''
        Forward distances (from release) for each ring boundary in a chain of
        num_cubes attached cubes.
        '''
        num_rings = num_cubes + 1
        return [release_to_first_ring_m + i * cube_edge_m for i in range(num_rings)]

    def forward_distance_at_time(self, elapsed_s: float, initial_speed_m_s: float, drag_coefficient: float = 0.0092) -> float:
        '''
        Forward distance travelled at time elapsed_s, given quadratic air-resistance
        deceleration (dv/dt = -drag_coefficient * v^2).
        '''
        if drag_coefficient <= 0:
            return initial_speed_m_s * elapsed_s
        return math.log(1 + drag_coefficient * initial_speed_m_s * elapsed_s) / drag_coefficient

    def lateral_distance_at_time(self, elapsed_s: float, initial_lateral_speed_m_s: float, lateral_drag_coefficient: float = LATERAL_DRAG_COEFFICIENT) -> float:
        '''
        Lateral distance travelled at time elapsed_s, using the same quadratic
        drag physics as forward_distance_at_time, applied to the lateral
        (release-angle) velocity component.
        '''
        if lateral_drag_coefficient <= 0 or abs(initial_lateral_speed_m_s) < 1e-9:
            return initial_lateral_speed_m_s * elapsed_s
        sign = 1.0 if initial_lateral_speed_m_s >= 0 else -1.0
        magnitude = abs(initial_lateral_speed_m_s)
        return sign * math.log(1 + lateral_drag_coefficient * magnitude * elapsed_s) / lateral_drag_coefficient

    def _fit_initial_lateral_velocity(self, ts: list[float], xs: list[float], lateral_drag_coefficient: float = LATERAL_DRAG_COEFFICIENT) -> float:
        '''
        Golden-section search for the initial lateral velocity that minimizes
        squared error against the ACTUAL lateral_distance_at_time formula -
        not a linear approximation of it.
        '''
        def sse(vx0):
            return sum((self.lateral_distance_at_time(t, vx0, lateral_drag_coefficient) - x) ** 2 for t, x in zip(ts, xs))
        lo, hi = -50.0, 50.0
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        for _ in range(100):
            if sse(c) < sse(d):
                hi = d
            else:
                lo = c
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
        return (lo + hi) / 2

    def pixel_to_xyz(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
    ) -> list[list[float]]:
        '''
        Converts tracked side-on pixel points into [x, y, z] coordinates in
        metres, relative to the ball's position at release.
        '''
        if not side_on_points:
            return []

        initial_speed_m_s = initial_speed_km_h / 3.6
        seconds_per_frame = 1 / fps
        release_frame = side_on_points[0].frame_number
        row_y = R_T[1, :]

        absolute_coords = []
        for point in side_on_points:
            elapsed_s = (point.frame_number - release_frame) * seconds_per_frame
            target_y = self.forward_distance_at_time(elapsed_s, initial_speed_m_s, drag_coefficient)

            d = K_inv @ np.array([point.data.centre.x, point.data.centre.y, 1.0])
            s = (target_y + row_y @ t_std) / (row_y @ d)
            P = R_T @ (s * d - t_std)
            absolute_coords.append(P)

        origin = absolute_coords[0]
        return [[p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]] for p in absolute_coords]

    def get_swinless_trajectory(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
    ) -> list[list[float]]:
        '''
        Extrapolates a swingless trajectory using the first FRAMES_FIT_SEAMLESS
        tracked points, fitting initial lateral velocity against the actual
        drag formula (not a linear approximation of it).
        '''
        if len(side_on_points) < 2:
            raise IndexError(f"Not enough side on points to calculate initial trajectory. Need 2, got {len(side_on_points)}")

        fit_points = side_on_points[:min(FRAMES_FIT_SEAMLESS, len(side_on_points))]
        real_xyz = self.pixel_to_xyz(fit_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient)

        seconds_per_frame = 1 / fps
        release_frame = side_on_points[0].frame_number
        ts = [(p.frame_number - release_frame) * seconds_per_frame for p in fit_points]
        xs = [xyz[0] for xyz in real_xyz]
        zs = [xyz[2] for xyz in real_xyz]

        vx_m_s = self._fit_initial_lateral_velocity(ts, xs)
        vz_m_s = sum(t * (z - 0.5 * GRAVITY * t ** 2) for t, z in zip(ts, zs)) / sum(t * t for t in ts)

        initial_speed_m_s = initial_speed_km_h / 3.6
        trajectory: list[list[float]] = []
        frame_offset = 0
        while True:
            elapsed_s = frame_offset * seconds_per_frame
            y_m = self.forward_distance_at_time(elapsed_s, initial_speed_m_s, drag_coefficient)
            if y_m > max_forward_distance_m:
                break

            x_m = self.lateral_distance_at_time(elapsed_s, vx_m_s)
            z_m = vz_m_s * elapsed_s + 0.5 * GRAVITY * elapsed_s ** 2

            trajectory.append([x_m, y_m, z_m])
            frame_offset += 1

        return trajectory

    def get_extrapolated_trajectory(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
        linear_fit_points: int = 10,
    ) -> tuple[list[list[float]], list[list[float]]]:
        '''
        Fits a quadratic curve of best fit (all tracked points) and a linear
        curve of best fit (last linear_fit_points tracked points) directly to
        the actual lateral position X(t), then extrapolates each to
        max_forward_distance_m. Returns absolute [y_m, x_cm] positions.
        '''
        real_xyz = self.pixel_to_xyz(side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient)
        seconds_per_frame = 1 / fps
        release_frame = side_on_points[0].frame_number
        all_ts = [(p.frame_number - release_frame) * seconds_per_frame for p in side_on_points]
        all_xs_cm = [xyz[0] * 100 for xyz in real_xyz]

        a_quad, b_quad, c_quad = np.polyfit(all_ts, all_xs_cm, 2)

        n = min(linear_fit_points, len(side_on_points))
        m_lin, b_lin = np.polyfit(all_ts[-n:], all_xs_cm[-n:], 1)

        initial_speed_m_s = initial_speed_km_h / 3.6
        quadratic_trajectory: list[list[float]] = []
        linear_trajectory: list[list[float]] = []
        frame_offset = 0
        while True:
            elapsed_s = frame_offset * seconds_per_frame
            y_m = self.forward_distance_at_time(elapsed_s, initial_speed_m_s, drag_coefficient)
            if y_m > max_forward_distance_m:
                break
            quadratic_trajectory.append([y_m, a_quad * elapsed_s ** 2 + b_quad * elapsed_s + c_quad])
            linear_trajectory.append([y_m, m_lin * elapsed_s + b_lin])
            frame_offset += 1

        return quadratic_trajectory, linear_trajectory

    def calculate_relative_swing(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
    ) -> list[list[float]]:
        '''
        Relative swing using the physics-based swingless model
        (get_swinless_trajectory) for both this delivery and the reference.
        NOTE: this model can only ever show increasing lateral displacement,
        never a reversal - the reference ball's own later frames show it
        doesn't hold that shape. calculate_relative_swing_bestfit doesn't
        have this limitation.
        '''
        reference_points = self._build_reference_points()
        ref_K_inv, ref_R_T, ref_t_std = self._load_reference_calibration()

        reference_trajectory = np.array(self.get_swinless_trajectory(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH,
            ref_K_inv, ref_R_T, ref_t_std, REFERENCE_DRAG_COEFFICIENT, max_forward_distance_m,
        ))
        delivery_trajectory = np.array(self.get_swinless_trajectory(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std,
            drag_coefficient, max_forward_distance_m,
        ))

        reference_x_interp = np.interp(
            delivery_trajectory[:, 1], reference_trajectory[:, 1], reference_trajectory[:, 0]
        )
        swing_cm = (delivery_trajectory[:, 0] - reference_x_interp) * 100

        return [[float(y), float(s)] for y, s in zip(delivery_trajectory[:, 1], swing_cm)]

    def calculate_relative_swing_bestfit(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
        linear_fit_points: int = 10,
    ) -> tuple[list[list[float]], list[list[float]]]:
        '''
        Relative swing using best-fit curves (get_extrapolated_trajectory)
        for both this delivery and the reference, compared directly - doesn't
        assume any particular functional shape, so isn't limited by the
        physics model's inability to represent non-monotonic behaviour.
        This is the recommended method.

        Returns (quadratic_swing, linear_swing), each [y_m, swing_cm] pairs.
        '''
        reference_points = self._build_reference_points()
        ref_K_inv, ref_R_T, ref_t_std = self._load_reference_calibration()

        ref_quad, ref_lin = self.get_extrapolated_trajectory(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH,
            ref_K_inv, ref_R_T, ref_t_std, REFERENCE_DRAG_COEFFICIENT,
            max_forward_distance_m, linear_fit_points,
        )
        delivery_quad, delivery_lin = self.get_extrapolated_trajectory(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std,
            drag_coefficient, max_forward_distance_m, linear_fit_points,
        )

        ref_quad_arr, ref_lin_arr = np.array(ref_quad), np.array(ref_lin)
        delivery_quad_arr, delivery_lin_arr = np.array(delivery_quad), np.array(delivery_lin)

        ref_quad_interp = np.interp(delivery_quad_arr[:, 0], ref_quad_arr[:, 0], ref_quad_arr[:, 1])
        ref_lin_interp = np.interp(delivery_lin_arr[:, 0], ref_lin_arr[:, 0], ref_lin_arr[:, 1])

        quadratic_swing = [
            [float(y), float(dx - rx)]
            for y, dx, rx in zip(delivery_quad_arr[:, 0], delivery_quad_arr[:, 1], ref_quad_interp)
        ]
        linear_swing = [
            [float(y), float(dx - rx)]
            for y, dx, rx in zip(delivery_lin_arr[:, 0], delivery_lin_arr[:, 1], ref_lin_interp)
        ]

        return quadratic_swing, linear_swing

    def plot_relative_swing_analysis(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
        save_path: str = "relative_swing_analysis.png",
    ) -> str:
        '''
        Plots the physics-model-based comparison (calculate_relative_swing):
        both deliveries' real points + get_swinless_trajectory extrapolation,
        and the resulting swing curve.
        '''
        actual = np.array(self.pixel_to_xyz(side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient))
        delivery_extrap = np.array(self.get_swinless_trajectory(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient, max_forward_distance_m
        ))

        reference_points = self._build_reference_points()
        ref_K_inv, ref_R_T, ref_t_std = self._load_reference_calibration()
        reference_actual = np.array(self.pixel_to_xyz(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH, ref_K_inv, ref_R_T, ref_t_std, REFERENCE_DRAG_COEFFICIENT
        ))
        reference_extrap = np.array(self.get_swinless_trajectory(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH, ref_K_inv, ref_R_T, ref_t_std,
            REFERENCE_DRAG_COEFFICIENT, max_forward_distance_m,
        ))

        swing = np.array(self.calculate_relative_swing(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient, max_forward_distance_m
        ))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        ax = axes[0]
        ax.plot(delivery_extrap[:, 1], delivery_extrap[:, 0] * 100, '--', color='#2563eb', linewidth=2, label="Delivery - extrapolated (physics model)")
        ax.plot(actual[:, 1], actual[:, 0] * 100, 'o', color='#2563eb', markersize=4, label="Delivery - real tracked points")
        ax.plot(reference_extrap[:, 1], reference_extrap[:, 0] * 100, '--', color='#16a34a', linewidth=2, label="Swingless reference - extrapolated (physics model)")
        ax.plot(reference_actual[:, 1], reference_actual[:, 0] * 100, 'o', color='#16a34a', markersize=4, label="Swingless reference - real tracked points")
        ax.set_xlabel("Forward distance, Y (m)")
        ax.set_ylabel("Lateral position, X (cm)")
        ax.set_title("Physics model: real data + extrapolation to 17m")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(swing[:, 0], swing[:, 1], color='#dc2626', linewidth=2.5)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.fill_between(swing[:, 0], swing[:, 1], 0, alpha=0.15, color='#dc2626')
        ax.set_xlabel("Forward distance, Y (m)")
        ax.set_ylabel("Relative swing (cm)")
        ax.set_title("Swing at each point (delivery minus reference)")
        ax.grid(alpha=0.3)

        plt.suptitle("Relative swing analysis - physics model")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    def plot_relative_swing_bestfit_analysis(
        self,
        side_on_points: list[SideOnBallDataPoint],
        fps: float,
        initial_speed_km_h: float,
        K_inv: np.ndarray,
        R_T: np.ndarray,
        t_std: np.ndarray,
        drag_coefficient: float = 0.0092,
        max_forward_distance_m: float = 17.0,
        linear_fit_points: int = 10,
        save_path: str = "relative_swing_bestfit_analysis.png",
    ) -> str:
        '''
        Plots the best-fit comparison (calculate_relative_swing_bestfit):
        both deliveries real points + quadratic/linear best-fit extrapolations,
        and the resulting quadratic-based and linear-based swing curves.
        '''
        actual = np.array(self.pixel_to_xyz(side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient))
        delivery_quad, delivery_lin = self.get_extrapolated_trajectory(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std, drag_coefficient, max_forward_distance_m, linear_fit_points
        )
        delivery_quad, delivery_lin = np.array(delivery_quad), np.array(delivery_lin)

        reference_points = self._build_reference_points()
        ref_K_inv, ref_R_T, ref_t_std = self._load_reference_calibration()
        reference_actual = np.array(self.pixel_to_xyz(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH, ref_K_inv, ref_R_T, ref_t_std, REFERENCE_DRAG_COEFFICIENT
        ))
        ref_quad, ref_lin = self.get_extrapolated_trajectory(
            reference_points, REFERENCE_FPS, REFERENCE_INITIAL_SPEED_KMH, ref_K_inv, ref_R_T, ref_t_std,
            REFERENCE_DRAG_COEFFICIENT, max_forward_distance_m, linear_fit_points,
        )
        ref_quad, ref_lin = np.array(ref_quad), np.array(ref_lin)

        quad_swing, lin_swing = self.calculate_relative_swing_bestfit(
            side_on_points, fps, initial_speed_km_h, K_inv, R_T, t_std,
            drag_coefficient, max_forward_distance_m, linear_fit_points,
        )
        quad_swing, lin_swing = np.array(quad_swing), np.array(lin_swing)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(delivery_quad[:, 0], delivery_quad[:, 1], '--', color='#2563eb', linewidth=2, label="Delivery - quadratic best-fit")
        ax.plot(actual[:, 1], actual[:, 0] * 100, 'o', color='#2563eb', markersize=4, label="Delivery - real points")
        ax.plot(ref_quad[:, 0], ref_quad[:, 1], '--', color='#16a34a', linewidth=2, label="Reference - quadratic best-fit")
        ax.plot(reference_actual[:, 1], reference_actual[:, 0] * 100, 'o', color='#16a34a', markersize=4, label="Reference - real points")
        ax.set_xlabel("Forward distance, Y (m)"); ax.set_ylabel("Lateral position, X (cm)")
        ax.set_title("Quadratic best-fit trajectories"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(delivery_lin[:, 0], delivery_lin[:, 1], ':', color='#2563eb', linewidth=2, label="Delivery - linear best-fit (last N pts)")
        ax.plot(actual[:, 1], actual[:, 0] * 100, 'o', color='#2563eb', markersize=4, label="Delivery - real points")
        ax.plot(ref_lin[:, 0], ref_lin[:, 1], ':', color='#16a34a', linewidth=2, label="Reference - linear best-fit (last N pts)")
        ax.plot(reference_actual[:, 1], reference_actual[:, 0] * 100, 'o', color='#16a34a', markersize=4, label="Reference - real points")
        ax.set_xlabel("Forward distance, Y (m)"); ax.set_ylabel("Lateral position, X (cm)")
        ax.set_title("Linear best-fit trajectories"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(quad_swing[:, 0], quad_swing[:, 1], color='#dc2626', linewidth=2.5)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.fill_between(quad_swing[:, 0], quad_swing[:, 1], 0, alpha=0.15, color='#dc2626')
        ax.set_xlabel("Forward distance, Y (m)"); ax.set_ylabel("Swing (cm)")
        ax.set_title(f"Quadratic-based swing (17m: {quad_swing[-1,1]:.1f}cm)"); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(lin_swing[:, 0], lin_swing[:, 1], color='#7c3aed', linewidth=2.5)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.fill_between(lin_swing[:, 0], lin_swing[:, 1], 0, alpha=0.15, color='#7c3aed')
        ax.set_xlabel("Forward distance, Y (m)"); ax.set_ylabel("Swing (cm)")
        ax.set_title(f"Linear-based swing (17m: {lin_swing[-1,1]:.1f}cm)"); ax.grid(alpha=0.3)

        plt.suptitle("Relative swing analysis - best-fit method (recommended)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def _build_reference_points() -> list[SideOnBallDataPoint]:
        return [
            SideOnBallDataPoint(
                frame_number=fn,
                data=SideOnBallData(
                    top_left=Coord(x=x - 15, y=z - 15),
                    bottom_right=Coord(x=x + 15, y=z + 15),
                    centre=Coord(x=x, y=z),
                ),
            )
            for fn, x, z in REFERENCE_SIDE_ON_DATA
        ]

    @staticmethod
    def _load_reference_calibration() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(REFERENCE_CALIBRATION_PATH)
        return data["K_inv"], data["R_T"], data["t_std"]

