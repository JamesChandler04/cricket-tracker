import math
import numpy as np
import cv2

from automations import TopDownBallDataPoint, SideOnBallDataPoint, Coord


BALL_METRE_DIAMETER = 0.072 # Ball diameter in metres
GRAVITY = 9.81
FRAMES_FIT_SEAMLESS = 15 # Number of early frames used to fit the swingless trajectory's initial velocity

class TopDownPhysicsEngine:
    def calculate_velocity(self, top_down_points: list[TopDownBallDataPoint], fps: float) -> float:
        '''
        Turns list of top down points to velocities (km/h).
        Uses average velocity of all points.
        '''
        velocities = []
        for i in range(1, len(top_down_points)):
            prev_point = top_down_points[i - 1]
            curr_point = top_down_points[i]

            pixel_distance = ((curr_point.data.centre.x - prev_point.data.centre.x) ** 2 + (curr_point.data.centre.y - prev_point.data.centre.y) ** 2) ** 0.5

            time_diff = (curr_point.frame_number - prev_point.frame_number) / fps

            ball_pixel_diameter = ((prev_point.data.bottom_right.x - prev_point.data.top_left.x) + (prev_point.data.bottom_right.y - prev_point.data.top_left.y)) / 2

            metre_distance = (pixel_distance / ball_pixel_diameter) * BALL_METRE_DIAMETER

            metre_per_second = metre_distance / time_diff if time_diff > 0 else 0
            kmh = metre_per_second * 3.6
            velocities.append(kmh)
        ave_velocity = sum(velocities) / len(velocities) if velocities else 0
        return ave_velocity

    def calculate_seam_angle(self, top_down_points: list[TopDownBallDataPoint]) -> float:
        '''
        Turns list of top down points to seam angles (degrees).
        Uses first seam angle found.
        '''
        for point in top_down_points:
            if point.data.seam_angle != -1:
                return point.data.seam_angle
        return -1 # If no seam angle on any points

class SideOnPhysicsEngine:

    # ---------------------------------------------------------------------------
    # STEP 1: Build the known 3D forward-distances for each ring boundary in the
    # chained cube tunnel (front face, every joint between cubes, and the back
    # face) - one call, using the cube edge length and how far release sits from
    # the front of the first cube.
    # ---------------------------------------------------------------------------
    def build_cube_ring_distances(
        self,
        num_cubes: int,
        cube_edge_m: float = 3.0,
        release_to_first_ring_m: float = 0.1,
    ) -> list[float]:
        '''
        Forward distances (from release) for each ring boundary in a chain of
        num_cubes attached cubes - front face, every internal joint, and the
        back face (num_cubes + 1 rings total).
        '''
        num_rings = num_cubes + 1
        return [release_to_first_ring_m + i * cube_edge_m for i in range(num_rings)]

    # ---------------------------------------------------------------------------
    # STEP 2: Forward distance travelled at a given time, using the ball's
    # initial speed (from the top-down camera's first few frames) and quadratic
    # air-resistance deceleration - since the ball's true speed keeps dropping
    # throughout the flight, not staying constant.
    # ---------------------------------------------------------------------------
    def forward_distance_at_time(self, elapsed_s: float, initial_speed_m_s: float, drag_coefficient: float = 0.0092) -> float:
        '''
        Forward distance travelled at time elapsed_s, given quadratic air-resistance
        deceleration (dv/dt = -drag_coefficient * v^2).

        drag_coefficient (units: 1/m) combines air density, drag coefficient Cd,
        cross-sectional area, and ball mass:
            drag_coefficient = 0.5 * air_density * Cd * cross_sectional_area / ball_mass
        Default (0.0092) is derived from wind-tunnel-measured Cd~0.59 for a new
        cricket ball; use ~0.010-0.011 for a well-worn ball instead.
        '''
        if drag_coefficient <= 0:
            return initial_speed_m_s * elapsed_s  # no drag - constant velocity fallback
        return math.log(1 + drag_coefficient * initial_speed_m_s * elapsed_s) / drag_coefficient

    # ---------------------------------------------------------------------------
    # STEP 3: The main conversion - tracked side-on pixel points -> [x, y, z]
    # coordinates in metres, relative to the ball's position at release.
    # ---------------------------------------------------------------------------
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
        metres, relative to the ball's position at release (side_on_points[0]).

        x: lateral position (swing)
        y: forward distance down the pitch
        z: vertical position (down = positive)

        initial_speed_km_h: speed measured from the top-down camera's first few
        frames. y for every later frame is extrapolated from that starting speed
        using quadratic drag deceleration (forward_distance_at_time), not a flat
        constant-speed assumption.

        K_inv, R_T, t_std: from resection_camera (see calibrate.py) - calibrated
        once per fixed camera setup, reused for every delivery.
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

    # ---------------------------------------------------------------------------
    # STEP 4: Swingless trajectory - the straight-line path the ball would have
    # taken if it kept going in the exact direction it left the hand in, with no
    # additional swing.
    #
    # Initial velocity is fit via least squares across FRAMES_FIT_SEAMLESS early
    # frames rather than a raw 2-point finite difference, since frame-to-frame
    # pixel movement can be sub-pixel at high frame rates - a 2-point estimate
    # ends up dominated by detection/rounding noise rather than the ball's true
    # direction. More frames -> more resistant to that noise, but too many
    # starts folding real curvature (drag/gravity/early swing) into what should
    # be a straight-line fit - see note on FRAMES_FIT_SEAMLESS below.
    # ---------------------------------------------------------------------------
    def get_swingless_trajectory(
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
        tracked points to establish initial lateral/vertical velocity, then
        projects forward (constant lateral velocity, gravity arc vertically,
        drag-based forward progress) out to max_forward_distance_m.

        Returned as [x, y, z] metres, relative to release - same convention as
        pixel_to_xyz, so the two can be compared frame-for-frame to get real
        swing (pixel_to_xyz result minus this).

        NOTE: if the ball genuinely starts swinging from release rather than
        developing swing later in the flight, some of that early curvature
        gets folded into the fitted velocity here, causing a systematic
        UNDERESTIMATE of total swing later on. A smaller FRAMES_FIT_SEAMLESS
        reduces this bias at the cost of more pixel-noise sensitivity - the
        two errors trade off against each other rather than one dominating.
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

        # x(t) = vx*t (no lateral force yet) - least squares through the origin
        vx_m_s = sum(t * x for t, x in zip(ts, xs)) / sum(t * t for t in ts)
        # z(t) = vz*t + 0.5*g*t^2 - subtract the known gravity term, fit vz through origin
        vz_m_s = sum(t * (z - 0.5 * GRAVITY * t ** 2) for t, z in zip(ts, zs)) / sum(t * t for t in ts)

        initial_speed_m_s = initial_speed_km_h / 3.6
        trajectory: list[list[float]] = []
        frame_offset = 0
        while True:
            elapsed_s = frame_offset * seconds_per_frame
            y_m = self.forward_distance_at_time(elapsed_s, initial_speed_m_s, drag_coefficient)
            if y_m > max_forward_distance_m:
                break

            x_m = vx_m_s * elapsed_s
            z_m = vz_m_s * elapsed_s + 0.5 * GRAVITY * elapsed_s ** 2

            trajectory.append([x_m, y_m, z_m])
            frame_offset += 1

        return trajectory